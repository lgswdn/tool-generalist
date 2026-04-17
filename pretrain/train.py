"""train.py — Training script for SDFSegmentor geometry encoder pretraining.

SDFSegmentor uses a joint ViT encoder (PointNet patch encoder + joint ViT
transformer) that processes tool and object clouds together, enabling
implicit cross-stream attention before SDF prediction.

Usage:
    # Point-level SDF (default)
    python train.py --data-dir tmp_data/

    # Patch-level SDF with larger ViT
    python train.py --data-dir tmp_data/ --head-mode patch --patch-agg mean \\
        --vit-depth 6 --vit-heads 8

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py --data-dir tmp_data/

    # Resume
    python train.py --data-dir tmp_data/ --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import make_split
from model import SDFSegmentor


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp() -> tuple[int, int]:
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #

def save_ckpt(path: Path, model: torch.nn.Module, optimizer, epoch: int,
              best_val: float):
    torch.save({
        "epoch":    epoch,
        "best_val": best_val,
        "model":    (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_ckpt(path: str, model: torch.nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def run_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer,
    device:    torch.device,
    train:     bool,
) -> tuple[float, dict]:
    model.train(train)
    total_loss = 0.0
    agg: dict[str, float] = {}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            # ---- Inputs: world-frame point clouds -------------------------
            tool_pc = batch["tool_pc"].to(device)   # (B, N, 3)
            obj_pc  = batch["obj_pc"].to(device)    # (B, N, 3)

            # ---- GT: per-point signed SDF --------------------------------
            tool_sdf_gt = batch["tool_pts_sdf"].to(device)  # (B, N)
            obj_sdf_gt  = batch["obj_pts_sdf"].to(device)   # (B, N)

            # Forward + loss
            m = model.module if isinstance(model, DDP) else model
            loss, metrics = m.loss(tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

    avg = {k: v / max(n, 1) for k, v in agg.items()}
    return total_loss / max(n, 1), avg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir",   default="tmp_data",
                        help="Root dir containing .pt contact files")
    parser.add_argument("--out-dir",    default="checkpoints",
                        help="Where to save checkpoints")
    parser.add_argument("--val-ratio",  type=float, default=0.1)

    # Training
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--resume",      default="",
                        help="Checkpoint path to resume from")

    # Logging
    parser.add_argument("--wandb",       action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="sdf-segmentor",
                        help="W&B project name")
    parser.add_argument("--wandb-name",  default=None,
                        help="W&B run name (default: auto-generated)")

    # Encoder
    parser.add_argument("--num-pts",          type=int, default=512,
                        help="Points per cloud (N)")
    parser.add_argument("--encoder-channel",  type=int, default=128,
                        help="Patch token dimension D")
    parser.add_argument("--patch-size",       type=int, default=32,
                        help="Points per FPS patch (K)")
    parser.add_argument("--vit-depth",        type=int, default=4,
                        help="Number of ViT transformer layers")
    parser.add_argument("--vit-heads",        type=int, default=4,
                        help="Number of ViT attention heads")
    parser.add_argument("--freeze-encoder",   action="store_true",
                        help="Freeze the ViT encoder (train SDF heads only)")

    # SDF head
    parser.add_argument("--head-mode",   default="point",
                        choices=["point", "patch"],
                        help="'point': per-point SDF  |  'patch': per-patch SDF")
    parser.add_argument("--patch-agg",   default="mean",
                        choices=["mean", "min", "max"],
                        help="GT aggregation for patch mode")

    args = parser.parse_args()

    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- Data ----------------------------------------------------------------
    train_ds, val_ds = make_split(args.data_dir, val_ratio=args.val_ratio)
    if is_main():
        print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader  = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ---------------------------------------------------------------
    model = SDFSegmentor(
        head_mode=args.head_mode,
        patch_agg=args.patch_agg,
        num_pts=args.num_pts,
        patch_size=args.patch_size,
        encoder_channel=args.encoder_channel,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    if is_main():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: SDFSegmentor  head={args.head_mode}  "
              f"vit_depth={args.vit_depth}  vit_heads={args.vit_heads}  "
              f"trainable params: {total_params:,}")

        # Initialize wandb
        if args.wandb:
            if not HAS_WANDB:
                raise RuntimeError("--wandb requires wandb to be installed. Run: pip install wandb")
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
            wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )


    # ---- Resume --------------------------------------------------------------
    start_epoch = 0
    best_val    = float("inf")
    if args.resume:
        start_epoch, best_val = load_ckpt(args.resume, model, optimizer)
        if is_main():
            print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ---- Train ---------------------------------------------------------------
    out_dir = Path(args.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss,   val_m   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        if is_main():
            dt = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1:04d}/{args.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"tool_sdf={val_m.get('tool_sdf_loss', float('nan')):.4f} "
                f"(raw={val_m.get('tool_sdf_loss_raw', float('nan')):.5f} "
                f"scale={val_m.get('tool_scale', float('nan')):.4f})  "
                f"obj_sdf={val_m.get('obj_sdf_loss', float('nan')):.4f} "
                f"(raw={val_m.get('obj_sdf_loss_raw', float('nan')):.5f} "
                f"scale={val_m.get('obj_scale', float('nan')):.4f})  "
                f"lr={lr:.2e}  t={dt:.1f}s"
            )

            # Wandb logging
            if args.wandb and HAS_WANDB:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/tool_sdf_loss": val_m.get('tool_sdf_loss', float('nan')),
                    "val/obj_sdf_loss": val_m.get('obj_sdf_loss', float('nan')),
                    "train/tool_sdf_loss": train_m.get('tool_sdf_loss', float('nan')),
                    "train/obj_sdf_loss": train_m.get('obj_sdf_loss', float('nan')),
                    "lr": lr,
                    "time": dt,
                })

            save_ckpt(out_dir / "last.pt", model, optimizer, epoch + 1, best_val)

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(out_dir / "best.pt", model, optimizer, epoch + 1, best_val)
                print(f"  ✓ New best val: {best_val:.5f}")

    # Finish wandb run
    if is_main() and args.wandb:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

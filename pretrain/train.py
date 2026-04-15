"""train.py — Training script for the geometry encoder (ContactPredictor).

Usage:
    # Single GPU
    python train.py --data-dir tmp_data/

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py --data-dir tmp_data/ --gpus 2 3

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

# Allow running from the pretrain/ directory directly
_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import make_split, ContactDataset, collect_pt_files
from model import ContactPredictor


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp() -> tuple[int, int]:
    """Initialize DDP from torchrun env vars. Returns (rank, local_rank)."""
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
        "epoch": epoch,
        "best_val": best_val,
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
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

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, dict]:
    model.train(train)
    total_loss = 0.0
    agg: dict[str, float] = {}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            tool_pc     = batch["tool_pc"].to(device)
            object_pc   = batch["object_pc"].to(device)
            contact_gt  = batch["contact_pts"].to(device)
            normals_gt  = batch["contact_normals"].to(device)

            m = model.module if isinstance(model, DDP) else model
            loss, metrics = m.loss(tool_pc, object_pc, contact_gt, normals_gt)

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
    parser.add_argument("--data-dir",    default="tmp_data",
                        help="Root dir containing .pt files")
    parser.add_argument("--out-dir",     default="checkpoints",
                        help="Where to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--resume",      default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--icp-weights", default="",
                        help="Optional pretrained ICPNet weights to warm-start")
    parser.add_argument("--freeze-icp",  action="store_true",
                        help="Freeze the ICP encoder (train head only)")
    # ICPNet architecture
    parser.add_argument("--num-pts",          type=int, default=512)
    parser.add_argument("--encoder-channel",  type=int, default=128)
    parser.add_argument("--patch-size",       type=int, default=32)
    args = parser.parse_args()

    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- Data ----------------------------------------------------------------
    train_ds, val_ds = make_split(
        args.data_dir,
        val_ratio=args.val_ratio,
        num_tool_pts=args.num_pts,
        num_obj_pts=args.num_pts,
    )
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
    model = ContactPredictor(
        num_contact_pts=5,
        icp_weights_path=args.icp_weights or None,
        freeze_icp=args.freeze_icp,
        num_pts=args.num_pts,
        patch_size=args.patch_size,
        encoder_channel=args.encoder_channel,
    ).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

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
    best_val = float("inf")
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
            chamfer_v = val_m.get("chamfer", float("nan"))
            normal_v  = val_m.get("normal_cos", float("nan"))
            print(f"Epoch {epoch+1:04d}/{args.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"chamfer={chamfer_v:.5f}  normal_cos={normal_v:.5f}  "
                  f"lr={lr:.2e}  t={dt:.1f}s")

            # Save last checkpoint
            save_ckpt(out_dir / "last.pt", model, optimizer, epoch + 1, best_val)

            # Save best
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(out_dir / "best.pt", model, optimizer, epoch + 1, best_val)
                print(f"  ✓ New best val: {best_val:.5f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

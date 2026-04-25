"""train.py — Training script for joint SDF + Diffusion encoder pretraining.

Proven recipe (from debug_overfit.py experiments):
  1. MLPNoisePredictor (bypasses cross-attention, faster for horizon=1)
  2. Regression warmup (prevents encoder posterior collapse)
  3. Auxiliary regression (keeps encoder discriminative during joint training)
  4. LR 1e-3

Usage:
    # Single GPU, default config
    python train.py --data-dir /path/to/data/

    # Distributed
    torchrun --nproc_per_node=2 train.py --data-dir /path/to/data/

    # Limit to N .pt files (for quick tests)
    python train.py --data-dir /path/to/data/ --max-files 1

    # Resume from checkpoint
    python train.py --data-dir /path/to/data/ --resume checkpoints/best.pt

    # Wandb logging
    python train.py --data-dir /path/to/data/ --wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

from config import TrainConfig, DEFAULT_CONFIG
from dataset import make_split
from model import SDFSegmentor, JointModel


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

def save_ckpt(path: Path, model: torch.nn.Module, optimizer, epoch: int, best_val: float):
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
    warmup:    bool = False,
    scaler:    torch.cuda.amp.GradScaler = None,
) -> tuple[float, dict]:
    """Run one epoch of training or validation.

    Args:
        warmup: If True, skip diffusion loss (regression-only phase).
    """
    model.train(train)
    total_loss = 0.0
    agg: dict[str, float] = {}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            tool_pc = batch["tool_pc"].to(device)
            obj_pc  = batch["obj_pc"].to(device)
            tool_sdf_gt = batch["tool_pts_sdf"].to(device)
            obj_sdf_gt  = batch["obj_pts_sdf"].to(device)

            # Diffusion inputs (optional)
            tool_pc_init = batch.get("tool_pc_init")
            delta_pose  = batch.get("delta_pose")
            init_pose   = batch.get("init_pose")

            if tool_pc_init is not None:
                if isinstance(tool_pc_init, list):
                    if any(t is None for t in tool_pc_init):
                        tool_pc_init = None
                    else:
                        tool_pc_init = torch.stack(tool_pc_init).to(device)
                else:
                    tool_pc_init = tool_pc_init.to(device)
            if delta_pose is not None:
                if isinstance(delta_pose, list):
                    if any(d is None for d in delta_pose):
                        delta_pose = None
                    else:
                        delta_pose = torch.stack(delta_pose).to(device)
                else:
                    delta_pose = delta_pose.to(device)
            if init_pose is not None:
                if isinstance(init_pose, list):
                    if any(p is None for p in init_pose):
                        init_pose = None
                    else:
                        init_pose = torch.stack(init_pose).to(device)
                else:
                    init_pose = init_pose.to(device)

            m = model.module if isinstance(model, DDP) else model

            # Forward + loss
            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    loss, metrics = m.loss(
                        tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                        tool_pc_init=tool_pc_init, delta_pose_gt=delta_pose,
                        init_pose_gt=init_pose, warmup=warmup,
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, metrics = m.loss(
                    tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                    tool_pc_init=tool_pc_init, delta_pose_gt=delta_pose,
                    init_pose_gt=init_pose, warmup=warmup,
                )
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
    parser = argparse.ArgumentParser(description="Train SDF + Diffusion encoder")
    parser.add_argument("--data-dir", required=True, help="Data directory with .pt files")
    parser.add_argument("--config", default="config.py", help="Path to custom config.py")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit number of .pt files (0=all)")
    parser.add_argument("--lr", type=float, default=0,
                        help="Override learning rate (0=use config)")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override epochs (0=use config)")
    parser.add_argument("--warmup-epochs", type=int, default=-1,
                        help="Override warmup epochs (-1=use config)")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        cfg = cfg_module.TrainConfig()
    else:
        cfg = DEFAULT_CONFIG

    # Override from CLI
    cfg.data_dir = args.data_dir
    if args.resume:
        cfg.resume = args.resume
    if args.wandb:
        cfg.wandb = True
    if args.no_amp:
        cfg.amp = False
    if args.max_files > 0:
        cfg.max_files = args.max_files
    if args.lr > 0:
        cfg.lr = args.lr
    if args.epochs > 0:
        cfg.epochs = args.epochs
    if args.warmup_epochs >= 0:
        cfg.warmup_epochs = args.warmup_epochs

    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Linear LR scaling: keep effective LR/sample constant across GPU counts
    effective_lr = cfg.lr * world_size
    effective_batch = cfg.batch_size * world_size

    # ---- Data ----
    train_ds, val_ds = make_split(
        cfg.data_dir, val_ratio=cfg.val_ratio, max_files=cfg.max_files,
    )
    if is_main():
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  GPUs: {world_size}")
        print(f"Config: diffusion={cfg.diffusion}, mlp_head={cfg.use_mlp_head}, "
              f"aux_reg={cfg.aux_reg}, warmup={cfg.warmup_epochs}ep")
        print(f"        amp={cfg.amp}, batch/gpu={cfg.batch_size}, "
              f"effective_batch={effective_batch}, "
              f"base_lr={cfg.lr:.1e}, effective_lr={effective_lr:.1e}, "
              f"epochs={cfg.epochs}")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
    )

    # ---- Model ----
    if cfg.diffusion:
        model = JointModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=cfg.freeze_encoder,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_emb=cfg.n_emb,
            p_drop_emb=cfg.p_drop_emb,
            p_drop_attn=cfg.p_drop_attn,
            use_mlp_head=cfg.use_mlp_head,
            aux_reg=cfg.aux_reg,
            sdf_weight=cfg.sdf_weight,
            diffusion_weight=cfg.diffusion_weight,
            aux_weight=cfg.aux_weight,
        ).to(device)
        model_name = "JointModel"
    else:
        model = SDFSegmentor(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=cfg.freeze_encoder,
        ).to(device)
        model_name = "SDFSegmentor"

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    if is_main():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_name}  trainable params: {total_params:,}")

        if cfg.wandb:
            if not HAS_WANDB:
                raise RuntimeError("--wandb requires wandb installed")
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=vars(cfg))
            wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=effective_lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler() if cfg.amp else None
    if cfg.amp and is_main():
        print("Using AMP")

    # ---- Resume ----
    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, best_val = load_ckpt(cfg.resume, model, optimizer)
        if is_main():
            print(f"Resumed from {cfg.resume} (epoch {start_epoch})")

    # ---- Train ----
    out_dir = Path(cfg.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Determine phase
        in_warmup = cfg.diffusion and epoch < cfg.warmup_epochs

        t0 = time.time()
        train_loss, train_m = run_epoch(
            model, train_loader, optimizer, device,
            train=True, warmup=in_warmup, scaler=scaler,
        )

        # Validate every 20 epochs
        val_freq = 20
        do_val = (epoch + 1) % val_freq == 0 or epoch == cfg.epochs - 1
        if do_val:
            val_loss, val_m = run_epoch(
                model, val_loader, optimizer, device,
                train=False, warmup=in_warmup, scaler=scaler,
            )
        scheduler.step()

        if is_main():
            dt = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            phase = "[WARMUP]" if in_warmup else "[JOINT] "

            # Print metrics
            metric_keys = ["tool_sdf_loss", "obj_sdf_loss", "diffusion_loss", "aux_loss"]
            if do_val:
                print_str = f"{phase} Epoch {epoch+1:04d}/{cfg.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
                for k in metric_keys:
                    if k in val_m:
                        short = k.replace("_loss", "").replace("_", "")
                        print_str += f"{short}={val_m[k]:.4f}  "
            else:
                print_str = f"{phase} Epoch {epoch+1:04d}/{cfg.epochs}  train={train_loss:.4f}  "
                for k in metric_keys:
                    if k in train_m:
                        short = k.replace("_loss", "").replace("_", "")
                        print_str += f"{short}={train_m[k]:.4f}  "
            print_str += f"lr={lr:.2e}  t={dt:.1f}s"
            print(print_str)

            # Log warmup→joint transition
            if epoch == cfg.warmup_epochs and cfg.warmup_epochs > 0:
                print(f"\n{'='*60}")
                print(f">>> WARMUP COMPLETE — switching to joint training")
                print(f"{'='*60}\n")

            # Wandb
            if cfg.wandb and HAS_WANDB:
                log_dict = {"epoch": epoch+1, "train/loss": train_loss, "lr": lr,
                            "time": dt, "phase": 0 if in_warmup else 1}
                if do_val:
                    log_dict["val/loss"] = val_loss
                    for k, v in val_m.items():
                        log_dict[f"val/{k}"] = v
                for k, v in train_m.items():
                    log_dict[f"train/{k}"] = v
                wandb.log(log_dict)

            save_ckpt(out_dir / "last.pt", model, optimizer, epoch+1, best_val)
            check_loss = val_loss if do_val else train_loss
            if check_loss < best_val:
                best_val = check_loss
                save_ckpt(out_dir / "best.pt", model, optimizer, epoch+1, best_val)
                print(f"  ✓ New best: {best_val:.5f}")

    if is_main() and cfg.wandb:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
"""train.py — Training loop for RPDiff-style joint SDF + pose denoising.

Follows the existing pretrain/train.py DDP pattern. Supports:
  - --task sdf        → SDF-only (contact pose)
  - --task sdf-diff   → Joint SDF + diffusion
  - --head-mode point|patch
  - wandb logging, DDP, checkpointing

Usage:
    # SDF-only, patch mode
    python pretrain/new_pretrain/train.py --data-dir /path/to/data --task sdf --head-mode patch

    # Joint SDF + diffusion, point mode
    python pretrain/new_pretrain/train.py --data-dir /path/to/data --task sdf-diff --head-mode point

    # Distributed
    torchrun --nproc_per_node=4 pretrain/new_pretrain/train.py --data-dir /path/to/data

    # Quick test
    python pretrain/new_pretrain/train.py --data-dir /path/to/data --max-files 2 --epochs 5 --batch-size 8
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

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PRETRAIN_DIR = _THIS_DIR.parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
_RPDIFF_SRC   = _PRETRAIN_DIR / "rpdiff" / "src"

for p in [str(_REPO_ROOT), str(_PRETRAIN_DIR), str(_RPDIFF_SRC), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import NewPretrainConfig
from dataset import make_split
from model import ContactDiffusionModel
from noise_utils import sample_noised_poses_batch, compute_on_the_fly_sdf

# ── Reuse RPDiff's quaternion conversion ─────────────────────────────────────
from rpdiff.utils.torch3d_util import matrix_to_quaternion


# ============================================================================ #
# Distributed helpers
# ============================================================================ #

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


# ============================================================================ #
# Checkpoint helpers
# ============================================================================ #

def save_ckpt(path: Path, model: torch.nn.Module, optimizer, epoch: int, best_val: float):
    torch.save({
        "epoch":    epoch,
        "best_val": best_val,
        "model":    (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_ckpt(path: str, model: torch.nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


# ============================================================================ #
# Collate: skip None-valued optional fields
# ============================================================================ #

def collate_fn(batch):
    """Stack tensors from batch dicts, skip None fields."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if vals[0] is None:
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out


# ============================================================================ #
# Training step
# ============================================================================ #

def train_step(
    model: ContactDiffusionModel,
    batch: dict,
    cfg: NewPretrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """One training step: encode → noise → cross-attend → SDF + denoise loss."""

    tool_canonical = batch["tool_canonical"].to(device)   # (B, P, 3)
    obj_pc         = batch["obj_pc"].to(device)           # (B, Q, 3)
    contact_R      = batch["contact_R"].to(device)        # (B, 3, 3)
    contact_t      = batch["contact_t"].to(device)        # (B, 3)
    tool_sdf_gt    = batch["tool_sdf"].to(device)         # (B, P)
    obj_sdf_gt     = batch["obj_sdf"].to(device)          # (B, Q)

    raw_model = model.module if isinstance(model, DDP) else model

    if raw_model.task == "sdf":
        # SDF-only: no noising, use contact pose SDF directly
        # The encoder sees canonical tool + object
        return model(
            tool_canonical=tool_canonical,
            obj_pc=obj_pc,
            tool_sdf_gt=tool_sdf_gt,
            obj_sdf_gt=obj_sdf_gt,
        )

    # ── sdf-diff: sample noised pose and compute losses ──────────────
    B = tool_canonical.shape[0]

    # 1. Sample noised pose (on-the-fly, RPDiff-style)
    noise_out = sample_noised_poses_batch(
        contact_R=contact_R,
        contact_t=contact_t,
        num_steps=cfg.num_diffusion_steps,
        max_trans=cfg.noise_max_trans,
        max_rot_deg=cfg.noise_max_rot_deg,
        interp=cfg.interp_trajectory,
        precise_prob=cfg.precise_diff_prob,
    )

    # 2. Compute on-the-fly SDF at the noised pose
    tool_sdf_noised, obj_sdf_noised = compute_on_the_fly_sdf(
        tool_canonical, obj_pc,
        noise_out["noised_R"], noise_out["noised_t"],
    )

    # For t=0 (contact pose), use the exact signed SDF from dataset
    is_contact = (noise_out["t_idx"] == 0)
    if is_contact.any():
        tool_sdf_noised[is_contact] = tool_sdf_gt[is_contact]
        obj_sdf_noised[is_contact] = obj_sdf_gt[is_contact]

    # 3. Build noised pose 7D: trans(3) + quaternion(4)
    noised_quat = matrix_to_quaternion(noise_out["noised_R"])  # (B, 4)
    noised_pose_7d = torch.cat([noise_out["noised_t"], noised_quat], dim=-1)  # (B, 7)

    # 4. Build child point clouds for chamfer loss
    # child = tool at noised pose
    child_start_pcd = torch.bmm(
        tool_canonical, noise_out["noised_R"].transpose(1, 2)
    ) + noise_out["noised_t"].unsqueeze(1)

    # child_final = where child SHOULD be after denoising one step
    # Apply target (R, t) to child_start
    child_start_mean = child_start_pcd.mean(dim=1, keepdim=True)
    child_centered = child_start_pcd - child_start_mean
    child_rotated = torch.bmm(
        noise_out["target_rot_mat"],
        child_centered.transpose(1, 2)
    ).transpose(1, 2)
    child_final_pcd = child_rotated + child_start_mean + noise_out["target_trans"].unsqueeze(1)

    # 5. Forward
    return model(
        tool_canonical=tool_canonical,
        obj_pc=obj_pc,
        tool_sdf_gt=tool_sdf_noised,
        obj_sdf_gt=obj_sdf_noised,
        noised_pose_7d=noised_pose_7d,
        timestep=noise_out["t_idx"],
        target_trans=noise_out["target_trans"],
        target_rot_mat=noise_out["target_rot_mat"],
        child_start_pcd=child_start_pcd,
        child_final_pcd=child_final_pcd,
    )


# ============================================================================ #
# Main training loop
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="RPDiff-style joint SDF + denoising pretraining")

    # Data
    parser.add_argument("--data-dir",    type=str, required=True)
    parser.add_argument("--max-files",   type=int, default=0)
    parser.add_argument("--val-ratio",   type=float, default=0.1)

    # Task
    parser.add_argument("--task",        type=str, default="sdf-diff", choices=["sdf", "sdf-diff"])
    parser.add_argument("--head-mode",   type=str, default="point", choices=["point", "patch"])
    parser.add_argument("--patch-agg",   type=str, default="mean", choices=["mean", "min", "max"])

    # Encoder
    parser.add_argument("--num-pts",          type=int, default=512)
    parser.add_argument("--patch-size",       type=int, default=32)
    parser.add_argument("--encoder-channel",  type=int, default=128)
    parser.add_argument("--vit-depth",        type=int, default=4)
    parser.add_argument("--vit-heads",        type=int, default=4)
    parser.add_argument("--freeze-encoder",   action="store_true")

    # Cross-attention
    parser.add_argument("--cross-attn-heads",  type=int, default=4)
    parser.add_argument("--cross-attn-layers", type=int, default=2)

    # Diffusion
    parser.add_argument("--num-diffusion-steps", type=int, default=10)
    parser.add_argument("--noise-max-trans",     type=float, default=0.15)
    parser.add_argument("--noise-max-rot-deg",   type=float, default=90.0)
    parser.add_argument("--interp-trajectory",   action="store_true", default=True)
    parser.add_argument("--no-interp-trajectory", dest="interp_trajectory", action="store_false")
    parser.add_argument("--precise-diff-prob",   action="store_true")

    # Loss weights
    parser.add_argument("--sdf-weight",     type=float, default=1.0)
    parser.add_argument("--denoise-weight", type=float, default=1.0)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)

    # Training
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs",       type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)

    # Checkpoint
    parser.add_argument("--resume",   type=str, default="")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_new")

    # Logging
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--wandb-project",  type=str, default="new_pretrain")
    parser.add_argument("--wandb-run-name", type=str, default="")

    args = parser.parse_args()

    # Build config from args
    cfg = NewPretrainConfig(**{k: v for k, v in vars(args).items()
                               if hasattr(NewPretrainConfig, k.replace("-", "_"))})
    # Handle hyphens → underscores
    cfg.data_dir = args.data_dir
    cfg.max_files = args.max_files
    cfg.val_ratio = args.val_ratio
    cfg.task = args.task
    cfg.head_mode = args.head_mode
    cfg.patch_agg = args.patch_agg
    cfg.num_pts = args.num_pts
    cfg.patch_size = args.patch_size
    cfg.encoder_channel = args.encoder_channel
    cfg.vit_depth = args.vit_depth
    cfg.vit_heads = args.vit_heads
    cfg.freeze_encoder = args.freeze_encoder
    cfg.cross_attn_heads = args.cross_attn_heads
    cfg.cross_attn_layers = args.cross_attn_layers
    cfg.num_diffusion_steps = args.num_diffusion_steps
    cfg.noise_max_trans = args.noise_max_trans
    cfg.noise_max_rot_deg = args.noise_max_rot_deg
    cfg.interp_trajectory = args.interp_trajectory
    cfg.precise_diff_prob = args.precise_diff_prob
    cfg.sdf_weight = args.sdf_weight
    cfg.denoise_weight = args.denoise_weight
    cfg.chamfer_weight = args.chamfer_weight
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.epochs = args.epochs
    cfg.log_interval = args.log_interval
    cfg.save_interval = args.save_interval
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.resume = args.resume
    cfg.ckpt_dir = args.ckpt_dir
    cfg.wandb = args.wandb
    cfg.wandb_project = args.wandb_project
    cfg.wandb_run_name = args.wandb_run_name

    # ── Setup ────────────────────────────────────────────────────────────
    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(cfg.seed + rank)

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds, val_ds = make_split(
        data_dir=cfg.data_dir,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        augment=cfg.augment,
        max_files=cfg.max_files,
    )

    if is_main():
        print(f"Train: {len(train_ds)} configs, Val: {len(val_ds)} configs")
        print(f"Task: {cfg.task}, Head: {cfg.head_mode}, Diffusion steps: {cfg.num_diffusion_steps}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = ContactDiffusionModel(
        head_mode=cfg.head_mode,
        patch_agg=cfg.patch_agg,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=cfg.freeze_encoder,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        denoise_hidden=256,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        chamfer_weight=cfg.chamfer_weight,
        num_diffusion_steps=cfg.num_diffusion_steps,
        task=cfg.task,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, best_val = load_ckpt(cfg.resume, model, optimizer)
        if is_main():
            print(f"Resumed from {cfg.resume} at epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Wandb ────────────────────────────────────────────────────────────
    if cfg.wandb and HAS_WANDB and is_main():
        run_name = cfg.wandb_run_name or f"{cfg.task}_{cfg.head_mode}_T{cfg.num_diffusion_steps}"
        wandb.init(project=cfg.wandb_project, name=run_name, config=vars(cfg))

    # ── Checkpoint directory ─────────────────────────────────────────────
    ckpt_dir = Path(cfg.ckpt_dir)
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_dl):
            loss, metrics = train_step(model, batch, cfg, device)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

            if is_main() and (batch_idx + 1) % cfg.log_interval == 0:
                avg = {k: v / n_batches for k, v in epoch_metrics.items()}
                print(f"  [{epoch+1}/{cfg.epochs}] batch {batch_idx+1}/{len(train_dl)} "
                      f"loss={loss.item():.6f} grad_norm={grad_norm:.4f} "
                      + " ".join(f"{k}={v:.6f}" for k, v in avg.items()))

        # ── Epoch summary ────────────────────────────────────────────────
        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_train["epoch_loss"] = epoch_loss / max(n_batches, 1)
        avg_train["epoch_time"] = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        n_val = 0

        with torch.no_grad():
            for batch in val_dl:
                loss, metrics = train_step(model, batch, cfg, device)
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
                n_val += 1

        avg_val = {f"val_{k}": v / max(n_val, 1) for k, v in val_metrics.items()}
        avg_val["val_loss"] = val_loss / max(n_val, 1)

        if is_main():
            print(f"Epoch {epoch+1}/{cfg.epochs} — "
                  f"train_loss={avg_train['epoch_loss']:.6f} "
                  f"val_loss={avg_val['val_loss']:.6f} "
                  f"time={avg_train['epoch_time']:.1f}s")

            # Log to wandb
            if cfg.wandb and HAS_WANDB:
                log_dict = {**avg_train, **avg_val, "epoch": epoch + 1}
                wandb.log(log_dict)

            # Save checkpoint
            if (epoch + 1) % cfg.save_interval == 0:
                save_ckpt(ckpt_dir / f"epoch_{epoch+1}.pt", model, optimizer, epoch + 1, best_val)

            if avg_val["val_loss"] < best_val:
                best_val = avg_val["val_loss"]
                save_ckpt(ckpt_dir / "best.pt", model, optimizer, epoch + 1, best_val)
                print(f"  → New best val_loss: {best_val:.6f}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()
    if is_main():
        print("Training complete.")


if __name__ == "__main__":
    main()

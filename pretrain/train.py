"""train.py — Training script for joint SDF + Flow Matching encoder pretraining.

Proven recipe (from debug_overfit.py experiments):
  1. MLPVelocityNet (bypasses cross-attention, faster for horizon=1)
  2. Regression warmup (prevents encoder posterior collapse)
  3. Auxiliary regression (keeps encoder discriminative during joint training)
  4. LR 5e-4

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

    # Movement-only training
    python train.py --data-dir /path/to/data/ --task movement
"""

from __future__ import annotations

import argparse
import collections
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
from model import SDFSegmentor, JointModel, MovementModel


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
# Safe-to-device helper
# --------------------------------------------------------------------------- #

def _safe_to_device(val, device):
    """Handle None, list-of-None, or tensor → device."""
    if val is None:
        return None
    if isinstance(val, list):
        if any(v is None for v in val):
            return None
        return torch.stack(val).to(device)
    return val.to(device)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def run_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer,
    device:    torch.device,
    train:     bool,
    scaler:    torch.cuda.amp.GradScaler = None,
    enable_flow: bool = True,
) -> tuple[float, dict]:
    """Run one epoch of training or validation."""
    model.train(train)
    total_loss = 0.0
    agg: dict[str, float] = {}
    n = 0
    _debug_history = collections.deque(maxlen=10)  # last 10 batches for NaN diagnosis
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            tool_pc = batch["tool_pc"].to(device)
            obj_pc  = batch["obj_pc"].to(device)
            tool_sdf_gt = batch["tool_pts_sdf"].to(device)
            obj_sdf_gt  = batch["obj_pts_sdf"].to(device)

            # Initial pose SDF (optional)
            init_tool_sdf_gt = _safe_to_device(batch.get("init_tool_pts_sdf"), device)
            init_obj_sdf_gt = _safe_to_device(batch.get("init_obj_pts_sdf"), device)

            # Flow matching inputs (optional)
            tool_pc_init = _safe_to_device(batch.get("tool_pc_init"), device)
            delta_pose = _safe_to_device(batch.get("delta_pose"), device)
            init_pose = _safe_to_device(batch.get("init_pose"), device)

            # Movement inputs (optional)
            obj_point_displacement = _safe_to_device(batch.get("obj_point_displacement"), device)
            tool_delta_pose = _safe_to_device(batch.get("tool_delta_pose"), device)

            # Forward — call model() (not model.module.loss()) so DDP hooks fire
            # Build kwargs based on model type
            fwd_kwargs = {}
            raw_model = model.module if isinstance(model, DDP) else model
            if isinstance(raw_model, MovementModel):
                fwd_kwargs["tool_delta_action"] = tool_delta_pose
                fwd_kwargs["obj_displacement_gt"] = obj_point_displacement
            elif isinstance(raw_model, JointModel):
                fwd_kwargs["tool_pc_init"] = tool_pc_init
                fwd_kwargs["delta_pose_gt"] = delta_pose
                fwd_kwargs["init_pose_gt"] = init_pose
                fwd_kwargs["enable_flow"] = enable_flow
                fwd_kwargs["init_tool_sdf_gt"] = init_tool_sdf_gt
                fwd_kwargs["init_obj_sdf_gt"] = init_obj_sdf_gt
                fwd_kwargs["obj_point_displacement"] = obj_point_displacement
                fwd_kwargs["tool_delta_pose"] = tool_delta_pose

            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    loss, metrics = model(
                        tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                        **fwd_kwargs,
                    )
            else:
                loss, metrics = model(
                    tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
                    **fwd_kwargs,
                )

            # NaN guard: dump context and STOP on first NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main():
                    print(f"\n{'='*60}")
                    print(f"⚠ FIRST NaN at batch {n} — dumping last {len(_debug_history)} good batches:")
                    for entry in _debug_history:
                        print(f"  {entry}")
                    print(f"{'='*60}")
                    print(f"NaN batch data stats:")
                    if delta_pose is not None:
                        print(f"  delta_pose: nan={torch.isnan(delta_pose).sum()} "
                              f"inf={torch.isinf(delta_pose).sum()} "
                              f"abs_max={delta_pose.abs().max():.4f} "
                              f"mean={delta_pose.mean():.4f} std={delta_pose.std():.4f}")
                    if tool_pc_init is not None:
                        print(f"  tool_pc_init: nan={torch.isnan(tool_pc_init).sum()} "
                              f"abs_max={tool_pc_init.abs().max():.4f}")
                    print(f"  tool_pc: abs_max={tool_pc.abs().max():.4f}")
                    print(f"  obj_pc:  abs_max={obj_pc.abs().max():.4f}")
                    # Which params went NaN?
                    nan_params = []
                    ok_params_max = []
                    for pname, p in raw_model.named_parameters():
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            nan_params.append(pname)
                        elif p.requires_grad:
                            ok_params_max.append((pname, p.abs().max().item()))
                    if nan_params:
                        print(f"  NaN params ({len(nan_params)}):")
                        for pn in nan_params:
                            print(f"    {pn}")
                    ok_params_max.sort(key=lambda x: -x[1])
                    print(f"  Top-5 largest OK param abs values:")
                    for pn, v in ok_params_max[:5]:
                        print(f"    {pn}: {v:.4f}")
                raise RuntimeError("NaN detected — stopping for diagnosis")

            # Backward + step
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # Sanitize: zero out any NaN/Inf per-parameter grads
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Sanitize: zero out any NaN/Inf per-parameter grads
                    n_sanitized = 0
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                            n_sanitized += 1
                    if n_sanitized > 0 and is_main():
                        print(f"  ⚠ sanitized {n_sanitized} param grads at batch {n}")
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            else:
                grad_norm = 0.0

            # Record history (keep last 10 batches for NaN diagnosis)
            if is_main():
                dp_max = delta_pose.abs().max().item() if delta_pose is not None else 0
                entry = (f"batch={n:03d} loss={loss.item():.4f} "
                         f"flow={metrics.get('flow_loss', 0):.4f} "
                         f"sdf_t={metrics.get('tool_sdf_loss', 0):.4f} "
                         f"sdf_o={metrics.get('obj_sdf_loss', 0):.4f} "
                         f"aux={metrics.get('aux_loss', 0):.4f} "
                         f"mvmt={metrics.get('movement_loss', 0):.4f} "
                         f"grad={grad_norm:.2f} dp_max={dp_max:.3f}")
                _debug_history.append(entry)

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
    parser = argparse.ArgumentParser(description="Train SDF + Flow Matching encoder")
    parser.add_argument("--data-dir", required=True, help="Data directory with .pt files")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Override checkpoint output directory")
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
    parser.add_argument("--total-steps", type=int, default=0,
                        help="Fixed step budget (overrides epochs; 0=use epochs)")
    parser.add_argument("--warmup-epochs", type=int, default=-1,
                        help="Override warmup epochs (-1=use config)")
    parser.add_argument("--task", type=str, default="",
                        choices=["", "joint", "movement", "sdf"],
                        help="Task: 'joint' (SDF+flow), 'movement' (SDF+movement), 'sdf' (SDF-only)")
    parser.add_argument("--head-mode", type=str, default="",
                        choices=["", "point", "patch"],
                        help="SDF head mode (default from config)")
    parser.add_argument("--patch-agg", type=str, default="",
                        choices=["", "mean", "min", "max"],
                        help="Patch aggregation for SDF (default from config)")
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
    if args.out_dir:
        cfg.out_dir = args.out_dir
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
    if args.total_steps > 0:
        cfg.total_steps = args.total_steps
    if args.warmup_epochs >= 0:
        cfg.warmup_epochs = args.warmup_epochs
    if args.task:
        cfg.task = args.task
    if args.head_mode:
        cfg.head_mode = args.head_mode
    if args.patch_agg:
        cfg.patch_agg = args.patch_agg
    # task shortcut: 'sdf' disables diffusion
    if cfg.task == "sdf":
        cfg.diffusion = False

    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # No LR scaling needed: batch_size is the GLOBAL batch size,
    # split evenly across GPUs so each step is identical to 1-GPU training.
    effective_lr = cfg.lr
    per_gpu_batch = cfg.batch_size // world_size

    # ---- Data ----
    train_ds, val_ds = make_split(
        cfg.data_dir, val_ratio=cfg.val_ratio, max_files=cfg.max_files,
    )
    if is_main():
        skipped = train_ds.get_skipped_files() + val_ds.get_skipped_files()
        if skipped:
            print(f"⚠ Skipped {len(skipped)} corrupted files:")
            for f in skipped:
                print(f"    {f}")
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  GPUs: {world_size}")
        print(f"Config: task={cfg.task}, head_mode={cfg.head_mode}, "
              f"patch_agg={cfg.patch_agg}, diffusion={cfg.diffusion}")
        print(f"        amp={cfg.amp}, global_batch={cfg.batch_size}, "
              f"per_gpu_batch={per_gpu_batch}, "
              f"lr={effective_lr:.1e}, epochs={cfg.epochs}")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds, batch_size=per_gpu_batch,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=per_gpu_batch,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
    )

    # ---- Model ----
    if cfg.task == "movement":
        model = MovementModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=cfg.freeze_encoder,
            movement_n_heads=cfg.movement_n_heads,
            sdf_weight=cfg.sdf_weight,
            movement_weight=cfg.movement_weight,
        ).to(device)
        model_name = "MovementModel"
    elif cfg.diffusion:
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
            movement_pred=cfg.movement_pred,
            movement_n_heads=cfg.movement_n_heads,
            sdf_weight=cfg.sdf_weight,
            diffusion_weight=cfg.diffusion_weight,
            aux_weight=cfg.aux_weight,
            movement_weight=cfg.movement_weight,
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
    # Per-epoch cosine LR decay.
    # When total_steps is set, epochs is derived so training budget
    # auto-adjusts to dataset size.
    steps_per_epoch = len(train_loader)
    if cfg.total_steps > 0:
        cfg.epochs = (cfg.total_steps + steps_per_epoch - 1) // steps_per_epoch  # ceil
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6,
    )
    if is_main():
        print(f"        steps/epoch={steps_per_epoch}, epochs={cfg.epochs}")

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

        # 2-phase protocol: warmup (SDF+aux only) → joint (SDF+aux+flow)
        enable_flow = (epoch >= cfg.warmup_epochs)

        t0 = time.time()
        train_loss, train_m = run_epoch(
            model, train_loader, optimizer, device,
            train=True, scaler=scaler, enable_flow=enable_flow,
        )

        # Validate every 20 epochs
        val_freq = 20
        do_val = (epoch + 1) % val_freq == 0 or epoch == cfg.epochs - 1
        if do_val:
            val_loss, val_m = run_epoch(
                model, val_loader, optimizer, device,
                train=False, scaler=scaler, enable_flow=enable_flow,
            )
        scheduler.step()

        if is_main():
            dt = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            phase = "[WARMUP]" if not enable_flow else "[JOINT] "

            # Print metrics
            metric_keys = ["tool_sdf_loss", "obj_sdf_loss",
                           "init_tool_sdf_loss", "init_obj_sdf_loss",
                           "flow_loss", "aux_loss", "movement_loss"]
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

            # Wandb
            if cfg.wandb and HAS_WANDB:
                log_dict = {"epoch": epoch+1, "train/loss": train_loss, "lr": lr,
                            "time": dt}
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
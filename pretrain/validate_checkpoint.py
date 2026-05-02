"""validate_checkpoint.py — Validate a checkpoint against a task and compute loss.

Usage:
    # Validate a movement checkpoint (auto-detects head_mode)
    python validate_checkpoint.py --task movement --checkpoint /mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_movement/best.pt

    # Validate an SDF-point checkpoint
    python validate_checkpoint.py --task sdf --head-mode point --checkpoint /mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_point/best.pt

    # Validate an SDF-patch checkpoint
    python validate_checkpoint.py --task sdf --head-mode patch --checkpoint /mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_patch/best.pt

    # Validate a diffusion checkpoint
    python validate_checkpoint.py --task diffusion --checkpoint /path/to/best.pt

    # Override data directory
    python validate_checkpoint.py --task sdf --checkpoint /path/to/best.pt \
        --data-dir /path/to/data/

    # Limit number of batches (for quick check)
    python validate_checkpoint.py --task sdf --checkpoint /path/to/best.pt \
        --max-batches 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAIN_DIR))

from config import TrainConfig
from dataset import make_split
from model import SDFSegmentor, DiffusionModel, MovementModel, JointModel


# --------------------------------------------------------------------------- #
# Task → expected model class mapping
# --------------------------------------------------------------------------- #

TASK_MODEL_MAP = {
    "sdf":       SDFSegmentor,
    "diffusion": DiffusionModel,
    "translation": DiffusionModel,
    "movement":  MovementModel,
    "joint":     JointModel,
}

# Discriminating keys: if a state_dict contains these, it's that model type
_DISCRIMINATORS = {
    "movement_head.mlp.0.weight":     "movement",
    "movement_head.xyz_embed.0.weight": "movement",
    "velocity_net.mlp.0.weight":      "diffusion",
    "noise_predictor.mlp.0.weight":   "diffusion",
    "cond_pos_proj.weight":           "diffusion",
    "aux_reg_head.head.0.weight":     "diffusion",
}


def detect_task_from_state_dict(sd: dict) -> str:
    """Infer task type from checkpoint state_dict keys."""
    has_movement = "movement_head.mlp.0.weight" in sd or "movement_head.xyz_embed.0.weight" in sd
    has_diffusion = (
        "velocity_net.mlp.0.weight" in sd
        or "noise_predictor.mlp.0.weight" in sd
        or "cond_pos_proj.weight" in sd
        or "aux_reg_head.head.0.weight" in sd
    )
    if has_movement and has_diffusion:
        return "joint"
    if "velocity_net.input_proj.0.weight" in sd:
        if sd["velocity_net.input_proj.0.weight"].shape[1] == 3:
            return "translation"
    for key, task in _DISCRIMINATORS.items():
        if key in sd:
            return task
    # No diffusion or movement keys → plain SDF
    return "sdf"


def detect_head_mode_from_state_dict(sd: dict) -> str:
    """Infer head_mode (point vs patch) from checkpoint state_dict keys.

    Point mode has xyz_embed layers; patch mode does not.
    Checks both standalone SDFSegmentor and nested sdf_head (in Joint/Movement models).
    """
    point_keys = [
        "xyz_embed.0.weight",           # standalone SDFSegmentor
        "sdf_head.xyz_embed.0.weight",  # nested in DiffusionModel/MovementModel
    ]
    for k in point_keys:
        if k in sd:
            return "point"
    return "patch"


def build_model(task: str, cfg: TrainConfig, device: torch.device, head_mode: str = "patch") -> torch.nn.Module:
    """Construct the model for the given task and head_mode."""
    cfg.head_mode = head_mode
    if task == "joint":
        return JointModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=False,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_emb=cfg.n_emb,
            p_drop_emb=cfg.p_drop_emb,
            p_drop_attn=cfg.p_drop_attn,
            use_mlp_head=cfg.use_mlp_head,
            pose_dim=9,
            aux_pose_dim=9,
            aux_reg=cfg.aux_reg,
            movement_head_hidden=cfg.movement_head_hidden,
            sdf_weight=cfg.sdf_weight,
            diffusion_weight=cfg.diffusion_weight,
            movement_weight=cfg.movement_weight,
            aux_weight=cfg.aux_weight,
        ).to(device)
    elif task == "movement":
        return MovementModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=False,  # doesn't matter for eval
            movement_head_hidden=cfg.movement_head_hidden,
            sdf_weight=cfg.sdf_weight,
            movement_weight=cfg.movement_weight,
        ).to(device)
    elif task in ("diffusion", "translation"):
        return DiffusionModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=False,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_emb=cfg.n_emb,
            p_drop_emb=cfg.p_drop_emb,
            p_drop_attn=cfg.p_drop_attn,
            use_mlp_head=cfg.use_mlp_head,
            pose_dim=3 if task == "translation" else 9,
            aux_pose_dim=3 if task == "translation" else 9,
            aux_reg=cfg.aux_reg,
            sdf_weight=cfg.sdf_weight,
            diffusion_weight=cfg.diffusion_weight,
            aux_weight=cfg.aux_weight,
        ).to(device)
    else:  # sdf
        return SDFSegmentor(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            freeze_encoder=False,
        ).to(device)


def _safe_to_device(val, device):
    return val.to(device) if val is not None else None


# --------------------------------------------------------------------------- #
# Collate (same as train.py)
# --------------------------------------------------------------------------- #

_OPTIONAL_KEYS = {
    "tool_pc_init", "init_tool_pts_sdf", "init_obj_pts_sdf",
    "delta_pose", "delta_translation", "init_pose", "init_translation",
    "obj_point_displacement", "tool_delta_pose",
}

def _collate_fn(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key in _OPTIONAL_KEYS:
            if vals[0] is None:
                out[key] = None
                continue
        out[key] = torch.stack(vals)
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    max_batches: int = 0,
) -> dict:
    """Run the model on data and return per-metric averages."""
    model.eval()
    agg: dict[str, float] = {}
    n = 0

    for batch in loader:
        tool_pc = batch["tool_pc"].to(device)
        obj_pc  = batch["obj_pc"].to(device)
        tool_sdf_gt = batch["tool_pts_sdf"].to(device)
        obj_sdf_gt  = batch["obj_pts_sdf"].to(device)

        init_tool_sdf_gt = _safe_to_device(batch.get("init_tool_pts_sdf"), device)
        init_obj_sdf_gt = _safe_to_device(batch.get("init_obj_pts_sdf"), device)
        tool_pc_init = _safe_to_device(batch.get("tool_pc_init"), device)
        delta_pose = _safe_to_device(batch.get("delta_pose"), device)
        delta_translation = _safe_to_device(batch.get("delta_translation"), device)
        init_pose = _safe_to_device(batch.get("init_pose"), device)
        init_translation = _safe_to_device(batch.get("init_translation"), device)
        obj_point_displacement = _safe_to_device(batch.get("obj_point_displacement"), device)
        tool_delta_pose = _safe_to_device(batch.get("tool_delta_pose"), device)

        fwd_kwargs = {}
        if isinstance(model, JointModel):
            fwd_kwargs["tool_pc_init"] = tool_pc_init
            fwd_kwargs["delta_pose_gt"] = delta_pose
            fwd_kwargs["init_pose_gt"] = init_pose
            fwd_kwargs["enable_flow"] = True
            fwd_kwargs["init_tool_sdf_gt"] = init_tool_sdf_gt
            fwd_kwargs["init_obj_sdf_gt"] = init_obj_sdf_gt
            fwd_kwargs["tool_delta_action"] = tool_delta_pose
            fwd_kwargs["obj_displacement_gt"] = obj_point_displacement
        elif isinstance(model, MovementModel):
            fwd_kwargs["tool_delta_action"] = tool_delta_pose
            fwd_kwargs["obj_displacement_gt"] = obj_point_displacement
            fwd_kwargs["tool_pc_init"] = tool_pc_init
            fwd_kwargs["init_tool_sdf_gt"] = init_tool_sdf_gt
            fwd_kwargs["init_obj_sdf_gt"] = init_obj_sdf_gt
        elif isinstance(model, DiffusionModel):
            fwd_kwargs["tool_pc_init"] = tool_pc_init
            if model.pose_dim == 3:
                fwd_kwargs["delta_pose_gt"] = delta_translation
                fwd_kwargs["init_pose_gt"] = init_translation
            else:
                fwd_kwargs["delta_pose_gt"] = delta_pose
                fwd_kwargs["init_pose_gt"] = init_pose
            fwd_kwargs["enable_flow"] = True
            fwd_kwargs["init_tool_sdf_gt"] = init_tool_sdf_gt
            fwd_kwargs["init_obj_sdf_gt"] = init_obj_sdf_gt
        else:  # SDFSegmentor
            fwd_kwargs["tool_pc_init"] = tool_pc_init
            fwd_kwargs["init_tool_sdf_gt"] = init_tool_sdf_gt
            fwd_kwargs["init_obj_sdf_gt"] = init_obj_sdf_gt

        loss, metrics = model(tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt, **fwd_kwargs)

        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

        if max_batches > 0 and n >= max_batches:
            break

    return {k: v / max(n, 1) for k, v in agg.items()}, n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Validate a pretrain checkpoint")
    parser.add_argument("--task", required=True, choices=["sdf", "diffusion", "translation", "movement", "joint"],
                        help="Expected task type for the checkpoint")
    parser.add_argument("--head-mode", default=None, choices=["point", "patch"],
                        help="SDF head mode (auto-detected from checkpoint if not specified)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: from config)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit .pt files (0 = all)")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit eval batches (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--split", choices=["val", "train", "both"], default="val",
                        help="Which split to evaluate on")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir

    # ---- Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"✗ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle both formats: {model: ..., epoch: ...} and raw state_dict
    if "model" in ckpt:
        sd = ckpt["model"]
        epoch = ckpt.get("epoch", "?")
        best_val = ckpt.get("best_val", "?")
        print(f"  Epoch: {epoch}  Best val: {best_val}")
    else:
        sd = ckpt
        epoch = "?"
        print("  (raw state_dict, no epoch/val metadata)")

    # ---- Detect actual task from checkpoint ----
    detected = detect_task_from_state_dict(sd)
    print(f"  Detected task: {detected}")
    print(f"  Requested task: {args.task}")

    if detected != args.task:
        print(f"\n✗ MISMATCH: checkpoint is '{detected}' but you requested '{args.task}'")
        print(f"  Checkpoint keys hint:")
        for key in sorted(sd.keys())[:20]:
            print(f"    {key}")
        if len(sd) > 20:
            print(f"    ... ({len(sd)} total keys)")
        sys.exit(1)

    print(f"  ✓ Task match confirmed")

    # ---- Detect / validate head_mode ----
    detected_head = detect_head_mode_from_state_dict(sd)
    head_mode = args.head_mode or detected_head
    print(f"  Detected head_mode: {detected_head}")
    if args.head_mode and args.head_mode != detected_head:
        print(f"  ⚠ WARNING: you specified --head-mode {args.head_mode} "
              f"but checkpoint looks like '{detected_head}'")
        print(f"  Proceeding with your override: {args.head_mode}")
    else:
        print(f"  ✓ Using head_mode: {head_mode}")

    # ---- Build model and load weights ----
    print(f"\nBuilding {args.task} model (head_mode={head_mode})...")
    model = build_model(args.task, cfg, device, head_mode=head_mode)

    # Try loading — catch mismatched keys
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  ⚠ Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... ({len(missing)} total)")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    {k}")
        if len(unexpected) > 10:
            print(f"    ... ({len(unexpected)} total)")
    if not missing and not unexpected:
        print(f"  ✓ All keys matched perfectly")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # ---- Load data ----
    print(f"\nLoading data from: {cfg.data_dir}")
    require_movement = (args.task == "movement" or args.task == "joint")
    train_ds, val_ds = make_split(
        cfg.data_dir, val_ratio=cfg.val_ratio, max_files=args.max_files,
        require_movement=require_movement,
    )
    print(f"  Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    if len(train_ds) == 0 and len(val_ds) == 0:
        print("✗ No data found!")
        sys.exit(1)

    # ---- Evaluate ----
    splits = {}
    if args.split in ("val", "both") and len(val_ds) > 0:
        splits["val"] = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=_collate_fn,
        )
    if args.split in ("train", "both") and len(train_ds) > 0:
        splits["train"] = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=_collate_fn,
        )

    for split_name, loader in splits.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating on {split_name} split")
        print(f"{'='*60}")

        avg_metrics, n_batches = evaluate(
            model, loader, device, args.task, max_batches=args.max_batches,
        )

        print(f"  Batches evaluated: {n_batches}")
        print(f"  Metrics:")
        # Print total first, then sorted sub-metrics
        if "total" in avg_metrics:
            print(f"    {'total':<30s} {avg_metrics['total']:.6f}")
        for k in sorted(avg_metrics.keys()):
            if k != "total":
                print(f"    {k:<30s} {avg_metrics[k]:.6f}")

    print(f"\n✓ Validation complete")


if __name__ == "__main__":
    main()

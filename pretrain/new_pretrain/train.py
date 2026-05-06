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

# NOTE: do NOT add _PRETRAIN_DIR — it has config.py/model.py/dataset.py
# that would shadow our new_pretrain versions.
for p in [str(_REPO_ROOT), str(_RPDIFF_SRC), str(_THIS_DIR)]:
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

# Keys that should be passed through as lists (not stacked into tensors)
# Strings / variable-length fields: pass through as lists, do not torch.stack
_LIST_KEYS = {"tool_mesh_path", "obj_mesh_path", "pt_path"}


def collate_fn(batch):
    """Stack tensors from batch dicts; pass string/list fields through as lists."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if vals[0] is None:
            out[key] = None
        elif key in _LIST_KEYS or not isinstance(vals[0], torch.Tensor):
            # strings, lists, variable-length tensors → keep as list
            out[key] = vals
        else:
            out[key] = torch.stack(vals)
    return out


# ============================================================================ #
# Mesh loading cache (shared across all workers in this process)
# ============================================================================ #

_MESH_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_mesh_cached(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load mesh via trimesh and cache CPU tensors. Returns (verts, faces)."""
    if path not in _MESH_CACHE:
        import trimesh as _trimesh
        mesh = _trimesh.load(path, force="mesh", process=False)
        _MESH_CACHE[path] = (
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces,    dtype=torch.int64),
        )
    return _MESH_CACHE[path]


def _load_mesh_batch(
    tool_paths:   list[str],
    obj_paths:    list[str],
    tool_scales:  torch.Tensor,   # (B,)     scale applied at generation time
    obj_scales:   torch.Tensor,   # (B,)
    obj_Rs:       torch.Tensor,   # (B, 3, 3) rotation applied to object
    obj_z_shifts: torch.Tensor,   # (B,)     z-grounding shift
) -> tuple[list, list, list, list]:
    """Return (tool_verts, tool_faces, obj_verts, obj_faces) lists for a batch.

    Tool mesh  : scaled then centered by its vertex centroid (approx surface centroid).
                 This matches the centroid-frame convention used by noised_t / tool_canonical.
    Object mesh: scaled, rotated by R_obj, and z-grounded — i.e. in world frame.
                 This matches the frame used by tool_world in compute_on_the_fly_sdf.
    """
    tv, tf, ov, of = [], [], [], []
    for i, (tp, op) in enumerate(zip(tool_paths, obj_paths)):
        t_v_raw, t_f = _load_mesh_cached(tp)
        o_v_raw, o_f = _load_mesh_cached(op)

        # ---- Tool: scale then center by vertex centroid ----
        # Centering by vertex centroid approximates the surface-point centroid used
        # in contact_gen to define t_adj = R @ surface_centroid + t_origin.
        # The error |vertex_centroid - surface_centroid| is typically < 1 mm.
        t_scale = tool_scales[i].item()
        t_v_sc  = t_v_raw * t_scale                    # (V, 3) scaled
        t_v     = t_v_sc - t_v_sc.mean(dim=0)          # (V, 3) centered ≈ centroid frame

        # ---- Object: scale + rotate (canonical → world) + z-ground ----
        o_scale   = obj_scales[i].item()
        R_obj_i   = obj_Rs[i].cpu()                     # (3, 3) — keep on CPU like mesh verts
        z_shift_i = obj_z_shifts[i].item()
        o_v_sc  = o_v_raw * o_scale                    # (V, 3) scaled
        o_v     = o_v_sc @ R_obj_i.T                   # (V, 3) rotated to world
        o_v     = o_v.clone()
        o_v[:, 2] -= z_shift_i                         # (V, 3) grounded (world frame)

        tv.append(t_v); tf.append(t_f)
        ov.append(o_v); of.append(o_f)
    return tv, tf, ov, of


# ============================================================================ #
# Training step
# ============================================================================ #

def train_step(
    model: ContactDiffusionModel,
    batch: dict,
    cfg: NewPretrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Unified training step for both 'sdf' and 'sdf-diff' tasks.

    Encoder input is identical for every task:
      - tool_rotated : canonical tool pts rotated to the sampled (noised) pose
      - obj_pc       : object pts centered at origin
      - pose_3d      : noised_tool_centroid - obj_centroid  (3D)

    SDF labels are computed on-the-fly at the ACTUAL sampled pose via kaolin,
    so they are geometrically correct for any amount of noise.

    The only difference between tasks is movement_cond:
      - SDF head always sees movement_cond = zeros  (pose-only conditioning)
      - Denoising head (sdf-diff only) sees real movement deltas
    """
    tool_canonical  = batch["tool_canonical"].to(device)   # (B, P, 3) centered
    obj_pc          = batch["obj_pc"].to(device)           # (B, Q, 3) centered
    obj_centroid    = batch["obj_centroid"].to(device)     # (B, 3)
    contact_R       = batch["contact_R"].to(device)        # (B, 3, 3)
    contact_t       = batch["contact_t"].to(device)        # (B, 3)
    tool_mesh_paths = batch["tool_mesh_path"]              # list[str]
    obj_mesh_paths  = batch["obj_mesh_path"]               # list[str]
    # Mesh pose/scale metadata (for geometrically correct on-the-fly SDF)
    tool_scales  = batch["tool_scale"].to(device)          # (B,)
    obj_scales   = batch["object_scale"].to(device)        # (B,)
    obj_Rs       = batch["obj_R"].to(device)               # (B, 3, 3)
    obj_z_shifts = batch["obj_z_shift"].to(device)         # (B,)
    B = tool_canonical.shape[0]

    raw_model = model.module if isinstance(model, DDP) else model

    # ── 1. Sample a noised (or contact) pose ─────────────────────────────
    noise_out = sample_noised_poses_batch(
        contact_R=contact_R,
        contact_t=contact_t,
        num_steps=cfg.num_diffusion_steps,
        max_trans=cfg.noise_max_trans,
        max_rot_deg=cfg.noise_max_rot_deg,
        interp=cfg.interp_trajectory,
        precise_prob=cfg.precise_diff_prob,
        tool_canonical=tool_canonical,
        obj_pc=obj_pc,
        obj_centroid=obj_centroid,   # fixes world-frame mismatch in rejection guard
    )
    noised_R = noise_out["noised_R"]   # (B, 3, 3)
    noised_t = noise_out["noised_t"]   # (B, 3)

    # ── 2. Encoder input: tool rotated to the sampled pose ───────────────
    # obj_pc stays centered at origin (no change needed)
    tool_rotated = torch.bmm(tool_canonical, noised_R.transpose(1, 2))  # (B, P, 3)

    # ── 3. Pose conditioning: tool centroid relative to obj centroid ─────
    pose_3d = noised_t - obj_centroid   # (B, 3)

    # ── 4. On-the-fly SDF at the actual sampled pose ─────────────────────
    # Correct for ANY pose — no pre-baked approximation.
    tv_list, tf_list, ov_list, of_list = _load_mesh_batch(
        tool_mesh_paths, obj_mesh_paths,
        tool_scales, obj_scales, obj_Rs, obj_z_shifts,
    )
    with torch.no_grad():
        tool_sdf_gt, obj_sdf_gt = compute_on_the_fly_sdf(
            tool_canonical=tool_canonical,
            obj_pc=obj_pc,
            noised_R=noised_R,
            noised_t=noised_t,
            tool_verts=tv_list,
            tool_faces=tf_list,
            obj_verts=ov_list,
            obj_faces=of_list,
            obj_centroid=obj_centroid,
        )
    tool_sdf_gt = tool_sdf_gt.to(device)
    obj_sdf_gt  = obj_sdf_gt.to(device)

    # ── Validation: at t_idx == 0 on-the-fly SDF must match stored SDF ──────────
    # The noised pose equals the contact pose at t_idx=0, so the on-the-fly
    # SDF should reproduce the pre-baked dataset values.  A large error means
    # the mesh transform (scale / rotation / centering) is still wrong.
    if is_main() and (noise_out["t_idx"] == 0).any():
        zero_mask = (noise_out["t_idx"] == 0)
        stored_t_sdf = batch["stored_tool_sdf"].to(device)[zero_mask]  # (K, P)
        stored_o_sdf = batch["stored_obj_sdf"].to(device)[zero_mask]   # (K, Q)
        err_tool = (tool_sdf_gt[zero_mask] - stored_t_sdf).abs().mean().item()
        err_obj  = (obj_sdf_gt[zero_mask]  - stored_o_sdf).abs().mean().item()
        if err_tool > 5e-3 or err_obj > 5e-3:
            print(f"  [SDF-VALIDATE] t_idx=0 MAE: tool={err_tool:.5f}  obj={err_obj:.5f}"
                  f"  ⚠  > 5mm — mesh transform may be wrong")
        else:
            print(f"  [SDF-VALIDATE] t_idx=0 MAE: tool={err_tool:.5f}  obj={err_obj:.5f}  ✓")

    # ── 5. Movement conditioning + child point clouds (sdf-diff only) ────
    if raw_model.task == "sdf-diff":
        delta_tool_t    = batch["delta_tool_t"].to(device)   # (B, 3)
        delta_tool_R    = batch["delta_tool_R"].to(device)   # (B, 3, 3)
        delta_obj_t     = batch["delta_obj_t"].to(device)    # (B, 3)
        delta_obj_R     = batch["delta_obj_R"].to(device)    # (B, 3, 3)
        delta_tool_quat = matrix_to_quaternion(delta_tool_R) # (B, 4)
        delta_obj_quat  = matrix_to_quaternion(delta_obj_R)  # (B, 4)
        movement_cond = torch.cat(
            [delta_tool_t, delta_tool_quat, delta_obj_t, delta_obj_quat], dim=-1
        )  # (B, 14)

        # Child clouds: tool at current noised pose → tool one step closer to contact
        child_start_pcd = tool_rotated + noised_t.unsqueeze(1)   # (B, P, 3)
        prev_R = torch.bmm(noise_out["target_rot_mat"], noised_R)
        prev_t = noised_t + noise_out["target_trans"]
        child_final_pcd = (
            torch.bmm(tool_canonical, prev_R.transpose(1, 2)) + prev_t.unsqueeze(1)
        )  # (B, P, 3)
    else:
        # sdf-only: movement unused (model will zero it for the SDF pass)
        movement_cond   = torch.zeros(B, cfg.movement_cond_dim, device=device)
        child_start_pcd = None
        child_final_pcd = None

    # ── 6. Forward ────────────────────────────────────────────────────────
    return model(
        tool_rotated=tool_rotated,
        obj_pc=obj_pc,
        tool_sdf_gt=tool_sdf_gt,
        obj_sdf_gt=obj_sdf_gt,
        pose_3d=pose_3d,
        timestep=noise_out["t_idx"],
        movement_cond=movement_cond,
        target_trans=noise_out["target_trans"] if raw_model.task == "sdf-diff" else None,
        target_rot_mat=noise_out["target_rot_mat"] if raw_model.task == "sdf-diff" else None,
        child_start_pcd=child_start_pcd,
        child_final_pcd=child_final_pcd,
    )


# ============================================================================ #
# Main training loop
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="RPDiff-style joint SDF + denoising pretraining")
    parser.add_argument("--data-dir",    type=str, required=True)
    parser.add_argument("--task",        type=str, default=None, choices=["sdf", "sdf-diff"])
    parser.add_argument("--head-mode",   type=str, default=None, choices=["point", "patch"])
    parser.add_argument("--resume",      type=str, default="")
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--max-files",   type=int, default=0,
                        help="Limit number of config files (0 = all)")
    args = parser.parse_args()

    # Everything else comes from config.py — edit NewPretrainConfig directly.
    cfg = NewPretrainConfig(data_dir=args.data_dir)
    if args.task is not None:
        cfg.task = args.task
    if args.head_mode is not None:
        cfg.head_mode = args.head_mode
    if args.resume:
        cfg.resume = args.resume
    if args.wandb:
        cfg.wandb = True
    if args.max_files:
        cfg.max_files = args.max_files

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
        require_movement=(cfg.task == "sdf-diff"),
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
        head_hidden=cfg.head_hidden,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=cfg.freeze_encoder,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        pose_dim=cfg.pose_dim,
        movement_cond_dim=cfg.movement_cond_dim,
        denoise_hidden=cfg.denoise_hidden,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        denoise_rot_weight=cfg.denoise_rot_weight,
        chamfer_weight=cfg.chamfer_weight,
        quat_norm_beta=cfg.quat_norm_beta,
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

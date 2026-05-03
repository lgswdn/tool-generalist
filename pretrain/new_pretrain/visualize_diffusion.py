"""visualize_diffusion.py — Visualize iterative denoising from a trained checkpoint.

Loads a trained ContactDiffusionModel, picks sample(s) from the dataset,
starts from a fully noised pose (t=T), iteratively denoises for T steps,
and renders the tool + object point clouds at each step as a video.

Usage:
    python pretrain/new_pretrain/visualize_diffusion.py \
        --data-dir /path/to/teardrop_contact/ \
        --ckpt checkpoints_new/best.pt \
        --num-samples 3 \
        --out-dir viz_diffusion/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PRETRAIN_DIR = _THIS_DIR.parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
_RPDIFF_SRC   = _PRETRAIN_DIR / "rpdiff" / "src"

for p in [str(_REPO_ROOT), str(_RPDIFF_SRC), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import NewPretrainConfig
from dataset import make_split
from model import ContactDiffusionModel
from noise_utils import (
    _random_quaternions, _quat_to_rotmat, _rotmat_to_quat,
    _quat_slerp, _quat_inverse, _quat_multiply,
)
from rpdiff.utils.torch3d_util import matrix_to_quaternion


# ============================================================================ #
# Inference: iterative denoising
# ============================================================================ #

@torch.no_grad()
def denoise_iterative(
    model: ContactDiffusionModel,
    tool_canonical: torch.Tensor,   # (1, P, 3)
    obj_pc: torch.Tensor,           # (1, Q, 3)
    start_R: torch.Tensor,          # (1, 3, 3) noised starting rotation
    start_t: torch.Tensor,          # (1, 3)   noised starting translation
    T: int,
) -> list[dict]:
    """Run iterative denoising from t=T down to t=0.

    Returns list of T+1 dicts with {R, t, tool_world} at each step.
    """
    device = tool_canonical.device
    trajectory = []

    cur_R = start_R
    cur_t = start_t

    for step in range(T, -1, -1):
        # Record current state
        tool_world = torch.bmm(tool_canonical, cur_R.transpose(1, 2)) + cur_t.unsqueeze(1)
        trajectory.append({
            "step": step,
            "R": cur_R.clone(),
            "t": cur_t.clone(),
            "tool_world": tool_world.squeeze(0).cpu().numpy(),
        })

        if step == 0:
            break

        # Build noised_pose_7d for model input
        noised_quat = matrix_to_quaternion(cur_R)  # (1, 4)
        noised_pose_7d = torch.cat([cur_t, noised_quat], dim=-1)  # (1, 7)
        timestep = torch.tensor([step], dtype=torch.long, device=device)

        # Encode
        encoder_result = model.encoder.encode(tool_canonical, obj_pc)
        P = model.num_patches
        fused = encoder_result.fused_tokens
        fused_cond = model.pose_cross_attn(fused, noised_pose_7d, timestep)

        # Predict denoising step
        pooled = fused_cond.mean(dim=1)
        pred = model.denoising_head(pooled)

        pred_R = pred["rot_mat"]   # (1, 3, 3)
        pred_t = pred["trans"]     # (1, 3)

        # Apply predicted transform to current pose
        # Rotate around centroid of child cloud, then translate
        child_mean = tool_world.mean(dim=1, keepdim=True)  # (1, 1, 3)
        child_centered = tool_world - child_mean            # (1, P, 3)
        child_rotated = torch.bmm(pred_R, child_centered.transpose(1, 2)).transpose(1, 2)
        tool_world_new = child_rotated + child_mean + pred_t.unsqueeze(1)

        # Extract new R, t from the transformed cloud
        # new_tool_world = tool_canonical @ new_R.T + new_t
        # Solve for new_R, new_t via Procrustes
        new_R, new_t = _procrustes(tool_canonical.squeeze(0), tool_world_new.squeeze(0))
        cur_R = new_R.unsqueeze(0)
        cur_t = new_t.unsqueeze(0)

    return trajectory


def _procrustes(src: torch.Tensor, tgt: torch.Tensor):
    """Solve for R, t such that tgt ≈ src @ R.T + t.

    Args:
        src: (P, 3) source points (canonical)
        tgt: (P, 3) target points (world)

    Returns:
        R: (3, 3), t: (3,)
    """
    src_mean = src.mean(dim=0)
    tgt_mean = tgt.mean(dim=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    # H = src_c.T @ tgt_c → SVD → R = V @ U.T
    H = src_c.T @ tgt_c  # (3, 3)
    U, S, Vh = torch.linalg.svd(H)
    d = torch.det(Vh.T @ U.T)
    sign = torch.tensor([1, 1, d.sign()], device=src.device)
    R = Vh.T @ torch.diag(sign) @ U.T  # R such that tgt_c ≈ src_c @ R.T

    t = tgt_mean - src_mean @ R.T
    return R, t


# ============================================================================ #
# Generate noised starting pose
# ============================================================================ #

def sample_fully_noised_pose(
    contact_R: torch.Tensor,  # (1, 3, 3)
    contact_t: torch.Tensor,  # (1, 3)
    max_trans: float,
    max_rot_deg: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fully noised pose (t=T) by applying a large perturbation."""
    pert_q = _random_quaternions(1, max_rot_deg, device)
    pert_R = _quat_to_rotmat(pert_q)  # (1, 3, 3)
    pert_t = (torch.rand(1, 3, device=device) * 2 - 1) * max_trans

    noised_R = torch.bmm(pert_R, contact_R)
    noised_t = torch.bmm(pert_R, contact_t.unsqueeze(-1)).squeeze(-1) + pert_t
    return noised_R, noised_t


# ============================================================================ #
# Rendering
# ============================================================================ #

def render_frame(
    ax,
    tool_pts: np.ndarray,     # (P, 3)
    obj_pts: np.ndarray,      # (Q, 3)
    gt_tool_pts: np.ndarray,  # (P, 3) — ground truth tool position
    step: int,
    total_steps: int,
    elev: float = 25,
    azim: float = 45,
):
    """Render one frame of the diffusion visualization."""
    ax.clear()

    # Object (gray)
    ax.scatter(
        obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
        c="gray", s=3, alpha=0.3, label="Object"
    )

    # GT tool position (green, transparent)
    ax.scatter(
        gt_tool_pts[:, 0], gt_tool_pts[:, 1], gt_tool_pts[:, 2],
        c="limegreen", s=3, alpha=0.15, label="GT pose"
    )

    # Current tool (colored by denoising progress)
    progress = 1.0 - step / total_steps
    color = plt.cm.coolwarm(progress)
    ax.scatter(
        tool_pts[:, 0], tool_pts[:, 1], tool_pts[:, 2],
        c=[color], s=5, alpha=0.7, label=f"Step {step}"
    )

    # Axis settings
    all_pts = np.concatenate([tool_pts, obj_pts, gt_tool_pts], axis=0)
    center = all_pts.mean(axis=0)
    max_range = np.abs(all_pts - center).max() * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Denoising step {step}/{total_steps}", fontsize=14)
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=elev, azim=azim)


def save_video(
    trajectory: list[dict],
    obj_pts: np.ndarray,
    gt_tool_pts: np.ndarray,
    out_path: str,
    T: int,
    fps: int = 3,
):
    """Save denoising trajectory as MP4 video."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        step_data = trajectory[frame_idx]
        render_frame(
            ax, step_data["tool_world"], obj_pts, gt_tool_pts,
            step=step_data["step"], total_steps=T,
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(trajectory),
        interval=1000 // fps, blit=False
    )

    # Try mp4, fallback to gif
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(out_path, writer=writer)
    except Exception:
        gif_path = out_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"  (FFmpeg unavailable, saved as GIF: {gif_path})")

    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_comparison_image(
    trajectory: list[dict],
    obj_pts: np.ndarray,
    gt_tool_pts: np.ndarray,
    out_path: str,
    T: int,
):
    """Save a single image with key frames side by side."""
    # Pick key frames: start, 25%, 50%, 75%, end
    n = len(trajectory)
    key_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    key_indices = sorted(set(key_indices))

    fig, axes = plt.subplots(1, len(key_indices), figsize=(6 * len(key_indices), 5),
                              subplot_kw={"projection": "3d"})
    if len(key_indices) == 1:
        axes = [axes]

    for ax, ki in zip(axes, key_indices):
        step_data = trajectory[ki]
        render_frame(
            ax, step_data["tool_world"], obj_pts, gt_tool_pts,
            step=step_data["step"], total_steps=T,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="Visualize diffusion denoising")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (best.pt)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--out-dir", type=str, default="viz_diffusion")
    parser.add_argument("--max-files", type=int, default=5,
                        help="Limit dataset files for fast loading")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = NewPretrainConfig(data_dir=args.data_dir)

    # ── Load dataset ──────────────────────────────────────────────────────
    _, val_ds = make_split(
        data_dir=args.data_dir,
        val_ratio=0.1,
        seed=cfg.seed,
        augment=False,
        max_files=args.max_files,
    )
    print(f"Val dataset: {len(val_ds)} configs")

    # ── Load model ────────────────────────────────────────────────────────
    assert cfg.task == "sdf-diff", "Visualization requires sdf-diff model"

    model = ContactDiffusionModel(
        head_mode=cfg.head_mode,
        patch_agg=cfg.patch_agg,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=True,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        denoise_hidden=cfg.denoise_hidden,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        chamfer_weight=cfg.chamfer_weight,
        num_diffusion_steps=cfg.num_diffusion_steps,
        task="sdf-diff",
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Visualize samples ─────────────────────────────────────────────────
    T = cfg.num_diffusion_steps
    indices = list(range(0, len(val_ds), max(1, len(val_ds) // args.num_samples)))
    indices = indices[:args.num_samples]

    for sample_i, data_idx in enumerate(indices):
        print(f"\nSample {sample_i + 1}/{len(indices)} (data_idx={data_idx})")

        sample = val_ds[data_idx]
        tool_canonical = sample["tool_canonical"].unsqueeze(0).to(device)  # (1, P, 3)
        obj_pc = sample["obj_pc"].unsqueeze(0).to(device)                  # (1, Q, 3)
        contact_R = sample["contact_R"].unsqueeze(0).to(device)            # (1, 3, 3)
        contact_t = sample["contact_t"].unsqueeze(0).to(device)            # (1, 3)

        # GT tool in world frame
        gt_tool_world = torch.bmm(
            tool_canonical, contact_R.transpose(1, 2)
        ) + contact_t.unsqueeze(1)  # (1, P, 3)
        gt_tool_np = gt_tool_world.squeeze(0).cpu().numpy()
        obj_np = obj_pc.squeeze(0).cpu().numpy()

        # Sample fully noised starting pose
        noised_R, noised_t = sample_fully_noised_pose(
            contact_R, contact_t,
            max_trans=cfg.noise_max_trans,
            max_rot_deg=cfg.noise_max_rot_deg,
            device=device,
        )

        # Run iterative denoising
        trajectory = denoise_iterative(
            model, tool_canonical, obj_pc,
            noised_R, noised_t, T,
        )

        # Save video
        video_path = str(out_dir / f"sample_{sample_i:02d}.mp4")
        save_video(trajectory, obj_np, gt_tool_np, video_path, T, fps=2)

        # Save comparison image
        img_path = str(out_dir / f"sample_{sample_i:02d}_keyframes.png")
        save_comparison_image(trajectory, obj_np, gt_tool_np, img_path, T)

    print(f"\nDone! Outputs in {out_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a ContactDiffusionModel checkpoint rollout as an MP4 video.

Usage:
    python new_pretrain/visualize_diffusion_checkpoint.py \
        --checkpoint checkpoints_new/best.pt \
        --input-dir tmp_data \
        --save vis_outputs/new_pretrain_diffusion.mp4

The script uses new_pretrain/config.py defaults, loads a sdf-diff checkpoint,
samples a max-noise pose around a dataset contact pose, and applies the learned
one-step denoising transform from t=T down to t=0.
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PRETRAIN_DIR = _THIS_DIR.parent
_REPO_ROOT = _PRETRAIN_DIR.parent
_RPDIFF_SRC = _PRETRAIN_DIR / "rpdiff" / "src"

for p in [str(_THIS_DIR), str(_PRETRAIN_DIR), str(_REPO_ROOT), str(_RPDIFF_SRC)]:
    while p in sys.path:
        sys.path.remove(p)
for p in [str(_RPDIFF_SRC), str(_REPO_ROOT), str(_PRETRAIN_DIR), str(_THIS_DIR)]:
    sys.path.insert(0, p)

from config import NewPretrainConfig
from dataset import NewPretrainDataset
from model import ContactDiffusionModel
from noise_utils import _quat_to_rotmat, _random_quaternions
from rpdiff.utils.torch3d_util import matrix_to_quaternion
from visualize_movement_delta import (
    CONTACT_PT_COLOUR,
    OBJ_COLOUR_BEFORE,
    TOOL_COLOUR_AFTER,
    _add_ground,
    _plot_contact_point,
    _plot_mesh,
    _set_equal_aspect,
    apply_object_pose,
    load_mesh_trimesh,
    transform_mesh,
)


GT_TOOL_COLOUR = (0.45, 0.80, 0.55, 0.30)
TRAJECTORY_COLOUR = (1.0, 0.45, 0.05)


def rotation_angle_deg_np(R: np.ndarray) -> float:
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta) * 180.0 / np.pi)


def rotation_angle_deg_torch(R: torch.Tensor) -> torch.Tensor:
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta) * 180.0 / torch.pi


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    if "pose_cross_attn.pose_proj.0.weight" not in state_dict:
        raise RuntimeError("Checkpoint is not a new_pretrain sdf-diff checkpoint.")
    return state_dict


def detect_head_mode(state_dict: dict) -> str:
    if "xyz_embed.0.weight" in state_dict:
        return "point"
    return "patch"


def build_model(cfg: NewPretrainConfig, state_dict: dict, device: torch.device) -> ContactDiffusionModel:
    cfg.task = "sdf-diff"
    cfg.head_mode = detect_head_mode(state_dict)
    model = ContactDiffusionModel(
        head_mode=cfg.head_mode,
        patch_agg=cfg.patch_agg,
        head_hidden=cfg.head_hidden,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=False,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        pose_dim=cfg.pose_dim,
        denoise_hidden=cfg.denoise_hidden,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        chamfer_weight=cfg.chamfer_weight,
        quat_norm_beta=cfg.quat_norm_beta,
        num_diffusion_steps=cfg.num_diffusion_steps,
        task=cfg.task,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def collect_input_files(args) -> list[str]:
    if args.input and args.input_dir:
        raise RuntimeError("Use either --input or --input-dir, not both.")
    if args.input:
        files = [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    else:
        raise RuntimeError("Provide --input or --input-dir.")
    if len(files) == 0:
        raise RuntimeError("No .pt files found.")
    return files


def sample_dataset_items(
    dataset: NewPretrainDataset,
    args,
) -> list[tuple[str, int]]:
    if args.config_index >= 0:
        if not args.input:
            raise RuntimeError("--config-index requires --input.")
        return [(args.input, args.config_index)]

    rng = random.Random(args.seed)
    n_samples = min(args.num_samples, len(dataset))
    sample_indices = rng.sample(range(len(dataset)), n_samples)
    return [dataset._index[i] for i in sample_indices]


def make_output_path(save_path: str, sample_i: int, n_samples: int) -> Path:
    path = Path(save_path)
    if n_samples == 1:
        return path
    return path.parent / f"{path.stem}_{sample_i:02d}{path.suffix}"


def build_centered_tool_mesh(data: dict):
    tool_mesh = load_mesh_trimesh(data["tool_mesh_path"])
    tool_mesh.vertices *= to_numpy(data["tool_scale"])
    tool_centroid = to_numpy(data["tool_pts_canonical"]).mean(axis=0)
    tool_mesh.vertices -= tool_centroid
    return tool_mesh


def build_object_mesh(data: dict):
    obj_mesh = load_mesh_trimesh(data["object_mesh_path"])
    obj_mesh.vertices *= to_numpy(data["object_scale"])
    return apply_object_pose(
        obj_mesh,
        to_numpy(data["object_rotation"]),
        float(to_numpy(data["obj_z_shift"])),
    )


def sample_start_pose(
    contact_R: torch.Tensor,
    contact_t: torch.Tensor,
    max_trans: float,
    max_rot_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = contact_R.device
    pert_q = _random_quaternions(1, max_rot_deg, device)
    pert_R = _quat_to_rotmat(pert_q)
    pert_t = (torch.rand(1, 3, device=device) * 2.0 - 1.0) * max_trans
    start_R = torch.bmm(pert_R, contact_R)
    start_t = torch.bmm(pert_R, contact_t.unsqueeze(-1)).squeeze(-1) + pert_t
    return start_R, start_t


@torch.no_grad()
def rollout_diffusion(
    model: ContactDiffusionModel,
    tool_canonical: torch.Tensor,
    obj_pc: torch.Tensor,
    contact_R: torch.Tensor,
    contact_t: torch.Tensor,
    max_trans: float,
    max_rot_deg: float,
    n_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, float]]]:
    cur_R, cur_t = sample_start_pose(contact_R, contact_t, max_trans, max_rot_deg)
    poses_R = [cur_R[0].detach().cpu().numpy()]
    poses_t = [cur_t[0].detach().cpu().numpy()]
    rel_R = torch.bmm(cur_R, contact_R.transpose(1, 2))
    stats = [{
        "frame": 0.0,
        "timestep": float(n_steps),
        "delta_tx": 0.0,
        "delta_ty": 0.0,
        "delta_tz": 0.0,
        "delta_trans_mm": 0.0,
        "delta_rot_deg": 0.0,
        "pose_tx": float(cur_t[0, 0].detach().cpu()),
        "pose_ty": float(cur_t[0, 1].detach().cpu()),
        "pose_tz": float(cur_t[0, 2].detach().cpu()),
        "trans_error_mm": float(torch.linalg.norm(cur_t - contact_t, dim=-1)[0].detach().cpu() * 1000.0),
        "rot_error_deg": float(rotation_angle_deg_torch(rel_R)[0].detach().cpu()),
    }]

    encoder_result = model.encoder.encode(tool_canonical, obj_pc)
    for frame_idx, step in enumerate(range(n_steps, 0, -1), start=1):
        quat = matrix_to_quaternion(cur_R)
        pose_7d = torch.cat([cur_t, quat], dim=-1)
        timestep = torch.tensor([step], device=tool_canonical.device, dtype=torch.long)
        fused = model.pose_cross_attn(encoder_result.fused_tokens, pose_7d, timestep)
        pooled = fused.mean(dim=1)
        denoise_out = model.denoising_head(pooled)
        delta_R = denoise_out["rot_mat"]
        delta_t = denoise_out["trans"]

        cur_R = torch.bmm(delta_R, cur_R)
        cur_t = cur_t + delta_t
        poses_R.append(cur_R[0].detach().cpu().numpy())
        poses_t.append(cur_t[0].detach().cpu().numpy())
        rel_R = torch.bmm(cur_R, contact_R.transpose(1, 2))
        stats.append({
            "frame": float(frame_idx),
            "timestep": float(step - 1),
            "delta_tx": float(delta_t[0, 0].detach().cpu()),
            "delta_ty": float(delta_t[0, 1].detach().cpu()),
            "delta_tz": float(delta_t[0, 2].detach().cpu()),
            "delta_trans_mm": float(torch.linalg.norm(delta_t, dim=-1)[0].detach().cpu() * 1000.0),
            "delta_rot_deg": float(rotation_angle_deg_torch(delta_R)[0].detach().cpu()),
            "pose_tx": float(cur_t[0, 0].detach().cpu()),
            "pose_ty": float(cur_t[0, 1].detach().cpu()),
            "pose_tz": float(cur_t[0, 2].detach().cpu()),
            "trans_error_mm": float(torch.linalg.norm(cur_t - contact_t, dim=-1)[0].detach().cpu() * 1000.0),
            "rot_error_deg": float(rotation_angle_deg_torch(rel_R)[0].detach().cpu()),
        })

    return poses_R, poses_t, stats


def write_rollout_stats(output_path: Path, stats: list[dict[str, float]]) -> Path:
    stats_path = output_path.with_suffix(".csv")
    with stats_path.open("w", newline="") as f:
        fieldnames = [
            "frame",
            "timestep",
            "delta_tx",
            "delta_ty",
            "delta_tz",
            "delta_trans_mm",
            "delta_rot_deg",
            "pose_tx",
            "pose_ty",
            "pose_tz",
            "trans_error_mm",
            "rot_error_deg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    return stats_path


def print_rollout_stats(stats: list[dict[str, float]]):
    print("frame,timestep,delta_trans_mm,delta_rot_deg,trans_error_mm,rot_error_deg,delta_t")
    for row in stats:
        print(
            f"{int(row['frame'])},"
            f"{int(row['timestep'])},"
            f"{row['delta_trans_mm']:.3f},"
            f"{row['delta_rot_deg']:.3f},"
            f"{row['trans_error_mm']:.3f},"
            f"{row['rot_error_deg']:.3f},"
            f"({row['delta_tx']:.6f},{row['delta_ty']:.6f},{row['delta_tz']:.6f})"
        )


def collect_bounds_vertices(
    obj_mesh,
    tool_mesh,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    poses_R: list[np.ndarray],
    poses_t: list[np.ndarray],
) -> np.ndarray:
    verts = [obj_mesh.vertices, transform_mesh(tool_mesh, gt_R, gt_t).vertices]
    for R, t in zip(poses_R, poses_t):
        verts.append(transform_mesh(tool_mesh, R, t).vertices)
    return np.concatenate(verts, axis=0)


def render_frame(
    fig,
    ax,
    obj_mesh,
    tool_mesh,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    cur_R: np.ndarray,
    cur_t: np.ndarray,
    contact_pt: np.ndarray,
    trajectory: np.ndarray,
    all_verts: np.ndarray,
    frame_i: int,
    n_frames: int,
    elev: float,
    azim: float,
    title: str,
) -> np.ndarray:
    ax.clear()
    tool_gt = transform_mesh(tool_mesh, gt_R, gt_t)
    tool_cur = transform_mesh(tool_mesh, cur_R, cur_t)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))

    _add_ground(ax, extent)
    _plot_mesh(ax, obj_mesh, OBJ_COLOUR_BEFORE, label="Object")
    _plot_mesh(ax, tool_gt, GT_TOOL_COLOUR, edge_alpha=0.04, label="GT contact")
    _plot_mesh(ax, tool_cur, TOOL_COLOUR_AFTER, label="Denoising pose")
    _plot_contact_point(ax, contact_pt, CONTACT_PT_COLOUR, size=80, label="Contact point")

    if len(trajectory) > 1:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=TRAJECTORY_COLOUR,
            linewidth=2.0,
            alpha=0.9,
            label="Tool-origin path",
        )
        ax.scatter(
            trajectory[-1:, 0],
            trajectory[-1:, 1],
            trajectory[-1:, 2],
            color=TRAJECTORY_COLOUR[:3],
            s=35,
            depthshade=False,
        )

    trans_err_mm = np.linalg.norm(cur_t - gt_t) * 1000.0
    ax.set_title(
        f"{title}\nframe {frame_i + 1}/{n_frames}  t={n_frames - frame_i - 1}  "
        f"translation error={trans_err_mm:.1f}mm",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    _set_equal_aspect(ax, all_verts)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.75)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[:, :, :3].copy()


def render_video_for_sample(
    model: ContactDiffusionModel,
    dataset: NewPretrainDataset,
    index_by_item: dict,
    pt_path: str,
    cfg_i: int,
    cfg: NewPretrainConfig,
    args,
    output_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    sample = dataset[index_by_item[(pt_path, cfg_i)]]
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed + cfg_i)
    tool_canonical = sample["tool_canonical"].unsqueeze(0).to(device)
    obj_pc = sample["obj_pc"].unsqueeze(0).to(device)
    contact_R = sample["contact_R"].unsqueeze(0).to(device)
    contact_t = sample["contact_t"].unsqueeze(0).to(device)

    poses_R, poses_t, stats = rollout_diffusion(
        model=model,
        tool_canonical=tool_canonical,
        obj_pc=obj_pc,
        contact_R=contact_R,
        contact_t=contact_t,
        max_trans=args.noise_trans,
        max_rot_deg=args.noise_rot_deg,
        n_steps=args.steps,
    )
    stats_path = write_rollout_stats(output_path, stats)
    print_rollout_stats(stats)

    obj_mesh = build_object_mesh(data)
    tool_mesh = build_centered_tool_mesh(data)
    gt_R = to_numpy(sample["contact_R"])
    gt_t = to_numpy(sample["contact_t"])
    contact_pt = to_numpy(data["contact_pts_world"][cfg_i, 0])
    all_verts = collect_bounds_vertices(obj_mesh, tool_mesh, gt_R, gt_t, poses_R, poses_t)

    title = (
        f"{Path(data['tool_mesh_path']).stem} x {Path(data['object_mesh_path']).stem} "
        f"cfg {cfg_i}  head={cfg.head_mode}"
    )
    fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
    fig.patch.set_facecolor("#f8f8f8")
    ax = fig.add_subplot(111, projection="3d")

    frames = []
    for frame_i, (cur_R, cur_t) in enumerate(zip(poses_R, poses_t)):
        trajectory = np.asarray(poses_t[:frame_i + 1])
        frame = render_frame(
            fig=fig,
            ax=ax,
            obj_mesh=obj_mesh,
            tool_mesh=tool_mesh,
            gt_R=gt_R,
            gt_t=gt_t,
            cur_R=cur_R,
            cur_t=cur_t,
            contact_pt=contact_pt,
            trajectory=trajectory,
            all_verts=all_verts,
            frame_i=frame_i,
            n_frames=len(poses_R),
            elev=args.elev,
            azim=args.azim,
            title=title,
        )
        frames.append(frame)

    for _ in range(args.hold_final):
        frames.append(frames[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)

    if args.save_frames:
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_i, frame in enumerate(frames):
            imageio.imwrite(frames_dir / f"frame_{frame_i:03d}.png", frame)

    plt.close(fig)
    final_err_mm = np.linalg.norm(poses_t[-1] - gt_t) * 1000.0
    final_rot_err_deg = rotation_angle_deg_np(poses_R[-1] @ gt_R.T)
    print(
        f"Saved {output_path}  stats={stats_path}  "
        f"final translation error={final_err_mm:.1f}mm  final rotation error={final_rot_err_deg:.2f}deg"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize new_pretrain diffusion denoising as MP4.")
    parser.add_argument("--checkpoint", required=True, help="Path to new_pretrain sdf-diff checkpoint.")
    parser.add_argument("--input", type=str, help="Single .pt data file.")
    parser.add_argument("--input-dir", type=str, help="Directory containing .pt data files.")
    parser.add_argument("--config-index", type=int, default=-1, help="Config index within --input. -1 samples randomly.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of videos to generate.")
    parser.add_argument("--save", type=str, default="vis_outputs/new_pretrain_diffusion.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="Denoising rollout steps. 0 uses config num_diffusion_steps.")
    parser.add_argument("--noise-trans", type=float, default=-1.0, help="Initial max translation noise in metres. <0 uses config.")
    parser.add_argument("--noise-rot-deg", type=float, default=-1.0, help="Initial max rotation noise in degrees. <0 uses config.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--hold-final", type=int, default=6, help="Extra final-frame copies appended to the video.")
    parser.add_argument("--save-frames", action="store_true", help="Also save rendered PNG frames next to the video.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = NewPretrainConfig()
    if args.steps == 0:
        args.steps = cfg.num_diffusion_steps
    if args.noise_trans < 0.0:
        args.noise_trans = cfg.noise_max_trans
    if args.noise_rot_deg < 0.0:
        args.noise_rot_deg = cfg.noise_max_rot_deg

    state_dict = load_state_dict(args.checkpoint)
    device = torch.device(args.device)
    model = build_model(cfg, state_dict, device)

    files = collect_input_files(args)
    dataset = NewPretrainDataset(files, augment=False)
    sampled_items = sample_dataset_items(dataset, args)
    index_by_item = {item: i for i, item in enumerate(dataset._index)}

    for sample_i, (pt_path, cfg_i) in enumerate(sampled_items):
        output_path = make_output_path(args.save, sample_i, len(sampled_items))
        render_video_for_sample(
            model=model,
            dataset=dataset,
            index_by_item=index_by_item,
            pt_path=pt_path,
            cfg_i=cfg_i,
            cfg=cfg,
            args=args,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()

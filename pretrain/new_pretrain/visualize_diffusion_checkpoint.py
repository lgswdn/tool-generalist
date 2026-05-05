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
import trimesh

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
    CONTACT_PT_AFTER,
    CONTACT_PT_COLOUR,
    OBJ_COLOUR_AFTER,
    OBJ_COLOUR_BEFORE,
    OBJ_COLOUR_GHOST,
    TOOL_COLOUR_AFTER,
    TOOL_COLOUR_GHOST,
    _add_ground,
    _plot_contact_point,
    _plot_mesh,
    _set_equal_aspect,
    apply_delta_to_object,
    apply_object_pose,
    load_mesh_trimesh,
    transform_mesh,
)


TRAJECTORY_COLOUR = (1.0, 0.45, 0.05)
OBJ_PRED_COLOUR = (0.65, 0.65, 0.70)
TOOL_PRED_COLOUR = (0.90, 0.35, 0.30)
OBJ_MOVED_COLOUR = (0.45, 0.80, 0.55)
TOOL_MOVED_COLOUR = (0.30, 0.55, 0.90)
GROUND_COLOUR = (0.90, 0.90, 0.88)
VIEWPOINTS = (
    (25.0, -55.0, "front"),
    (70.0, -45.0, "top"),
    (10.0, -125.0, "side"),
)


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


def min_sdf_np(tool_canonical: np.ndarray, obj_pc: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    tool_world = tool_canonical @ R.T + t
    dists = np.linalg.norm(tool_world[:, None, :] - obj_pc[None, :, :], axis=-1)
    return float(dists.min())


def apply_delta_to_points(points: np.ndarray, delta_R: np.ndarray, delta_t: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    return (points - pivot) @ delta_R.T + pivot + delta_t


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


def infer_denoise_hidden(state_dict: dict) -> tuple[int, ...]:
    linear_layers = sorted(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("denoising_head.out_trans.") and key.endswith(".weight")
    )
    if len(linear_layers) == 0:
        raise RuntimeError("Cannot infer denoising head hidden layers from checkpoint.")
    return tuple(
        int(state_dict[f"denoising_head.out_trans.{layer_idx}.weight"].shape[0])
        for layer_idx in linear_layers[:-1]
    )


def infer_movement_cond_dim(state_dict: dict) -> int:
    return int(state_dict["pose_cross_attn.movement_proj.0.weight"].shape[1])


def build_model(cfg: NewPretrainConfig, state_dict: dict, device: torch.device) -> ContactDiffusionModel:
    cfg.task = "sdf-diff"
    cfg.head_mode = detect_head_mode(state_dict)
    denoise_hidden = infer_denoise_hidden(state_dict)
    movement_cond_dim = infer_movement_cond_dim(state_dict)
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
        movement_cond_dim=movement_cond_dim,
        denoise_hidden=denoise_hidden,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        denoise_rot_weight=cfg.denoise_rot_weight,
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
    movement_cond: torch.Tensor,
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
        fused = model.pose_cross_attn(encoder_result.fused_tokens, pose_7d, timestep, movement_cond)
        P = model.num_patches
        tool_cond = fused[:, :P, :]
        obj_cond = fused[:, P:, :]
        pooled = torch.cat([tool_cond.mean(dim=1), obj_cond.mean(dim=1)], dim=-1)
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
    poses_R: list[np.ndarray],
    poses_t: list[np.ndarray],
) -> np.ndarray:
    verts = [obj_mesh.vertices]
    for R, t in zip(poses_R, poses_t):
        verts.append(transform_mesh(tool_mesh, R, t).vertices)
    return np.concatenate(verts, axis=0)


def render_frame(
    fig,
    axes,
    obj_mesh,
    tool_mesh,
    cur_R: np.ndarray,
    cur_t: np.ndarray,
    contact_pt: np.ndarray,
    trajectory: np.ndarray,
    min_sdf: float,
    all_verts: np.ndarray,
    frame_i: int,
    n_frames: int,
    title: str,
) -> np.ndarray:
    tool_cur = transform_mesh(tool_mesh, cur_R, cur_t)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))

    for view_i, (ax, (view_elev, view_azim, view_name)) in enumerate(zip(axes, VIEWPOINTS)):
        ax.clear()
        _add_ground(ax, extent)
        _plot_mesh(ax, obj_mesh, OBJ_COLOUR_BEFORE, label="Object" if view_i == 0 else None)
        _plot_mesh(ax, tool_cur, TOOL_COLOUR_AFTER, label="Denoising pose" if view_i == 0 else None)
        _plot_contact_point(ax, contact_pt, CONTACT_PT_COLOUR, size=80, label="Contact point" if view_i == 0 else None)

        if len(trajectory) > 1:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=TRAJECTORY_COLOUR,
                linewidth=2.0,
                alpha=0.9,
                label="Tool-origin path" if view_i == 0 else None,
            )
            ax.scatter(
                trajectory[-1:, 0],
                trajectory[-1:, 1],
                trajectory[-1:, 2],
                color=TRAJECTORY_COLOUR[:3],
                s=35,
                depthshade=False,
            )

        ax.set_title(view_name, fontsize=10, fontweight="bold", pad=8)
        _set_equal_aspect(ax, all_verts)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=view_elev, azim=view_azim)
        if view_i == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.75)

    fig.suptitle(
        f"{title}\nframe {frame_i + 1}/{n_frames}  t={n_frames - frame_i - 1}  "
        f"min_sdf={min_sdf * 1000.0:.2f}mm",
        fontsize=10,
        fontweight="bold",
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[:, :, :3].copy()


def render_movement_frame(
    fig,
    axes,
    obj_mesh,
    obj_after,
    tool_mesh,
    pred_R: np.ndarray,
    pred_t: np.ndarray,
    moved_R: np.ndarray,
    moved_t: np.ndarray,
    contact_pt: np.ndarray,
    anchor_new: np.ndarray,
    min_sdf: float,
    all_verts: np.ndarray,
    title: str,
) -> np.ndarray:
    tool_pred = transform_mesh(tool_mesh, pred_R, pred_t)
    tool_moved = transform_mesh(tool_mesh, moved_R, moved_t)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))
    for view_i, (ax, (view_elev, view_azim, view_name)) in enumerate(zip(axes, VIEWPOINTS)):
        ax.clear()
        _add_ground(ax, extent)
        _plot_mesh(ax, obj_mesh, OBJ_COLOUR_GHOST, edge_alpha=0.04, label="Object before move" if view_i == 0 else None)
        _plot_mesh(ax, tool_pred, TOOL_COLOUR_GHOST, edge_alpha=0.04, label="Pred tool before move" if view_i == 0 else None)
        _plot_mesh(ax, obj_after, OBJ_COLOUR_AFTER, label="Object after delta_O" if view_i == 0 else None)
        _plot_mesh(ax, tool_moved, TOOL_COLOUR_AFTER, label="Pred tool after delta_T" if view_i == 0 else None)
        _plot_contact_point(ax, contact_pt, CONTACT_PT_COLOUR, size=70, label="Contact point" if view_i == 0 else None)
        _plot_contact_point(ax, anchor_new, CONTACT_PT_AFTER, size=100, marker="*", label="Movement anchor" if view_i == 0 else None)

        ax.set_title(view_name, fontsize=10, fontweight="bold", pad=8)
        _set_equal_aspect(ax, all_verts)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=view_elev, azim=view_azim)
        if view_i == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.75)

    fig.suptitle(f"{title}\nmin_sdf={min_sdf * 1000.0:.2f}mm", fontsize=10, fontweight="bold")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[:, :, :3].copy()


def write_obj_mtl(
    obj_path: Path,
    meshes: list[trimesh.Trimesh],
    names: list[str],
    colours: list[tuple[float, float, float]],
):
    mtl_path = obj_path.with_suffix(".mtl")
    with mtl_path.open("w") as f:
        for name, (r, g, b) in zip(names, colours):
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write("Ka 0.1000 0.1000 0.1000\n")
            f.write("Ks 0.3000 0.3000 0.3000\n")
            f.write("Ns 50.0\n")
            f.write("d 1.0\n")
            f.write("illum 2\n\n")

    vertex_offset = 0
    with obj_path.open("w") as f:
        f.write("# new_pretrain diffusion visualization scene\n")
        f.write(f"mtllib {mtl_path.name}\n\n")
        for mesh, name in zip(meshes, names):
            f.write(f"o {name}\n")
            f.write(f"usemtl {name}\n")
            for v in mesh.vertices:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for face in mesh.faces:
                i0 = face[0] + vertex_offset + 1
                i1 = face[1] + vertex_offset + 1
                i2 = face[2] + vertex_offset + 1
                f.write(f"f {i0} {i1} {i2}\n")
            vertex_offset += len(mesh.vertices)
            f.write("\n")


def export_visualization_obj(
    output_path: Path,
    obj_mesh,
    tool_mesh,
    pred_R: np.ndarray,
    pred_t: np.ndarray,
):
    tool_pred = transform_mesh(tool_mesh, pred_R, pred_t)
    all_verts = np.concatenate([
        obj_mesh.vertices,
        tool_pred.vertices,
    ], axis=0)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))
    ground = trimesh.creation.box(extents=[extent * 2.5, extent * 2.5, 0.001])
    ground.apply_translation([0.0, 0.0, -0.0005])
    obj_path = output_path.with_suffix(".obj")
    write_obj_mtl(
        obj_path=obj_path,
        meshes=[obj_mesh, tool_pred, ground],
        names=["Object", "Pred_tool_contact", "Ground"],
        colours=[OBJ_PRED_COLOUR, TOOL_PRED_COLOUR, GROUND_COLOUR],
    )
    return obj_path


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
    delta_tool_t = sample["delta_tool_t"].unsqueeze(0).to(device)
    delta_tool_R = sample["delta_tool_R"].unsqueeze(0).to(device)
    delta_obj_t = sample["delta_obj_t"].unsqueeze(0).to(device)
    delta_obj_R = sample["delta_obj_R"].unsqueeze(0).to(device)
    movement_cond = torch.cat(
        [
            delta_tool_t,
            matrix_to_quaternion(delta_tool_R),
            delta_obj_t,
            matrix_to_quaternion(delta_obj_R),
        ],
        dim=-1,
    )

    poses_R, poses_t, stats = rollout_diffusion(
        model=model,
        tool_canonical=tool_canonical,
        obj_pc=obj_pc,
        contact_R=contact_R,
        contact_t=contact_t,
        movement_cond=movement_cond,
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
    tool_canonical_np = to_numpy(sample["tool_canonical"])
    obj_pc_np = to_numpy(sample["obj_pc"])
    contact_pt = to_numpy(data["contact_pts_world"][cfg_i, 0])
    pred_R = poses_R[-1]
    pred_t = poses_t[-1]
    delta_tool_R_np = to_numpy(sample["delta_tool_R"])
    delta_tool_t_np = to_numpy(sample["delta_tool_t"])
    delta_obj_R_np = to_numpy(sample["delta_obj_R"])
    delta_obj_t_np = to_numpy(sample["delta_obj_t"])
    anchor_new = to_numpy(data["movement_contact_pts"][cfg_i])
    moved_tool_R = delta_tool_R_np @ pred_R
    moved_tool_t = delta_tool_R_np @ pred_t + delta_tool_t_np
    obj_after = apply_delta_to_object(obj_mesh, delta_obj_R_np, delta_obj_t_np, anchor_new)
    obj_pc_after = apply_delta_to_points(obj_pc_np, delta_obj_R_np, delta_obj_t_np, anchor_new)
    final_min_sdf = min_sdf_np(tool_canonical_np, obj_pc_after, moved_tool_R, moved_tool_t)
    all_verts = np.concatenate([
        collect_bounds_vertices(obj_mesh, tool_mesh, poses_R, poses_t),
        obj_after.vertices,
        transform_mesh(tool_mesh, moved_tool_R, moved_tool_t).vertices,
    ], axis=0)
    obj_export_path = export_visualization_obj(
        output_path=output_path,
        obj_mesh=obj_mesh,
        tool_mesh=tool_mesh,
        pred_R=pred_R,
        pred_t=pred_t,
    )

    title = (
        f"{Path(data['tool_mesh_path']).stem} x {Path(data['object_mesh_path']).stem} "
        f"cfg {cfg_i}  head={cfg.head_mode}"
    )
    fig = plt.figure(figsize=(18, 6), dpi=args.dpi)
    fig.patch.set_facecolor("#f8f8f8")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    frames = []
    for frame_i, (cur_R, cur_t) in enumerate(zip(poses_R, poses_t)):
        trajectory = np.asarray(poses_t[:frame_i + 1])
        frame_min_sdf = min_sdf_np(tool_canonical_np, obj_pc_np, cur_R, cur_t)
        frame = render_frame(
            fig=fig,
            axes=axes,
            obj_mesh=obj_mesh,
            tool_mesh=tool_mesh,
            cur_R=cur_R,
            cur_t=cur_t,
            contact_pt=contact_pt,
            trajectory=trajectory,
            min_sdf=frame_min_sdf,
            all_verts=all_verts,
            frame_i=frame_i,
            n_frames=len(poses_R),
            title=title,
        )
        frames.append(frame)

    movement_title = (
        f"{title}\nfinal movement: apply delta_T to predicted tool and delta_O to object"
    )
    movement_frame = render_movement_frame(
        fig=fig,
        axes=axes,
        obj_mesh=obj_mesh,
        obj_after=obj_after,
        tool_mesh=tool_mesh,
        pred_R=pred_R,
        pred_t=pred_t,
        moved_R=moved_tool_R,
        moved_t=moved_tool_t,
        contact_pt=contact_pt,
        anchor_new=anchor_new,
        min_sdf=final_min_sdf,
        all_verts=all_verts,
        title=movement_title,
    )
    for _ in range(args.movement_frames):
        frames.append(movement_frame)

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
        f"Saved {output_path}  stats={stats_path}  obj={obj_export_path}  "
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
    parser.add_argument("--movement-frames", type=int, default=8, help="Frames appended after denoising with delta_T/delta_O applied.")
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
    dataset = NewPretrainDataset(files, augment=False, require_movement=True)
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

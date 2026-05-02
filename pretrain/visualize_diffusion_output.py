#!/usr/bin/env python3
"""
visualize_diffusion_output.py — Render model-predicted init → contact tool motion.

The model predicts a delta from the initial tool pose. This script applies that
delta to the init pose and renders:

  LEFT:  object + tool at init pose
  RIGHT: object + predicted contact tool pose, with ground-truth contact as a ghost

Usage:
    python visualize_diffusion_output.py --checkpoint checkpoints_diffusion_sdf_2/best.pt --input-dir tmp_data --num-samples 4
    python visualize_diffusion_output.py --checkpoint checkpoints_translation/best.pt --task translation --input config.pt
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAIN_DIR))

from config import TrainConfig
from dataset import (
    ContactDataset,
    DELTA_ROTATION_6D_NORM_SCALE,
    DELTA_TRANSLATION_NORM_SCALE,
)
from model import DiffusionModel
from visualize_movement_delta import (
    CONTACT_PT_AFTER,
    CONTACT_PT_COLOUR,
    OBJ_COLOUR_BEFORE,
    TOOL_COLOUR_AFTER,
    TOOL_COLOUR_BEFORE,
    TOOL_COLOUR_GHOST,
    _add_ground,
    _plot_arrow,
    _plot_contact_point,
    _plot_mesh,
    _set_equal_aspect,
    apply_object_pose,
    load_data,
    load_mesh_trimesh,
    transform_mesh,
)

def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    cols = rot_6d.reshape(3, 2)
    a1 = cols[:, 0]
    a2 = cols[:, 1]
    b1 = a1 / np.linalg.norm(a1)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / np.linalg.norm(a2_orth)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def detect_pose_dim(state_dict: dict) -> int:
    if "velocity_net.input_proj.0.weight" in state_dict:
        return state_dict["velocity_net.input_proj.0.weight"].shape[1]
    if "noise_predictor.input_proj.0.weight" in state_dict:
        return state_dict["noise_predictor.input_proj.0.weight"].shape[1]
    raise RuntimeError("Cannot infer diffusion pose dimension from checkpoint")


def detect_head_mode(state_dict: dict) -> str:
    point_keys = [
        "xyz_embed.0.weight",
        "sdf_head.xyz_embed.0.weight",
    ]
    for key in point_keys:
        if key in state_dict:
            return "point"
    return "patch"


def build_model(cfg: TrainConfig, pose_dim: int, device: torch.device) -> DiffusionModel:
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
        pose_dim=pose_dim,
        aux_pose_dim=3 if pose_dim == 3 else 9,
        aux_reg=cfg.aux_reg,
        sdf_weight=cfg.sdf_weight,
        diffusion_weight=cfg.diffusion_weight,
        aux_weight=cfg.aux_weight,
    ).to(device)


def load_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        return ckpt["model"]
    return ckpt


def collect_input_files(args) -> list[str]:
    if args.input:
        return [args.input]
    if args.input_dir:
        return sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    raise RuntimeError("Provide --input or --input-dir")


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def render_sample(
    fig,
    axes,
    obj_mesh,
    tool_mesh,
    init_R: np.ndarray,
    init_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    pred_R: np.ndarray,
    pred_t: np.ndarray,
    contact_pt: np.ndarray,
    title: str,
):
    tool_init = transform_mesh(tool_mesh, init_R, init_t)
    tool_gt = transform_mesh(tool_mesh, gt_R, gt_t)
    tool_pred = transform_mesh(tool_mesh, pred_R, pred_t)

    all_verts = np.concatenate([
        obj_mesh.vertices,
        tool_init.vertices,
        tool_gt.vertices,
        tool_pred.vertices,
    ], axis=0)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))

    ax_init = axes[0]
    _add_ground(ax_init, extent)
    _plot_mesh(ax_init, obj_mesh, OBJ_COLOUR_BEFORE, label="Object")
    _plot_mesh(ax_init, tool_init, TOOL_COLOUR_BEFORE, label="Tool init")
    _plot_contact_point(ax_init, contact_pt, CONTACT_PT_COLOUR, label="GT contact")
    ax_init.set_title("Init pose", fontsize=11, fontweight="bold", pad=8)
    _set_equal_aspect(ax_init, all_verts)
    ax_init.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax_init.view_init(elev=25, azim=-55)

    viewpoints = [
        (25, -55, "Pred contact (front)"),
        (60, -30, "Pred contact (top-front)"),
        (10, -120, "Pred contact (side)"),
    ]
    for i, (ax, (elev, azim, view_title)) in enumerate(zip(axes[1:], viewpoints)):
        _add_ground(ax, extent)
        _plot_mesh(ax, obj_mesh, OBJ_COLOUR_BEFORE, label="Object" if i == 0 else None)
        _plot_mesh(ax, tool_init, TOOL_COLOUR_GHOST, edge_alpha=0.05, label="Tool init" if i == 0 else None)
        _plot_mesh(ax, tool_gt, (0.45, 0.80, 0.55, 0.35), edge_alpha=0.05, label="Tool GT contact" if i == 0 else None)
        _plot_mesh(ax, tool_pred, TOOL_COLOUR_AFTER, label="Tool pred contact" if i == 0 else None)
        _plot_contact_point(ax, contact_pt, CONTACT_PT_COLOUR, size=60, label="GT contact" if i == 0 else None)
        _plot_contact_point(ax, pred_t, CONTACT_PT_AFTER, size=80, marker="o", label="Pred tool origin" if i == 0 else None)

        direction = pred_t - init_t
        d_norm = np.linalg.norm(direction)
        if d_norm > 1e-6:
            _plot_arrow(ax, init_t, direction / d_norm, d_norm, (1.0, 0.4, 0.0), lw=2.5)

        ax.set_title(view_title, fontsize=11, fontweight="bold", pad=8)
        _set_equal_aspect(ax, all_verts)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(title, fontsize=10, y=0.98, fontweight="bold", color="#333")


def main():
    parser = argparse.ArgumentParser(description="Visualise diffusion model init → predicted contact pose.")
    parser.add_argument("--checkpoint", required=True, help="Diffusion checkpoint")
    parser.add_argument("--input", type=str, help="Single .pt file")
    parser.add_argument("--input-dir", type=str, help="Directory of .pt files")
    parser.add_argument("--config", default="config.py", help="Path to config.py")
    parser.add_argument("--task", choices=["auto", "diffusion", "translation"], default="auto")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20, help="Euler inference steps")
    parser.add_argument("--save", type=str, default="diffusion_output.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.config and Path(args.config).exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("visualize_config", args.config)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        cfg = cfg_module.TrainConfig()
    else:
        cfg = TrainConfig()
    state_dict = load_state_dict(args.checkpoint)
    detected_pose_dim = detect_pose_dim(state_dict)
    cfg.head_mode = detect_head_mode(state_dict)
    pose_dim = detected_pose_dim
    if args.task == "diffusion":
        pose_dim = 9
    if args.task == "translation":
        pose_dim = 3

    device = torch.device(args.device)
    model = build_model(cfg, pose_dim, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    input_files = collect_input_files(args)
    dataset = ContactDataset(input_files, augment=False)
    skipped_files = dataset.get_skipped_files()
    if skipped_files:
        print(f"Skipped {len(skipped_files)} corrupted/unreadable files:")
        for path in skipped_files:
            print(f"  {path}")

    valid_files = []
    for path, data in dataset._pt_cache.items():
        if "init_translations" in data and "init_rotations" in data:
            valid_files.append(path)
    if not valid_files:
        raise RuntimeError("No .pt files with init pose data found")

    rng = random.Random(args.seed)
    if len(valid_files) == 1:
        data = dataset._pt_cache[valid_files[0]]
        n = data["tool_translations"].shape[0]
        config_indices = rng.sample(range(n), min(args.num_samples, n))
        sampled_items = [(valid_files[0], cfg_i) for cfg_i in config_indices]
    else:
        sampled_files = rng.sample(valid_files, min(args.num_samples, len(valid_files)))
        sampled_items = []
        for path in sampled_files:
            data = dataset._pt_cache[path]
            n = data["tool_translations"].shape[0]
            sampled_items.append((path, rng.randint(0, n - 1)))

    fig = plt.figure(figsize=(28, 7 * len(sampled_items)), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")

    mesh_cache = {}
    for row, (pt_path, cfg_i) in enumerate(sampled_items):
        data = dataset._pt_cache[pt_path]
        item_index = dataset._index.index((pt_path, cfg_i))
        sample = dataset[item_index]

        tool_pc_init = sample["tool_pc_init"].unsqueeze(0).to(device)
        obj_pc = sample["obj_pc"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.sample(tool_pc_init, obj_pc, n_steps=args.steps)[0].cpu().numpy()

        pred_delta_t = pred[:3] * DELTA_TRANSLATION_NORM_SCALE
        init_t = to_numpy(data["init_translations"][cfg_i])
        init_R = to_numpy(data["init_rotations"][cfg_i])
        gt_t = to_numpy(data["tool_translations"][cfg_i])
        gt_R = to_numpy(data["tool_rotations"][cfg_i])

        if pose_dim == 3:
            pred_R = init_R
            pred_t = init_t + pred_delta_t
        else:
            pred_delta_R = rotation_6d_to_matrix(pred[3:9] * DELTA_ROTATION_6D_NORM_SCALE)
            pred_R = pred_delta_R @ init_R
            pred_t = init_t + pred_delta_t

        if pt_path not in mesh_cache:
            obj_path = data["object_mesh_path"]
            tool_path = data["tool_mesh_path"]
            obj_mesh_raw = load_mesh_trimesh(obj_path)
            tool_mesh_raw = load_mesh_trimesh(tool_path)
            obj_mesh_raw.vertices *= to_numpy(data["object_scale"])
            tool_mesh_raw.vertices *= to_numpy(data["tool_scale"])
            obj_mesh = apply_object_pose(
                obj_mesh_raw,
                to_numpy(data["object_rotation"]),
                to_numpy(data["obj_z_shift"]),
            )
            mesh_cache[pt_path] = (obj_mesh, tool_mesh_raw, obj_path, tool_path)
        obj_mesh, tool_mesh, obj_path, tool_path = mesh_cache[pt_path]

        axes = [
            fig.add_subplot(len(sampled_items), 4, row * 4 + 1, projection="3d"),
            fig.add_subplot(len(sampled_items), 4, row * 4 + 2, projection="3d"),
            fig.add_subplot(len(sampled_items), 4, row * 4 + 3, projection="3d"),
            fig.add_subplot(len(sampled_items), 4, row * 4 + 4, projection="3d"),
        ]
        trans_err = np.linalg.norm(pred_t - gt_t) * 1000.0
        title = (
            f"{Path(tool_path).stem} x {Path(obj_path).stem}  config {cfg_i}\n"
            f"pred |dt|={np.linalg.norm(pred_delta_t) * 1000.0:.1f}mm  "
            f"translation error={trans_err:.1f}mm  pose_dim={pose_dim}"
        )
        render_sample(
            fig,
            axes,
            obj_mesh,
            tool_mesh,
            init_R,
            init_t,
            gt_R,
            gt_t,
            pred_R,
            pred_t,
            to_numpy(data["contact_pts_world"][cfg_i, 0]),
            title,
        )
        print(f"{Path(pt_path).name} cfg {cfg_i}: pred |dt|={np.linalg.norm(pred_delta_t) * 1000.0:.1f}mm, err={trans_err:.1f}mm")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

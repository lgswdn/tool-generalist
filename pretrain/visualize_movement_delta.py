#!/usr/bin/env python3
"""
visualize_movement_delta.py — Render before/after movement-delta configurations.

For each selected config, renders two panels side-by-side:

  LEFT  ("Before"):  Original contact config  (tool + object + contact point P)
  RIGHT ("After"):   Tool after ΔT applied, object after ΔO applied,
                     and the anchor point P moved with the tool.

All geometry is rendered as solid meshes (not point clouds) for clarity.

Usage:
    # Random samples from a single .pt file
    python visualize_movement_delta.py --input config.pt --num-samples 3

    # Random samples from a directory (one config per file)
    python visualize_movement_delta.py --input-dir tmp_data/ --num-samples 4

    # Save to disk instead of interactive display
    python visualize_movement_delta.py --input-dir tmp_data/ --save viz_deltas.png
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

# ═══════════════════════════════════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════════════════════════════════

# Before state
TOOL_COLOUR_BEFORE  = (0.35, 0.55, 0.90, 0.70)   # steel blue, semi-transparent
OBJ_COLOUR_BEFORE   = (0.65, 0.65, 0.70, 0.85)   # cool grey, solid

# After state
TOOL_COLOUR_AFTER   = (0.90, 0.40, 0.35, 0.70)   # coral red, semi-transparent
OBJ_COLOUR_AFTER    = (0.45, 0.80, 0.55, 0.85)   # jade green, solid

# Ghost (original position shown in "After" panel)
TOOL_COLOUR_GHOST   = (0.35, 0.55, 0.90, 0.35)   # blue (original tool position)
OBJ_COLOUR_GHOST    = (0.85, 0.65, 0.35, 0.35)   # amber/tan, contrasts with green

# Contact point markers
CONTACT_PT_COLOUR   = (1.0, 0.85, 0.0)            # gold
CONTACT_PT_AFTER    = (1.0, 0.35, 0.0)            # orange-red

ARROW_SCALE = 0.015  # normal arrow length


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(pt_path: str) -> dict:
    import torch
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy()
        else:
            out[k] = v
    return out


def load_mesh_trimesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh", process=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mesh transform helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_angle(R: np.ndarray) -> float:
    """Compute rotation angle in radians from rotation matrix (0 to π)."""
    trace = np.trace(R)
    # Clamp to [-1, 3] to handle numerical errors
    trace = np.clip(trace, -1.0, 3.0)
    return np.arccos((trace - 1.0) / 2.0)


def transform_mesh(mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices = m.vertices @ R.T + t
    return m


def apply_object_pose(mesh: trimesh.Trimesh, R_obj: np.ndarray, z_shift: float) -> trimesh.Trimesh:
    """Apply object rotation + z_shift grounding (same as contact_gen)."""
    m = mesh.copy()
    m.vertices = m.vertices @ R_obj.T
    m.vertices[:, 2] -= z_shift
    return m


def apply_delta_to_object(
    mesh: trimesh.Trimesh,
    delta_R: np.ndarray,     # (3, 3) ΔO rotation
    delta_t: np.ndarray,     # (3,)   ΔO translation (computed from rotation)
    pivot: np.ndarray,       # (3,)   pivot point (P_anchor_new)
) -> trimesh.Trimesh:
    """Apply SE(3) delta to object mesh, pivoting around anchor point.

    The transform is: obj_new = delta_R @ (obj - pivot) + pivot + delta_t

    This matches gen_movement_delta.py where delta_t is computed analytically:
        delta_t = -delta_R @ (contact_pts_original - pivot)
    """
    m = mesh.copy()
    # Apply rotation around pivot, then add translation
    m.vertices = (m.vertices - pivot) @ delta_R.T + pivot + delta_t
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Matplotlib rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_mesh(ax, mesh: trimesh.Trimesh, colour, alpha=None, label=None, edge_alpha=0.15):
    """Render a triangle mesh as solid polygons."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = mesh.vertices[mesh.faces]
    a = alpha if alpha is not None else (colour[3] if len(colour) == 4 else 1.0)
    pc = Poly3DCollection(
        polys, alpha=a,
        facecolor=colour[:3],
        edgecolor=(0.15, 0.15, 0.15, edge_alpha),
        linewidth=0.15,
    )
    ax.add_collection3d(pc)
    if label:
        ax.scatter([], [], [], color=colour[:3], label=label, s=40, alpha=0.9)


def _plot_contact_point(ax, pt: np.ndarray, colour, size=100, marker="*", label=None):
    """Draw a prominent contact-point marker."""
    ax.scatter(
        [pt[0]], [pt[1]], [pt[2]],
        c=[colour[:3]], s=size, marker=marker,
        edgecolors="black", linewidths=0.8,
        zorder=10, depthshade=False, label=label,
    )


def _plot_arrow(ax, origin: np.ndarray, direction: np.ndarray, length: float, colour, lw=2.0):
    """Draw a direction arrow from a point."""
    end = origin + direction * length
    ax.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        [origin[2], end[2]],
        color=colour, linewidth=lw, alpha=0.9,
    )
    ax.scatter(
        [end[0]], [end[1]], [end[2]],
        color=colour, s=20, marker=">", depthshade=False,
    )


def _add_ground(ax, extent: float):
    g = extent * 1.2
    xx, yy = np.meshgrid(np.linspace(-g, g, 2), np.linspace(-g, g, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color="grey")


def _set_equal_aspect(ax, all_verts: np.ndarray, pad: float = 1.15):
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)
    centre = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * pad
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(max(-0.005, centre[2] - span), centre[2] + span)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main rendering logic
# ═══════════════════════════════════════════════════════════════════════════════

def render_one_sample(
    fig,
    ax_before,
    ax_after_list,                       # list of 3 axes for different viewpoints
    obj_mesh_posed: trimesh.Trimesh,     # object in world frame (before ΔO)
    tool_mesh_raw: trimesh.Trimesh,      # tool mesh at canonical scale
    tool_R: np.ndarray,                  # (3, 3) original tool rotation
    tool_t: np.ndarray,                  # (3,)   original tool translation
    contact_pt: np.ndarray,              # (3,)   anchor contact point (world frame)
    delta_tool_R: np.ndarray,            # (3, 3) ΔT rotation
    delta_tool_t: np.ndarray,            # (3,)   ΔT translation
    delta_obj_R: np.ndarray,             # (3, 3) ΔO rotation
    delta_obj_t: np.ndarray,             # (3,)   ΔO translation
    anchor_pt_new: np.ndarray,           # (3,)   anchor after moving with tool
    title: str = "",
):
    """Render before/after panels for one (ΔT, ΔO) pair with 3 viewpoints for after state."""

    # ---- Posed meshes ----
    tool_before = transform_mesh(tool_mesh_raw, tool_R, tool_t)

    # After ΔT: R_new = ΔR @ R, t_new = ΔR @ t + Δt
    tool_R_new = delta_tool_R @ tool_R
    tool_t_new = delta_tool_R @ tool_t + delta_tool_t
    tool_after = transform_mesh(tool_mesh_raw, tool_R_new, tool_t_new)

    # After ΔO: object moves
    obj_after = apply_delta_to_object(obj_mesh_posed, delta_obj_R, delta_obj_t, anchor_pt_new)

    # Collect all vertices for axis scaling
    all_verts = np.concatenate([
        obj_mesh_posed.vertices, tool_before.vertices,
        tool_after.vertices, obj_after.vertices,
    ], axis=0)
    extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))

    # ════════════  LEFT PANEL: Before  ════════════
    _add_ground(ax_before, extent)
    _plot_mesh(ax_before, obj_mesh_posed, OBJ_COLOUR_BEFORE, label="Object")
    _plot_mesh(ax_before, tool_before, TOOL_COLOUR_BEFORE, label="Tool")
    _plot_contact_point(ax_before, contact_pt, CONTACT_PT_COLOUR, label="Contact P")

    ax_before.set_title("Before ΔT", fontsize=11, fontweight="bold", pad=8)
    _set_equal_aspect(ax_before, all_verts)
    ax_before.set_xlabel("X", fontsize=8)
    ax_before.set_ylabel("Y", fontsize=8)
    ax_before.set_zlabel("Z", fontsize=8)
    ax_before.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax_before.view_init(elev=25, azim=-55)

    # ════════════  RIGHT PANELS: After (3 viewpoints)  ════════════
    viewpoints = [
        (25, -55, "After (front)"),
        (60, -30, "After (top-front)"),
        (10, -120, "After (side)"),
    ]

    for i, (ax_after, (elev, azim, view_title)) in enumerate(zip(ax_after_list, viewpoints)):
        _add_ground(ax_after, extent)

        # Ghost of original positions
        _plot_mesh(ax_after, obj_mesh_posed, OBJ_COLOUR_GHOST, edge_alpha=0.05, label="Object (orig)" if i == 0 else None)
        _plot_mesh(ax_after, tool_before, TOOL_COLOUR_GHOST, edge_alpha=0.05, label="Tool (orig)" if i == 0 else None)

        # New positions (solid)
        _plot_mesh(ax_after, obj_after, OBJ_COLOUR_AFTER, label="Object after ΔO" if i == 0 else None)
        _plot_mesh(ax_after, tool_after, TOOL_COLOUR_AFTER, label="Tool after ΔT" if i == 0 else None)

        # Anchor points
        _plot_contact_point(ax_after, contact_pt, CONTACT_PT_COLOUR, size=60, label="P (orig)" if i == 0 else None)
        _plot_contact_point(ax_after, anchor_pt_new, CONTACT_PT_AFTER, size=120, marker="*", label="P (moved)" if i == 0 else None)

        # Arrow: P_orig → P_new (shows the push)
        direction = anchor_pt_new - contact_pt
        d_norm = np.linalg.norm(direction)
        if d_norm > 1e-6:
            _plot_arrow(ax_after, contact_pt, direction / d_norm, d_norm, (1.0, 0.4, 0.0), lw=2.5)

        ax_after.set_title(view_title, fontsize=11, fontweight="bold", pad=8)
        _set_equal_aspect(ax_after, all_verts)
        ax_after.set_xlabel("X", fontsize=8)
        ax_after.set_ylabel("Y", fontsize=8)
        ax_after.set_zlabel("Z", fontsize=8)
        if i == 0:
            ax_after.legend(loc="upper left", fontsize=7, framealpha=0.7)
        ax_after.view_init(elev=elev, azim=azim)

    if title:
        fig.suptitle(title, fontsize=10, y=0.98, fontweight="bold", color="#333")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise (ΔT, ΔO) movement-delta before/after.",
    )
    parser.add_argument("--input", type=str, help="Single .pt file")
    parser.add_argument("--input-dir", type=str, help="Directory of .pt files")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of random samples to visualise (default: 3)")
    parser.add_argument("--save", type=str, default="movement.png",
                        help="Save figure to path instead of interactive display")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sample selection")
    parser.add_argument("--max-rotation", action="store_true",
                        help="Select samples with maximum ΔO rotation magnitude")
    parser.add_argument("--push-down", action="store_true",
                        help="Select samples with clear push-down (negative z in ΔT)")
    parser.add_argument("--min-z-delta", type=float, default=0.02,
                        help="Minimum z delta (meters) for push-down selection (default: 0.02)")
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Collect .pt files ----
    if args.input:
        all_files = [args.input]
    elif args.input_dir:
        all_files = sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    else:
        print("ERROR: Must provide --input or --input-dir")
        sys.exit(1)

    # Filter files that have movement deltas
    valid_files = []
    for f in all_files:
        data = load_data(f)
        if "delta_tool_translations" in data:
            valid_files.append(f)
    if not valid_files:
        print("ERROR: No .pt files found with movement delta data.")
        print("  Run gen_movement_delta.py first.")
        sys.exit(1)

    print(f"Found {len(valid_files)} files with movement deltas (out of {len(all_files)} total)")

    # ---- Sample files ----
    rng = random.Random(args.seed)

    # Compute rotation angles for ΔO and select samples with max rotation
    if args.max_rotation:
        sampled_items = []
        for pt_path in valid_files:
            data = load_data(pt_path)
            N = data["tool_translations"].shape[0]
            delta_obj_R = data["delta_obj_rotations"]  # (N, 3, 3)

            # Compute rotation angles for all configs
            angles = np.array([rotation_angle(delta_obj_R[i]) for i in range(N)])

            # Get indices sorted by rotation angle (descending)
            sorted_indices = np.argsort(angles)[::-1]
            top_indices = sorted_indices[:args.num_samples]

            for cfg_i in top_indices:
                sampled_items.append((pt_path, int(cfg_i)))
                print(f"  {Path(pt_path).name} cfg {cfg_i}: ΔO rotation = {angles[cfg_i]*180/np.pi:.1f}°")

        # Limit total samples
        sampled_items = sampled_items[:args.num_samples]
        print(f"Selected {len(sampled_items)} configs with maximum ΔO rotation")

    # Select samples with push-down movement (negative z in ΔT)
    elif args.push_down:
        sampled_items = []
        push_down_candidates = []

        for pt_path in valid_files:
            data = load_data(pt_path)
            N = data["tool_translations"].shape[0]
            delta_tool_t = data["delta_tool_translations"]  # (N, 3)

            # Find configs with negative z delta (push down)
            for cfg_i in range(N):
                z_delta = delta_tool_t[cfg_i, 2]  # z component
                if z_delta < -args.min_z_delta:  # push down threshold
                    push_down_candidates.append((pt_path, cfg_i, z_delta))

        # Sort by z delta (most negative first) and take top samples
        push_down_candidates.sort(key=lambda x: x[2])  # sort by z (ascending, most negative first)
        for pt_path, cfg_i, z_delta in push_down_candidates[:args.num_samples]:
            sampled_items.append((pt_path, cfg_i))
            print(f"  {Path(pt_path).name} cfg {cfg_i}: ΔT_z = {z_delta*1000:.1f}mm (push down)")

        print(f"Selected {len(sampled_items)} configs with push-down movement")

    elif len(valid_files) == 1:
        # When only one file, sample multiple configs from it
        pt_path = valid_files[0]
        data = load_data(pt_path)
        N = data["tool_translations"].shape[0]
        n_configs = min(args.num_samples, N)
        config_indices = rng.sample(range(N), n_configs)
        sampled_items = [(pt_path, cfg_i) for cfg_i in config_indices]
        print(f"Single file: sampling {n_configs} configs from {N} available")
    else:
        n_files = min(args.num_samples, len(valid_files))
        sampled_files = rng.sample(valid_files, n_files)
        sampled_items = []
        for pt_path in sampled_files:
            data = load_data(pt_path)
            N = data["tool_translations"].shape[0]
            cfg_i = rng.randint(0, N - 1)
            sampled_items.append((pt_path, cfg_i))
        print(f"Multiple files: sampling 1 config from each of {n_files} files")

    n_rows = len(sampled_items)

    # ---- Create figure: n_rows × 4 columns (before + 3 viewpoints) ----
    fig = plt.figure(figsize=(28, 7 * n_rows), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")

    # Pre-load mesh data for reuse when single file has multiple configs
    mesh_cache = {}

    for row, (pt_path, cfg_i) in enumerate(sampled_items):
        print(f"\n[{row+1}/{n_rows}] {Path(pt_path).name}  config {cfg_i}")

        # ---- Load data (cache to avoid re-loading same file) ----
        if pt_path not in mesh_cache:
            data = load_data(pt_path)
            obj_path = data["object_mesh_path"]
            tool_path = data["tool_mesh_path"]

            if not Path(obj_path).exists():
                print(f"  [SKIP] Object mesh not found: {obj_path}")
                continue
            if not Path(tool_path).exists():
                print(f"  [SKIP] Tool mesh not found: {tool_path}")
                continue

            obj_mesh_raw = load_mesh_trimesh(obj_path)
            tool_mesh_raw = load_mesh_trimesh(tool_path)

            # Scales
            tool_scale = data.get("tool_scale", 0.1)
            obj_scale = data.get("object_scale", 0.15)
            tool_mesh_raw.vertices *= tool_scale
            obj_mesh_raw.vertices *= obj_scale

            # Object world-frame pose
            R_obj = data.get("object_rotation", np.eye(3))
            z_shift = data.get("obj_z_shift", 0.0)
            if hasattr(z_shift, "item"):
                z_shift = z_shift.item()
            obj_mesh_posed = apply_object_pose(obj_mesh_raw, R_obj, z_shift)

            mesh_cache[pt_path] = {
                "data": data,
                "obj_mesh_posed": obj_mesh_posed,
                "tool_mesh_raw": tool_mesh_raw,
                "obj_path": obj_path,
                "tool_path": tool_path,
            }
        else:
            cache = mesh_cache[pt_path]
            data = cache["data"]
            obj_mesh_posed = cache["obj_mesh_posed"]
            tool_mesh_raw = cache["tool_mesh_raw"]
            obj_path = cache["obj_path"]
            tool_path = cache["tool_path"]

        # ---- Extract config data ----
        tool_R = data["tool_rotations"][cfg_i]          # (3, 3)
        tool_t = data["tool_translations"][cfg_i]       # (3,)
        contact_pt = data["contact_pts_world"][cfg_i, 0]  # (3,) first contact pt

        delta_tool_R = data["delta_tool_rotations"][cfg_i]  # (3, 3)
        delta_tool_t = data["delta_tool_translations"][cfg_i]  # (3,)
        delta_obj_R = data["delta_obj_rotations"][cfg_i]    # (3, 3)
        delta_obj_t = data["delta_obj_translations"][cfg_i]  # (3,)
        anchor_new = data["movement_contact_pts"][cfg_i]     # (3,)

        # ---- Axes: 1 before + 3 after viewpoints ----
        ax_before = fig.add_subplot(n_rows, 4, row * 4 + 1, projection="3d")
        ax_after_list = [
            fig.add_subplot(n_rows, 4, row * 4 + 2, projection="3d"),
            fig.add_subplot(n_rows, 4, row * 4 + 3, projection="3d"),
            fig.add_subplot(n_rows, 4, row * 4 + 4, projection="3d"),
        ]

        # Compute rotation angles
        delta_obj_rot_angle = rotation_angle(delta_obj_R) * 180 / np.pi  # in degrees
        delta_tool_rot_angle = rotation_angle(delta_tool_R) * 180 / np.pi

        title = (
            f"{Path(tool_path).stem}  ×  {Path(obj_path).stem}\n"
            f"|Δt_T|={np.linalg.norm(delta_tool_t)*1000:.1f}mm  ΔR_T={delta_tool_rot_angle:.1f}°   "
            f"|Δt_O|={np.linalg.norm(delta_obj_t)*1000:.1f}mm  ΔR_O={delta_obj_rot_angle:.1f}°"
        )

        render_one_sample(
            fig, ax_before, ax_after_list,
            obj_mesh_posed, tool_mesh_raw,
            tool_R, tool_t, contact_pt,
            delta_tool_R, delta_tool_t,
            delta_obj_R, delta_obj_t,
            anchor_new, title=title,
        )
        print(f"  |ΔT| = {np.linalg.norm(delta_tool_t)*1000:.1f} mm, ΔR_T = {delta_tool_rot_angle:.1f}°  "
              f"|ΔO| = {np.linalg.norm(delta_obj_t)*1000:.1f} mm, ΔR_O = {delta_obj_rot_angle:.1f}°")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nSaved to {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

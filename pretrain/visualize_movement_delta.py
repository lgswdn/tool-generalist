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

# Ghost (original position shown as wireframe-like overlay in "After" panel)
TOOL_COLOUR_GHOST   = (0.35, 0.55, 0.90, 0.12)   # faint blue
OBJ_COLOUR_GHOST    = (0.65, 0.65, 0.70, 0.12)   # faint grey

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
    delta_t: np.ndarray,     # (3,)   ΔO translation
    pivot: np.ndarray,       # (3,)   pivot point (P_anchor_new)
) -> trimesh.Trimesh:
    """Apply SE(3) delta to object mesh, pivoting around anchor point."""
    m = mesh.copy()
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
    ax_after,
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
    """Render before/after panels for one (ΔT, ΔO) pair."""

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

    # ════════════  RIGHT PANEL: After  ════════════
    _add_ground(ax_after, extent)

    # Ghost of original positions (faint)
    _plot_mesh(ax_after, obj_mesh_posed, OBJ_COLOUR_GHOST, edge_alpha=0.05, label="Object (orig)")
    _plot_mesh(ax_after, tool_before, TOOL_COLOUR_GHOST, edge_alpha=0.05, label="Tool (orig)")

    # New positions (solid)
    _plot_mesh(ax_after, obj_after, OBJ_COLOUR_AFTER, label="Object after ΔO")
    _plot_mesh(ax_after, tool_after, TOOL_COLOUR_AFTER, label="Tool after ΔT")

    # Anchor points
    _plot_contact_point(ax_after, contact_pt, CONTACT_PT_COLOUR, size=60, label="P (orig)")
    _plot_contact_point(ax_after, anchor_pt_new, CONTACT_PT_AFTER, size=120, marker="*", label="P (moved)")

    # Arrow: P_orig → P_new (shows the push)
    direction = anchor_pt_new - contact_pt
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-6:
        _plot_arrow(ax_after, contact_pt, direction / d_norm, d_norm, (1.0, 0.4, 0.0), lw=2.5)

    ax_after.set_title("After ΔT + ΔO", fontsize=11, fontweight="bold", pad=8)
    _set_equal_aspect(ax_after, all_verts)
    ax_after.set_xlabel("X", fontsize=8)
    ax_after.set_ylabel("Y", fontsize=8)
    ax_after.set_zlabel("Z", fontsize=8)
    ax_after.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax_after.view_init(elev=25, azim=-55)

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
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of random samples to visualise (default: 3)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to path instead of interactive display")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sample selection")
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
    n_samples = min(args.num_samples, len(valid_files))
    sampled_files = rng.sample(valid_files, n_samples)

    # ---- Create figure: n_samples rows × 2 columns ----
    fig = plt.figure(figsize=(16, 7 * n_samples), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")

    for row, pt_path in enumerate(sampled_files):
        print(f"\n[{row+1}/{n_samples}] Loading {pt_path}")
        data = load_data(pt_path)

        N = data["tool_translations"].shape[0]
        cfg_i = rng.randint(0, N - 1)
        print(f"  Config {cfg_i}/{N}")

        # ---- Load meshes ----
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

        # ---- Extract config data ----
        tool_R = data["tool_rotations"][cfg_i]          # (3, 3)
        tool_t = data["tool_translations"][cfg_i]       # (3,)
        contact_pt = data["contact_pts_world"][cfg_i, 0]  # (3,) first contact pt

        delta_tool_R = data["delta_tool_rotations"][cfg_i]  # (3, 3)
        delta_tool_t = data["delta_tool_translations"][cfg_i]  # (3,)
        delta_obj_R = data["delta_obj_rotations"][cfg_i]    # (3, 3)
        delta_obj_t = data["delta_obj_translations"][cfg_i]  # (3,)
        anchor_new = data["movement_contact_pts"][cfg_i]     # (3,)

        # ---- Axes ----
        ax_before = fig.add_subplot(n_samples, 2, row * 2 + 1, projection="3d")
        ax_after  = fig.add_subplot(n_samples, 2, row * 2 + 2, projection="3d")

        title = (
            f"{Path(tool_path).stem}  ×  {Path(obj_path).stem}\n"
            f"|Δt_T|={np.linalg.norm(delta_tool_t)*1000:.1f}mm   "
            f"|Δt_O|={np.linalg.norm(delta_obj_t)*1000:.1f}mm"
        )

        render_one_sample(
            fig, ax_before, ax_after,
            obj_mesh_posed, tool_mesh_raw,
            tool_R, tool_t, contact_pt,
            delta_tool_R, delta_tool_t,
            delta_obj_R, delta_obj_t,
            anchor_new, title=title,
        )
        print(f"  |ΔT| = {np.linalg.norm(delta_tool_t)*1000:.1f} mm,  "
              f"|ΔO| = {np.linalg.norm(delta_obj_t)*1000:.1f} mm")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nSaved to {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

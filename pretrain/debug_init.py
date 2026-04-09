#!/usr/bin/env python3
"""
debug_init.py  –  Visualise mesh scales & initial tool poses (before optimisation).

Usage:
    python debug_init.py \
        --object path/to/object.obj \
        --tool   path/to/tool.obj \
        --device cuda:4 \
        --num-tools 8 \
        --save debug_init.png
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse helpers from contact_gen
from contact_gen import (
    load_mesh,
    sample_surface_points,
    randomise_object_pose,
    random_rotation_matrices,
    _project_orientation,
    rot6d_to_matrix,
    matrix_to_rot6d,
)


def print_mesh_stats(name: str, verts: torch.Tensor):
    """Print bounding box, centroid, and extent of a mesh."""
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    extent = vmax - vmin
    centroid = verts.mean(dim=0)
    print(f"\n  [{name}]")
    print(f"    Num verts:  {verts.shape[0]}")
    print(f"    BBox min:   ({vmin[0]:.4f}, {vmin[1]:.4f}, {vmin[2]:.4f})")
    print(f"    BBox max:   ({vmax[0]:.4f}, {vmax[1]:.4f}, {vmax[2]:.4f})")
    print(f"    Extent:     ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})")
    print(f"    Max extent: {extent.max().item():.4f}")
    print(f"    Centroid:   ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")


def plot_points(ax, pts, color, label, size=0.3, alpha=0.5):
    """Scatter plot of a point cloud."""
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=[color], s=size, alpha=alpha, label=label)


def main():
    p = argparse.ArgumentParser(description="Debug initial pose generation")
    p.add_argument("--object", required=True)
    p.add_argument("--tool", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-tools", type=int, default=2)
    p.add_argument("--num-pts", type=int, default=1024)
    p.add_argument("--save", default="debug_init.png")
    args = p.parse_args()

    device = args.device
    N = args.num_tools
    torch.manual_seed(42)

    # ---- Load meshes ----
    print("=" * 60)
    print("MESH DIAGNOSTICS")
    print("=" * 60)

    obj_verts, obj_faces = load_mesh(args.object, device)
    print_mesh_stats("Object (raw)", obj_verts)

    tool_verts, tool_faces = load_mesh(args.tool, device)
    print_mesh_stats("Tool (raw)", tool_verts)

    # ---- Ground object ----
    obj_verts, R_obj = randomise_object_pose(obj_verts, obj_faces)
    print_mesh_stats("Object (grounded)", obj_verts)

    # ---- Sample & zero-centre tool points ----
    tool_pts = sample_surface_points(tool_verts, tool_faces, args.num_pts)
    tool_centroid = tool_pts.mean(dim=0)
    tool_pts_centered = tool_pts - tool_centroid.unsqueeze(0)
    print_mesh_stats("Tool points (raw)", tool_pts)
    print_mesh_stats("Tool points (zero-centred)", tool_pts_centered)

    # ---- Generate init poses (same logic as contact_gen) ----
    OFFSET_MAX = 0.05
    surf_pts = sample_surface_points(obj_verts, obj_faces, N)
    obj_centre = obj_verts.mean(dim=0)
    normals = F.normalize(surf_pts - obj_centre.unsqueeze(0), dim=-1)
    offset = torch.rand(N, 1, device=device) * OFFSET_MAX
    t = surf_pts + normals * offset

    R = random_rotation_matrices(N, device)
    R = _project_orientation(R)

    # Floor guard
    transformed = torch.einsum("pi, nji -> npj", tool_pts_centered, R) + t.unsqueeze(1)
    z_mins = transformed[:, :, 2].min(dim=1).values
    lift = torch.clamp(-z_mins, min=0.0)
    t[:, 2] += lift

    # ---- Print init stats ----
    print("\n" + "=" * 60)
    print("INITIAL POSE STATS")
    print("=" * 60)
    for i in range(N):
        pts_i = torch.einsum("pi, ji -> pj", tool_pts_centered, R[i]) + t[i]
        dmin = pts_i.min(dim=0).values
        dmax = pts_i.max(dim=0).values
        print(f"  Tool #{i}: t=({t[i,0]:.3f}, {t[i,1]:.3f}, {t[i,2]:.3f})  "
              f"z_range=[{dmin[2]:.3f}, {dmax[2]:.3f}]")

    # ---- Visualise ----
    print(f"\nRendering {N} initial tool poses → {args.save}")

    obj_pts_np = sample_surface_points(obj_verts, obj_faces, 4096).cpu().numpy()

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Object point cloud
    plot_points(ax, obj_pts_np, "steelblue", "Object", size=0.5, alpha=0.3)

    # Tool poses
    colours = plt.cm.Set1(np.linspace(0, 1, N))
    for i in range(N):
        pts_i = torch.einsum("pi, ji -> pj", tool_pts_centered, R[i]) + t[i]
        pts_np = pts_i.cpu().numpy()
        plot_points(ax, pts_np, colours[i], f"Tool #{i}", size=1.0, alpha=0.6)

    # Translation markers
    t_np = t.cpu().numpy()
    ax.scatter(t_np[:, 0], t_np[:, 1], t_np[:, 2],
               c="red", s=50, marker="x", label="Trans origins")

    # Equal axes
    all_pts = np.concatenate([obj_pts_np] +
        [torch.einsum("pi,ji->pj", tool_pts_centered, R[i]).add(t[i]).cpu().numpy()
         for i in range(N)], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centres = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.15
    ax.set_xlim(centres[0] - span, centres[0] + span)
    ax.set_ylim(centres[1] - span, centres[1] + span)
    ax.set_zlim(max(0, centres[2] - span), centres[2] + span)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Debug: Initial Tool Poses (before optimisation)", fontsize=13)
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"✓ Saved to {args.save}")


if __name__ == "__main__":
    main()

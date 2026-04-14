#!/usr/bin/env python3
"""
visualize_contacts.py  –  Render the object + multiple tool poses in one figure.

Loads a `.pt` file produced by contact_gen.py, re-loads the original meshes
(paths stored inside the .pt), transforms tools to their optimised poses,
and renders everything in a single 3D plot using matplotlib.

Contact overlays (--show-contacts / --no-contacts):
  When the .pt file contains contact metadata (from the enriched contact_gen.py),
  the visualiser draws:
    • Small scatter dots  – the contact points on the tool surface (in object frame),
      colour-coded by their SDF distance to the object surface.
    • Quiver arrows       – the outward face normal at each contact point, showing
      which direction the object surface is facing at the point of contact.

Usage:
    python visualize_contacts.py --input contact_configs.pt --num-tools 8
    python visualize_contacts.py --input contact_configs.pt --num-tools 8 --save viz.png
    python visualize_contacts.py --input contact_configs.pt --no-contacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

# =============================================================================
#                            COLOUR PALETTE
# =============================================================================
TOOL_COLOURS = [
    (0.90, 0.30, 0.30, 0.15),  # red
    (0.30, 0.70, 0.90, 0.15),  # sky blue
    (0.20, 0.85, 0.45, 0.15),  # green
    (0.95, 0.65, 0.15, 0.15),  # orange
    (0.70, 0.35, 0.90, 0.15),  # purple
    (0.95, 0.85, 0.20, 0.15),  # yellow
    (0.35, 0.90, 0.85, 0.15),  # cyan
    (0.90, 0.45, 0.70, 0.15),  # pink
    (0.55, 0.55, 0.55, 0.15),  # grey
    (0.60, 0.80, 0.30, 0.15),  # lime
]

OBJECT_COLOUR = (0.65, 0.65, 0.70, 1.0)


# =============================================================================
#                           LOAD DATA
# =============================================================================

def load_data(pt_path: str) -> dict:
    """Load the .pt file; tensors → numpy, list-of-tensors → list-of-numpy."""
    import torch
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy()
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "numpy"):
            out[k] = [t.numpy() for t in v]
        else:
            out[k] = v
    return out


def load_mesh_trimesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh", process=False)


# =============================================================================
#                       TRANSFORM HELPERS
# =============================================================================

def transform_mesh(mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices = m.vertices @ R.T + t
    return m


def transform_object_mesh(mesh: trimesh.Trimesh, R_obj: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices = m.vertices @ R_obj.T
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    return m


# =============================================================================
#                   MATPLOTLIB VISUALISATION
# =============================================================================

def _plot_mesh_on_ax(ax, mesh: trimesh.Trimesh, colour, alpha=1.0, label=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = mesh.vertices[mesh.faces]
    pc = Poly3DCollection(
        polys, alpha=alpha,
        facecolor=colour[:3] if len(colour) == 4 else colour,
        edgecolor=(0.2, 0.2, 0.2, 0.15), linewidth=0.1,
    )
    ax.add_collection3d(pc)
    if label:
        ax.scatter([], [], [], color=colour[:3], label=label, s=30)


def _add_ground_plane(ax, extent: float):
    g = extent * 1.2
    xx, yy = np.meshgrid(np.linspace(-g, g, 2), np.linspace(-g, g, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color="grey")


def _set_equal_aspect(ax, all_verts: np.ndarray):
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)
    centres = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.15
    ax.set_xlim(centres[0] - span, centres[0] + span)
    ax.set_ylim(centres[1] - span, centres[1] + span)
    ax.set_zlim(max(0, centres[2] - span), centres[2] + span)


def visualize_matplotlib(
    obj_mesh: trimesh.Trimesh,
    tool_meshes: list[trimesh.Trimesh],
    pen_losses: np.ndarray | None = None,
    contact_losses: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "Contact Configurations",
    # Contact overlays (optional – silently skipped if None)
    contact_pts: list[np.ndarray] | None = None,      # list[C×3]  contact points
    contact_normals: list[np.ndarray] | None = None,  # list[C×3]  face normals
    contact_sdfs: list[np.ndarray] | None = None,     # list[C]    SDF at each point
):
    """Render object + tool poses in a single matplotlib 3D figure.

    Contact overlays:
      contact_pts     – (C, 3) contact points; scatter-plotted and colour-coded
                        by SDF value (if contact_sdfs provided) or tool colour.
      contact_normals – (C, 3) outward face normals; drawn as quiver arrows.
      contact_sdfs    – (C,)   SDF distance values; used for colour-mapping dots.
    """
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    obj_extent = np.linalg.norm(obj_mesh.vertices.max(axis=0) - obj_mesh.vertices.min(axis=0))
    _add_ground_plane(ax, obj_extent)
    _plot_mesh_on_ax(ax, obj_mesh, OBJECT_COLOUR, alpha=0.85, label="Object")

    all_verts = [obj_mesh.vertices]
    arrow_len = obj_extent * 0.08

    for i, tm in enumerate(tool_meshes):
        c = TOOL_COLOURS[i % len(TOOL_COLOURS)]
        lbl = f"Tool #{i}"
        if pen_losses is not None and contact_losses is not None:
            lbl += f"  (pen={pen_losses[i]:.4f}, cont={contact_losses[i]:.4f})"
        _plot_mesh_on_ax(ax, tm, c, alpha=c[3], label=lbl)
        all_verts.append(tm.vertices)

        if contact_pts is not None and i < len(contact_pts):
            pts = contact_pts[i]   # (C, 3)

            # Colour by SDF if available, else use tool colour
            if contact_sdfs is not None and i < len(contact_sdfs):
                sdf = contact_sdfs[i]              # (C,)
                sdf_norm = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-9)
                dot_colors = cm.plasma(sdf_norm)[:, :3]  # plasma: yellow=far, purple=close
            else:
                dot_colors = [c[:3]] * len(pts)

            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=dot_colors, s=30, zorder=5,
                depthshade=False, alpha=0.95,
                label="Contact pts (coloured by SDF)" if i == 0 else None,
            )

            # Face normal arrows
            if contact_normals is not None and i < len(contact_normals):
                n = contact_normals[i]   # (C, 3) unit normals
                ax.quiver(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    n[:, 0] * arrow_len, n[:, 1] * arrow_len, n[:, 2] * arrow_len,
                    color=c[:3], alpha=0.85, linewidth=1.5,
                    arrow_length_ratio=0.3,
                    label="Contact normals" if i == 0 else None,
                )

    all_verts = np.concatenate(all_verts, axis=0)
    _set_equal_aspect(ax, all_verts)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
#                               MAIN
# =============================================================================

def _extract_contact_overlays(
    data: dict,
    indices: np.ndarray,
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    """Pull per-config contact overlays out of the data dict.

    Returns:
        contact_pts     – list of (C, 3) arrays (object frame)
        contact_normals – list of (C, 3) unit normal arrays
        contact_sdfs    – list of (C,)   SDF distance arrays  (None if absent)
        All lists are None if the .pt file predates the contact metadata.
    """
    has_pts     = "contact_pts_obj_frame" in data
    has_normals = "contact_normals"       in data
    has_sdfs    = "contact_sdfs"          in data

    if not has_pts and not has_normals:
        return None, None, None

    pts_arr = data.get("contact_pts_obj_frame")   # (N, C, 3)
    nor_arr = data.get("contact_normals")          # (N, C, 3)
    sdf_arr = data.get("contact_sdfs")             # (N, C)

    contact_pts_list     = []
    contact_normals_list = []
    contact_sdfs_list    = []

    for idx in indices:
        idx = int(idx)
        if has_pts:
            contact_pts_list.append(pts_arr[idx])
        if has_normals:
            contact_normals_list.append(nor_arr[idx])
        if has_sdfs:
            contact_sdfs_list.append(sdf_arr[idx])

    return (
        contact_pts_list     if has_pts     else None,
        contact_normals_list if has_normals else None,
        contact_sdfs_list    if has_sdfs    else None,
    )


def main():
    p = argparse.ArgumentParser(
        description="Visualise object + tool contact configurations from a .pt file.",
    )
    p.add_argument("--input", type=str, required=True, help="Path to contact_configs.pt")
    p.add_argument("--num-tools", type=int, default=4,
                   help="Max number of tool poses to display (default: 4)")
    p.add_argument("--save", type=str, default=None,
                   help="If set, save the figure to this path instead of showing interactively")
    p.add_argument("--object", type=str, default=None,
                   help="Override object mesh path (else uses path from .pt)")
    p.add_argument("--tool", type=str, default=None,
                   help="Override tool mesh path (else uses path from .pt)")
    contact_grp = p.add_mutually_exclusive_group()
    contact_grp.add_argument("--show-contacts", dest="show_contacts", action="store_true",
                             default=True, help="Overlay contact points and normals (default: on)")
    contact_grp.add_argument("--no-contacts", dest="show_contacts", action="store_false",
                             help="Disable contact overlays")
    args = p.parse_args()

    print(f"Loading {args.input} …")
    data = load_data(args.input)

    n_total = data["tool_translations"].shape[0]
    n_show  = min(args.num_tools, n_total)
    print(f"  {n_total} valid configs found, showing {n_show}")

    obj_path  = args.object or data.get("object_mesh_path")
    tool_path = args.tool   or data.get("tool_mesh_path")

    for label, path in [("Object", obj_path), ("Tool", tool_path)]:
        if path is None:
            print(f"ERROR: {label} mesh path missing. Pass --object / --tool.")
            sys.exit(1)
        if not Path(path).exists():
            print(f"ERROR: {label} mesh not found: {path}")
            sys.exit(1)

    print(f"  Object mesh: {obj_path}")
    print(f"  Tool mesh:   {tool_path}")

    obj_mesh_raw  = load_mesh_trimesh(obj_path)
    tool_mesh_raw = load_mesh_trimesh(tool_path)

    tool_scale = data.get("tool_scale", 1.0)
    tool_mesh_raw.vertices *= tool_scale
    print(f"  Tool scale:  {tool_scale:.4f}")

    R_obj    = data["object_rotation"]
    obj_mesh = transform_object_mesh(obj_mesh_raw, R_obj)

    # Diverse subset: evenly spaced along contact_loss ranking
    order   = np.argsort(data["contact_loss"]) if "contact_loss" in data else np.arange(n_total)
    indices = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

    tool_meshes = []
    for idx in indices:
        tm = transform_mesh(tool_mesh_raw, data["tool_rotations"][idx], data["tool_translations"][idx])
        tool_meshes.append(tm)

    pen_losses     = data["pen_loss"][indices]     if "pen_loss"     in data else None
    contact_losses = data["contact_loss"][indices] if "contact_loss" in data else None

    contact_pts = contact_normals = contact_sdfs = None
    if args.show_contacts:
        contact_pts, contact_normals, contact_sdfs = _extract_contact_overlays(data, indices)
        if contact_pts is not None:
            C = contact_pts[0].shape[0]
            sdf_note = " (with SDF colour-coding)" if contact_sdfs is not None else ""
            print(f"  Showing {C} contact pts/config{sdf_note} + normals")
        else:
            print("  ⚠ No contact metadata in .pt file (pre-dates enrichment).")

    title = (
        f"Contact Configurations  ({n_show}/{n_total} shown)\n"
        f"Object: {Path(obj_path).name}   Tool: {Path(tool_path).name}"
    )

    visualize_matplotlib(
        obj_mesh, tool_meshes, pen_losses, contact_losses,
        save_path=args.save, title=title,
        contact_pts=contact_pts,
        contact_normals=contact_normals,
        contact_sdfs=contact_sdfs,
    )


if __name__ == "__main__":
    main()

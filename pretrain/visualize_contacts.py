#!/usr/bin/env python3
"""
visualize_contacts.py  –  Render the object + multiple tool poses in one figure.

Loads a `.pt` file produced by contact_gen.py, re-loads the original meshes
(paths stored inside the .pt), transforms tools to their optimised poses,
and renders everything in a single 3D plot using matplotlib (or Open3D if
available for higher quality).

Usage:
    python visualize_contacts.py --input contact_configs.pt --num-tools 8
    python visualize_contacts.py --input contact_configs.pt --num-tools 8 --save viz.png
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
# A curated set of visually distinct colours for tool instances.
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

OBJECT_COLOUR = (0.65, 0.65, 0.70, 1.0)   # neutral steel
GROUND_COLOUR = (0.92, 0.92, 0.90, 0.30)  # faint ground plane


# =============================================================================
#                           LOAD DATA
# =============================================================================

def load_data(pt_path: str) -> dict:
    """Load the .pt file and convert tensors to numpy."""
    import torch
    data = torch.load(pt_path, map_location="cpu")

    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy()
        else:
            out[k] = v  # strings (paths)
    return out


def load_mesh_trimesh(path: str) -> trimesh.Trimesh:
    """Load mesh via trimesh."""
    mesh = trimesh.load(path, force="mesh", process=False)
    return mesh


# =============================================================================
#                       TRANSFORM HELPERS
# =============================================================================

def transform_mesh(
    mesh: trimesh.Trimesh,
    R: np.ndarray,
    t: np.ndarray,
) -> trimesh.Trimesh:
    """Apply rotation R (3,3) and translation t (3,) to a mesh copy."""
    m = mesh.copy()
    m.vertices = m.vertices @ R.T + t
    return m


def transform_object_mesh(
    mesh: trimesh.Trimesh,
    R_obj: np.ndarray,
) -> trimesh.Trimesh:
    """Rotate the object by R_obj and ground it (z_min = 0)."""
    m = mesh.copy()
    m.vertices = m.vertices @ R_obj.T
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    return m


# =============================================================================
#                   MATPLOTLIB VISUALISATION
# =============================================================================

def _plot_mesh_on_ax(ax, mesh: trimesh.Trimesh, colour, alpha=1.0, label=None):
    """Add a triangulated mesh surface to a 3D matplotlib axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = mesh.vertices
    faces = mesh.faces

    # Build polygon list
    polys = verts[faces]

    pc = Poly3DCollection(
        polys,
        alpha=alpha,
        facecolor=colour[:3] if len(colour) == 4 else colour,
        edgecolor=(0.2, 0.2, 0.2, 0.15),
        linewidth=0.1,
    )
    ax.add_collection3d(pc)

    if label:
        # Invisible scatter for legend
        ax.scatter([], [], [], color=colour[:3], label=label, s=30)


def _add_ground_plane(ax, extent: float):
    """Draw a faint ground plane at z=0."""
    g = extent * 1.2
    xx, yy = np.meshgrid(
        np.linspace(-g, g, 2),
        np.linspace(-g, g, 2),
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.08, color="grey")


def _set_equal_aspect(ax, all_verts: np.ndarray):
    """Equalise axis scales for a 3-D plot."""
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
):
    """Render object + tool poses in a single matplotlib 3D figure."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Ground plane
    obj_extent = np.linalg.norm(
        obj_mesh.vertices.max(axis=0) - obj_mesh.vertices.min(axis=0)
    )
    _add_ground_plane(ax, obj_extent)

    # Object
    _plot_mesh_on_ax(ax, obj_mesh, OBJECT_COLOUR, alpha=0.85, label="Object")

    # Collect all vertices for axis scaling
    all_verts = [obj_mesh.vertices]

    # Tools
    for i, tm in enumerate(tool_meshes):
        c = TOOL_COLOURS[i % len(TOOL_COLOURS)]
        lbl = f"Tool #{i}"
        if pen_losses is not None and contact_losses is not None:
            lbl += f"  (pen={pen_losses[i]:.4f}, cont={contact_losses[i]:.4f})"
        _plot_mesh_on_ax(ax, tm, c, alpha=c[3], label=lbl)
        all_verts.append(tm.vertices)

    all_verts = np.concatenate(all_verts, axis=0)
    _set_equal_aspect(ax, all_verts)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # Nice viewing angle
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
#               OPEN3D VISUALISATION  (optional, higher quality)
# =============================================================================

def _try_open3d_visualize(
    obj_mesh: trimesh.Trimesh,
    tool_meshes: list[trimesh.Trimesh],
    save_path: str | None = None,
) -> bool:
    """Attempt Open3D visualisation.  Returns True if successful."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    geometries = []

    # Object
    o_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(obj_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(obj_mesh.faces),
    )
    o_mesh.compute_vertex_normals()
    o_mesh.paint_uniform_color(OBJECT_COLOUR[:3])
    geometries.append(o_mesh)

    # Tools
    for i, tm in enumerate(tool_meshes):
        c = TOOL_COLOURS[i % len(TOOL_COLOURS)]
        t_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tm.vertices),
            triangles=o3d.utility.Vector3iVector(tm.faces),
        )
        t_mesh.compute_vertex_normals()
        t_mesh.paint_uniform_color(c[:3])
        geometries.append(t_mesh)

    # Ground plane
    ground = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.001)
    ground.translate([-1.0, -1.0, -0.001])
    ground.paint_uniform_color([0.9, 0.9, 0.88])
    geometries.append(ground)

    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"Saved Open3D render to {save_path}")
    else:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Contact Configurations",
            width=1400,
            height=900,
        )

    return True


# =============================================================================
#                               MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Visualise object + tool contact configurations from a .pt file.",
    )
    p.add_argument("--input", type=str, required=True, help="Path to contact_configs.pt")
    p.add_argument("--num-tools", type=int, default=4,
                   help="Max number of tool poses to display (default: 8)")
    p.add_argument("--save", type=str, default=None,
                   help="If set, save the figure to this path instead of showing interactively")
    p.add_argument("--backend", type=str, choices=["matplotlib", "open3d", "auto"],
                   default="auto",
                   help="Render backend (default: auto – tries Open3D first)")
    # Allow overriding mesh paths (in case files were moved)
    p.add_argument("--object", type=str, default=None,
                   help="Override object mesh path (else uses path from .pt)")
    p.add_argument("--tool", type=str, default=None,
                   help="Override tool mesh path (else uses path from .pt)")
    args = p.parse_args()

    # ---- Load saved data ----
    print(f"Loading {args.input} …")
    data = load_data(args.input)

    n_total = data["tool_translations"].shape[0]
    n_show = min(args.num_tools, n_total)
    print(f"  {n_total} valid configs found, showing {n_show}")

    # ---- Resolve mesh paths ----
    obj_path = args.object or data.get("object_mesh_path")
    tool_path = args.tool or data.get("tool_mesh_path")

    if obj_path is None or tool_path is None:
        print("ERROR: Mesh paths not found in .pt file and not provided via --object / --tool.")
        sys.exit(1)

    if not Path(obj_path).exists():
        print(f"ERROR: Object mesh not found: {obj_path}")
        sys.exit(1)
    if not Path(tool_path).exists():
        print(f"ERROR: Tool mesh not found: {tool_path}")
        sys.exit(1)

    print(f"  Object mesh: {obj_path}")
    print(f"  Tool mesh:   {tool_path}")

    # ---- Load meshes ----
    obj_mesh_raw = load_mesh_trimesh(obj_path)
    tool_mesh_raw = load_mesh_trimesh(tool_path)

    # ---- Apply tool scale (contact_gen auto-scales tool to match object) ----
    tool_scale = data.get("tool_scale", 1.0)
    tool_mesh_raw.vertices = tool_mesh_raw.vertices * tool_scale
    print(f"  Tool scale:  {tool_scale:.4f}")

    # ---- Transform object (same rotation + grounding as generation) ----
    R_obj = data["object_rotation"]  # (3, 3)
    obj_mesh = transform_object_mesh(obj_mesh_raw, R_obj)

    # ---- Transform selected tool poses ----
    # Pick a diverse subset: evenly spaced indices sorted by contact_loss
    if "contact_loss" in data:
        order = np.argsort(data["contact_loss"])
    else:
        order = np.arange(n_total)

    # Take evenly spaced samples from the sorted list to show diversity
    indices = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

    tool_meshes = []
    for idx in indices:
        R_tool = data["tool_rotations"][idx]      # (3, 3)
        t_tool = data["tool_translations"][idx]    # (3,)
        tm = transform_mesh(tool_mesh_raw, R_tool, t_tool)
        tool_meshes.append(tm)

    pen_losses = data["pen_loss"][indices] if "pen_loss" in data else None
    contact_losses = data["contact_loss"][indices] if "contact_loss" in data else None

    # ---- Render ----
    title = (
        f"Contact Configurations  ({n_show}/{n_total} shown)\n"
        f"Object: {Path(obj_path).name}   Tool: {Path(tool_path).name}"
    )

    backend = args.backend
    if backend == "auto":
        if not _try_open3d_visualize(obj_mesh, tool_meshes, args.save):
            print("Open3D not available, falling back to matplotlib")
            visualize_matplotlib(
                obj_mesh, tool_meshes, pen_losses, contact_losses,
                save_path=args.save, title=title,
            )
    elif backend == "open3d":
        if not _try_open3d_visualize(obj_mesh, tool_meshes, args.save):
            print("ERROR: Open3D not installed.  pip install open3d")
            sys.exit(1)
    else:
        visualize_matplotlib(
            obj_mesh, tool_meshes, pen_losses, contact_losses,
            save_path=args.save, title=title,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
export_contact_heatmap.py  –  Colour the object mesh by contact density.

For every vertex on the object, count how many contact points (across ALL
stored configurations in a .pt file) fall within a given radius.  Vertices
with zero nearby contacts are light (near-white), vertices with many contacts
are dark (deep red / plasma-style).

The coloured mesh is written as an .obj where vertex lines carry RGB:

    v  x  y  z  r  g  b

This "xyzrgb" extension is read natively by MeshLab and most modern viewers.

Usage:
    python export_contact_heatmap.py --input contact_configs.pt
    python export_contact_heatmap.py --input contact_configs.pt \\
        --radius 0.03 --colormap plasma -o heatmap.obj

    # aggregate across multiple .pt files (e.g. same object, different tools)
    python export_contact_heatmap.py \\
        --input results/cfg_001.pt results/cfg_002.pt results/cfg_003.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


# =============================================================================
#                           COLORMAP  (no matplotlib needed)
# =============================================================================

def _plasma_lut() -> np.ndarray:    """50-entry approximation of matplotlib's 'plasma' colormap (dark→light).

    Returns shape (50, 3) float32 in [0, 1].  Index 0 = darkest (deep purple),
    index 49 = lightest (bright yellow).
    """
    # Sampled key stops from plasma: purple→magenta→orange→yellow
    stops = np.array([
        [0.050, 0.030, 0.528],
        [0.259, 0.012, 0.612],
        [0.440, 0.005, 0.659],
        [0.601, 0.088, 0.632],
        [0.735, 0.215, 0.541],
        [0.841, 0.341, 0.424],
        [0.922, 0.468, 0.312],
        [0.970, 0.609, 0.174],
        [0.988, 0.762, 0.021],
        [0.940, 0.975, 0.131],
    ], dtype=np.float32)
    # Interpolate to 50 entries
    n = 50
    indices = np.linspace(0, len(stops) - 1, n)
    lut = np.zeros((n, 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, len(stops) - 1)
        t = idx - lo
        lut[i] = (1 - t) * stops[lo] + t * stops[hi]
    return lut


def _viridis_light_to_dark_lut() -> np.ndarray:
    """White → viridis dark blue.  Low count = white, high count = dark."""
    stops = np.array([
        [0.98, 0.98, 0.98],   # near-white
        [0.90, 0.95, 0.85],
        [0.60, 0.90, 0.75],
        [0.20, 0.73, 0.63],
        [0.13, 0.57, 0.55],
        [0.12, 0.43, 0.52],
        [0.13, 0.29, 0.49],
        [0.15, 0.17, 0.39],
        [0.09, 0.05, 0.29],
        [0.00, 0.00, 0.13],   # near-black
    ], dtype=np.float32)
    n = 50
    indices = np.linspace(0, len(stops) - 1, n)
    lut = np.zeros((n, 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, len(stops) - 1)
        t = idx - lo
        lut[i] = (1 - t) * stops[lo] + t * stops[hi]
    return lut


def _hot_lut() -> np.ndarray:
    """White → yellow → orange → dark red.  Low = white, high = dark red."""
    stops = np.array([
        [0.97, 0.97, 0.97],   # white
        [0.99, 0.93, 0.80],
        [0.99, 0.85, 0.55],
        [0.99, 0.70, 0.30],
        [0.96, 0.52, 0.15],
        [0.88, 0.32, 0.08],
        [0.73, 0.15, 0.05],
        [0.55, 0.05, 0.02],
        [0.38, 0.01, 0.01],
        [0.20, 0.00, 0.00],   # dark red
    ], dtype=np.float32)
    n = 50
    indices = np.linspace(0, len(stops) - 1, n)
    lut = np.zeros((n, 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, len(stops) - 1)
        t = idx - lo
        lut[i] = (1 - t) * stops[lo] + t * stops[hi]
    return lut


COLORMAPS = {
    "hot":     _hot_lut,      # white → dark red  (default – matches user request)
    "plasma":  _plasma_lut,   # dark purple → bright yellow
    "viridis": _viridis_light_to_dark_lut,
}


def heat_to_rgb(values: np.ndarray, colormap: str = "hot") -> np.ndarray:
    """Map scalar heat values in [0, 1] → RGB colours (N, 3).

    0 → lightest colour,  1 → darkest colour.
    """
    lut = COLORMAPS[colormap]()          # (50, 3)
    # Clamp and map to LUT indices
    v = np.clip(values, 0.0, 1.0)
    idx = (v * (len(lut) - 1)).astype(int)
    return lut[idx]


# =============================================================================
#                              HELPERS
# =============================================================================

def load_pt(pt_path: str) -> dict:
    """Load a .pt file and convert tensors to numpy."""
    import torch
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in data.items():
        out[k] = v.numpy() if hasattr(v, "numpy") else v
    return out


def gather_contact_points(data_list: list[dict]) -> np.ndarray | None:
    """Collect all contact_pts_obj_frame from every .pt dataset.

    Returns:
        pts: (M, 3)  all contact points in object frame, or None if absent.
    """
    all_pts = []
    for data in data_list:
        if "contact_pts_obj_frame" not in data:
            continue
        pts = data["contact_pts_obj_frame"]   # (N, C, 3)
        pts = pts.reshape(-1, 3)              # (N*C, 3)
        all_pts.append(pts)
    if not all_pts:
        return None
    return np.concatenate(all_pts, axis=0)


def compute_vertex_heat(
    verts: np.ndarray,          # (V, 3)
    contact_pts: np.ndarray,    # (M, 3)
    radius: float,
) -> np.ndarray:
    """For each vertex, count contact points within `radius`.

    Uses a chunked distance computation so it doesn't OOM on large meshes.

    Returns:
        heat: (V,)  float  – raw counts, not yet normalised.
    """
    V = verts.shape[0]
    M = contact_pts.shape[0]
    heat = np.zeros(V, dtype=np.float32)
    r2 = radius ** 2

    # Process contacts in chunks to limit memory
    chunk = 4096
    for start in range(0, M, chunk):
        c = contact_pts[start : start + chunk]          # (K, 3)
        diff = verts[:, None, :] - c[None, :, :]        # (V, K, 3)
        sq   = (diff ** 2).sum(axis=-1)                 # (V, K)
        heat += (sq < r2).sum(axis=-1).astype(np.float32)

    return heat


def transform_object_verts(verts: np.ndarray, R_obj: np.ndarray) -> np.ndarray:
    """Rotate and ground the object (z_min = 0)."""
    v = verts @ R_obj.T
    v[:, 2] -= v[:, 2].min()
    return v


def write_obj_vertex_colour(
    obj_path: str,
    verts: np.ndarray,      # (V, 3)
    faces: np.ndarray,      # (F, 3) int
    colours: np.ndarray,    # (V, 3) float [0,1]
) -> None:
    """Write an OBJ with per-vertex colours via the 'v x y z r g b' extension.

    Supported by: MeshLab, Blender (import), CloudCompare.
    """
    with open(obj_path, "w") as f:
        f.write("# Contact heatmap — per-vertex colour (v x y z r g b)\n")
        f.write(f"# Vertices: {len(verts)}  Faces: {len(faces)}\n\n")
        for (x, y, z), (r, g, b) in zip(verts, colours):
            f.write(f"v {x:.8f} {y:.8f} {z:.8f} {r:.5f} {g:.5f} {b:.5f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# =============================================================================
#                                 MAIN
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export object mesh coloured by contact point density."
    )
    p.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="One or more .pt files produced by contact_gen.py",
    )
    p.add_argument(
        "--radius", "-r", type=float, default=0.03,
        help="Radius (metres) within which a contact point 'touches' a vertex (default: 0.03)",
    )
    p.add_argument(
        "--colormap", "-c", choices=list(COLORMAPS.keys()), default="hot",
        help="Colour scheme: hot (white→dark-red), plasma, viridis (default: hot)",
    )
    p.add_argument(
        "--object", type=str, default=None,
        help="Override object mesh path (default: read from .pt file)",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .obj path (default: <first_input_stem>_heatmap.obj)",
    )
    p.add_argument(
        "--gamma", type=float, default=0.5,
        help="Gamma applied to normalised heat before colour mapping (< 1 = stretch low end, "
             "default: 0.5)",
    )
    args = p.parse_args()

    # ---- Load all .pt files ----
    data_list = []
    for pt in args.input:
        print(f"Loading {pt} …")
        d = load_pt(pt)
        data_list.append(d)
        n = d["tool_translations"].shape[0] if "tool_translations" in d else "?"
        print(f"  {n} configs, object: {d.get('object_mesh_path', 'unknown')}")

    # ---- Resolve object mesh path ----
    obj_mesh_path = args.object
    if obj_mesh_path is None:
        obj_mesh_path = data_list[0].get("object_mesh_path")
    if obj_mesh_path is None or not Path(obj_mesh_path).exists():
        print(f"ERROR: Object mesh not found: {obj_mesh_path}")
        sys.exit(1)

    # ---- All datasets must share the same object rotation (first one wins) ----
    R_obj = data_list[0]["object_rotation"]  # (3, 3)

    # ---- Gather all contact points  ----
    contact_pts = gather_contact_points(data_list)
    if contact_pts is None or contact_pts.shape[0] == 0:
        print("ERROR: No contact_pts_obj_frame found in any input file.")
        sys.exit(1)
    print(f"\nTotal contact points: {contact_pts.shape[0]}")

    # ---- Load and transform object mesh ----
    print(f"Loading object mesh: {obj_mesh_path}")
    obj_mesh = trimesh.load(obj_mesh_path, force="mesh", process=False)
    verts = transform_object_verts(np.array(obj_mesh.vertices, dtype=np.float32), R_obj)
    faces = np.array(obj_mesh.faces, dtype=np.int32)

    # ---- Compute per-vertex heat ----
    print(f"Computing contact density (radius={args.radius:.4f}) …")
    heat = compute_vertex_heat(verts, contact_pts, radius=args.radius)
    print(f"  Max heat: {heat.max():.0f}  Mean (non-zero): "
          f"{heat[heat > 0].mean():.2f} " if heat.any() else "  All zeros — try a larger --radius")

    # ---- Normalise and apply gamma ----
    h_max = heat.max()
    if h_max > 0:
        heat_norm = (heat / h_max) ** args.gamma
    else:
        heat_norm = heat

    # ---- Map to colours ----
    colours = heat_to_rgb(heat_norm, colormap=args.colormap)

    # ---- Output path ----
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input[0]).stem
        out_path = str(Path(args.input[0]).parent / f"{stem}_heatmap.obj")

    # ---- Write OBJ ----
    write_obj_vertex_colour(out_path, verts, faces, colours)
    print(f"\n✓ Heatmap written to {out_path}")
    print(f"  Colormap: {args.colormap}  |  Radius: {args.radius} m  |  Gamma: {args.gamma}")
    print(f"  Open in MeshLab: File → Import Mesh → {out_path}")


if __name__ == "__main__":
    main()

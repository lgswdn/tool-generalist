#!/usr/bin/env python3
"""
export_contacts_obj.py  –  Export contact configurations as a merged .obj + .mtl.

Loads a .pt file produced by contact_gen.py, re-loads the original meshes,
transforms tools to their optimised poses, and writes a single .obj (with a
companion .mtl for per-tool colours) that can be opened in MeshLab, Blender,
VS Code preview, or any standard 3D viewer.

Usage:
    python export_contacts_obj.py --input contact_configs.pt --num-tools 8 -o scene.obj
    python export_contacts_obj.py --input contact_configs.pt                 # writes contact_configs_scene.obj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

# =============================================================================
#                         COLOUR PALETTE
# =============================================================================
TOOL_COLOURS_RGB = [
    (0.90, 0.30, 0.30),  # red
    (0.30, 0.70, 0.90),  # sky blue
    (0.20, 0.85, 0.45),  # green
    (0.95, 0.65, 0.15),  # orange
    (0.70, 0.35, 0.90),  # purple
    (0.95, 0.85, 0.20),  # yellow
    (0.35, 0.90, 0.85),  # cyan
    (0.90, 0.45, 0.70),  # pink
    (0.55, 0.55, 0.55),  # grey
    (0.60, 0.80, 0.30),  # lime
]
OBJECT_COLOUR_RGB = (0.65, 0.65, 0.70)
GROUND_COLOUR_RGB = (0.90, 0.90, 0.88)


# =============================================================================
#                          HELPERS
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
            out[k] = v
    return out


def transform_mesh(mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    """Apply rotation R (3,3) and translation t (3,) to a mesh copy."""
    m = mesh.copy()
    m.vertices = m.vertices @ R.T + t
    return m


def transform_object_mesh(mesh: trimesh.Trimesh, R_obj: np.ndarray) -> trimesh.Trimesh:
    """Rotate the object by R_obj and ground it (z_min = 0)."""
    m = mesh.copy()
    m.vertices = m.vertices @ R_obj.T
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    return m


def write_obj_mtl(
    obj_path: str,
    meshes: list[trimesh.Trimesh],
    names: list[str],
    colours: list[tuple[float, float, float]],
):
    """Write a merged .obj + .mtl by hand to guarantee correctness.

    Args:
        obj_path: output .obj file path
        meshes:   list of trimesh.Trimesh
        names:    per-mesh group/material names
        colours:  per-mesh (r, g, b) in [0, 1]
    """
    obj_p = Path(obj_path)
    mtl_p = obj_p.with_suffix(".mtl")
    mtl_name = mtl_p.name

    # ---- Write MTL ----
    with open(mtl_p, "w") as f:
        for name, (r, g, b) in zip(names, colours):
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write(f"Ka 0.1000 0.1000 0.1000\n")
            f.write(f"Ks 0.3000 0.3000 0.3000\n")
            f.write(f"Ns 50.0\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 2\n\n")

    # ---- Write OBJ ----
    vertex_offset = 0
    with open(obj_p, "w") as f:
        f.write(f"# Contact configuration scene\n")
        f.write(f"mtllib {mtl_name}\n\n")

        for mesh, name, colour in zip(meshes, names, colours):
            verts = mesh.vertices
            faces = mesh.faces

            f.write(f"o {name}\n")
            f.write(f"usemtl {name}\n")

            # Vertices
            for v in verts:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

            # Faces (1-indexed, offset by prior vertices)
            for face in faces:
                i0 = face[0] + vertex_offset + 1
                i1 = face[1] + vertex_offset + 1
                i2 = face[2] + vertex_offset + 1
                f.write(f"f {i0} {i1} {i2}\n")

            vertex_offset += len(verts)
            f.write("\n")


# =============================================================================
#                              MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Export contact configurations as a merged .obj + .mtl file.",
    )
    p.add_argument("--input", type=str, required=True, help="Path to contact_configs.pt")
    p.add_argument("--num-tools", type=int, default=8,
                   help="Max number of tool poses to include (default: 8)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output .obj path.  Default: <input_stem>_scene.obj")
    # Optional mesh path overrides
    p.add_argument("--object", type=str, default=None,
                   help="Override object mesh path")
    p.add_argument("--tool", type=str, default=None,
                   help="Override tool mesh path")
    args = p.parse_args()

    # ---- Load data ----
    print(f"Loading {args.input} …")
    data = load_data(args.input)

    n_total = data["tool_translations"].shape[0]
    n_show = min(args.num_tools, n_total)
    print(f"  {n_total} valid configs, exporting {n_show}")

    # ---- Resolve mesh paths ----
    obj_path = args.object or data.get("object_mesh_path")
    tool_path = args.tool or data.get("tool_mesh_path")
    if obj_path is None or tool_path is None:
        print("ERROR: Mesh paths not found in .pt and not provided via --object/--tool.")
        sys.exit(1)
    if not Path(obj_path).exists():
        print(f"ERROR: Object mesh not found: {obj_path}")
        sys.exit(1)
    if not Path(tool_path).exists():
        print(f"ERROR: Tool mesh not found: {tool_path}")
        sys.exit(1)

    print(f"  Object: {obj_path}")
    print(f"  Tool:   {tool_path}")

    # ---- Load meshes ----
    obj_mesh_raw = trimesh.load(obj_path, force="mesh", process=False)
    tool_mesh_raw = trimesh.load(tool_path, force="mesh", process=False)

    # Apply scales used during data generation.
    tool_scale = data.get("tool_scale", 1.0)
    object_scale = data.get("object_scale", 1.0)
    tool_mesh_raw.vertices = tool_mesh_raw.vertices * tool_scale
    obj_mesh_raw.vertices = obj_mesh_raw.vertices * object_scale
    print(f"  Tool scale:   {tool_scale:.4f}")
    print(f"  Object scale: {object_scale:.4f}")

    # ---- Transform object ----
    R_obj = data.get("object_rotation", np.eye(3))
    obj_mesh = transform_object_mesh(obj_mesh_raw, R_obj)

    # ---- Select diverse tool indices ----
    if "contact_loss" in data:
        order = np.argsort(data["contact_loss"])
    else:
        order = np.arange(n_total)
    indices = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

    # ---- Collect all meshes, names, and colours ----
    meshes = [obj_mesh]
    names = ["Object"]
    colours = [OBJECT_COLOUR_RGB]

    for i, idx in enumerate(indices):
        R_tool = data["tool_rotations"][idx]
        t_tool = data["tool_translations"][idx]
        tm = transform_mesh(tool_mesh_raw, R_tool, t_tool)

        c = TOOL_COLOURS_RGB[i % len(TOOL_COLOURS_RGB)]
        label = f"Tool_{i:03d}"
        if "pen_loss" in data and "contact_loss" in data:
            pl = data["pen_loss"][idx]
            cl = data["contact_loss"][idx]
            label += f"_p{pl:.4f}_c{cl:.4f}"

        meshes.append(tm)
        names.append(label)
        colours.append(c)

    # Ground plane
    extent = np.linalg.norm(obj_mesh.vertices.max(axis=0) - obj_mesh.vertices.min(axis=0))
    ground = trimesh.creation.box(extents=[extent * 2.5, extent * 2.5, 0.001])
    ground.apply_translation([0, 0, -0.0005])
    meshes.append(ground)
    names.append("Ground")
    colours.append(GROUND_COLOUR_RGB)

    # ---- Output path ----
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input).stem
        out_path = str(Path(args.input).parent / f"{stem}_scene.obj")

    # ---- Write OBJ + MTL ----
    write_obj_mtl(out_path, meshes, names, colours)

    mtl_path = Path(out_path).with_suffix(".mtl")
    print(f"\n✓ Exported to {out_path}")
    print(f"  MTL file: {mtl_path}")
    print(f"  Meshes:   1 object + {n_show} tools + 1 ground = {n_show + 2} total")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
view_contacts_isaac.py  –  Isaac Sim gallery viewer for contact configurations.

Loads one or more .pt files produced by contact_gen.py and spawns every
tool–object pair in a tiled Isaac Sim stage for visual inspection.

Each config gets its own "cell" in a grid, with the grounded object and one
or more tool poses placed at their optimised positions.  No robot, no RL
loop — just a static scene rendered with Isaac Sim's real-time renderer.

Usage:
    # Interactive viewport — view 4 .pt files with 4 tool poses each
    python view_contacts_isaac.py \\
        --inputs dir/config_001.pt dir/config_002.pt dir/config_003.pt dir/config_004.pt \\
        --num-tools-per-cell 4

    # Use a glob
    python view_contacts_isaac.py \\
        --input-dir results/ \\
        --num-tools-per-cell 4

    # Headless screenshot
    python view_contacts_isaac.py \\
        --inputs results/*.pt \\
        --num-tools-per-cell 4 \\
        --save gallery.png

NOTE: This script must be run inside an Isaac Lab / Isaac Sim Python environment,
      e.g.  `isaaclab -p view_contacts_isaac.py --inputs ...`
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

# ============================================================================
# 1) Parse arguments BEFORE launching Isaac Sim (AppLauncher pattern)
# ============================================================================

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Isaac Sim gallery viewer for contact configurations.",
)
# --- Input selection ---
parser.add_argument("--inputs", type=str, nargs="+", default=[],
                    help="One or more .pt files to visualise.")
parser.add_argument("--input-dir", type=str, default="",
                    help="Directory to glob for *.pt files (alternative to --inputs).")
# --- Display options ---
parser.add_argument("--num-tools-per-cell", type=int, default=4,
                    help="Max tool poses to show per cell (default: 4)")
parser.add_argument("--spacing", type=float, default=3.0,
                    help="Grid cell spacing in metres (default: 3.0)")
parser.add_argument("--cols", type=int, default=0,
                    help="Number of grid columns (0 = auto-sqrt)")
# --- Save options ---
parser.add_argument("--save", type=str, default="",
                    help="If set, capture a screenshot to this path and exit.")
parser.add_argument("--settle-steps", type=int, default=20,
                    help="Simulation steps to run before screenshot (default: 20)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If saving, force headless + cameras
if args_cli.save:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2) Imports that require Isaac Sim to be running
# ============================================================================

import numpy as np
import torch
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics

import omni.usd

# ============================================================================
#                        COLOUR PALETTE
# ============================================================================

TOOL_COLOURS = [
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
OBJECT_COLOUR = (0.65, 0.65, 0.70)

# ============================================================================
#                       DATA LOADING
# ============================================================================


def load_pt(path: str) -> dict:
    """Load a .pt file, converting tensors to numpy."""
    data = torch.load(path, map_location="cpu")
    out = {}
    for k, v in data.items():
        out[k] = v.numpy() if isinstance(v, torch.Tensor) else v
    return out


def collect_inputs(inputs: list[str], input_dir: str) -> list[str]:
    """Resolve the final list of .pt paths."""
    pts = list(inputs)  # explicit --inputs

    # Expand globs from input_dir
    if input_dir:
        pts += sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    # Expand shell globs that argparse didn't resolve
    expanded = []
    for p in pts:
        g = glob.glob(p)
        expanded.extend(sorted(g) if g else [p])

    # Deduplicate + verify existence
    seen = set()
    result = []
    for p in expanded:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            result.append(ap)
    return result


# ============================================================================
#                     USD MESH SPAWNING
# ============================================================================


def _create_preview_material(stage: Usd.Stage, prim_path: str, colour: tuple) -> UsdShade.Material:
    """Create a simple UsdPreviewSurface material with the given diffuse colour."""
    mat = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, prim_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _spawn_trimesh_as_usd(
    stage: Usd.Stage,
    prim_path: str,
    mesh: trimesh.Trimesh,
    colour: tuple,
    mat_path: str | None = None,
) -> None:
    """Spawn a trimesh.Trimesh as a UsdGeom.Mesh prim with a preview material."""
    verts = mesh.vertices.astype(np.float64)
    faces = mesh.faces.astype(int)

    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    usd_mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in verts])

    # Flatten face vertex indices and create face vertex counts (all triangles)
    fvc = [3] * len(faces)
    fvi = faces.flatten().tolist()
    usd_mesh.CreateFaceVertexCountsAttr().Set(fvc)
    usd_mesh.CreateFaceVertexIndicesAttr().Set(fvi)
    usd_mesh.CreateSubdivisionSchemeAttr().Set("none")

    # Compute normals
    usd_mesh.SetNormalsInterpolation("faceVarying")

    # Material
    if mat_path is None:
        mat_path = prim_path + "_Mat"
    mat = _create_preview_material(stage, mat_path, colour)
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(mat)


def _spawn_ground_plane(stage: Usd.Stage, extent: float):
    """Add a large ground plane at z=0."""
    ground_path = "/World/GroundPlane"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    e = extent
    ground.CreatePointsAttr().Set([
        Gf.Vec3f(-e, -e, 0), Gf.Vec3f(e, -e, 0),
        Gf.Vec3f(e, e, 0), Gf.Vec3f(-e, e, 0),
    ])
    ground.CreateFaceVertexCountsAttr().Set([4])
    ground.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    mat = _create_preview_material(stage, "/World/GroundPlane_Mat", (0.85, 0.85, 0.82))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat)


def _spawn_dome_light(stage: Usd.Stage):
    """Add a dome light for decent ambient illumination."""
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr().Set(1500.0)
    light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 0.95))


# ============================================================================
#                              MAIN
# ============================================================================

def main():
    pt_files = collect_inputs(args_cli.inputs, args_cli.input_dir)
    if not pt_files:
        print("ERROR: No .pt files provided.  Use --inputs or --input-dir.")
        sys.exit(1)

    n_cells = len(pt_files)
    n_cols = args_cli.cols if args_cli.cols > 0 else max(1, int(math.ceil(math.sqrt(n_cells))))
    n_rows = int(math.ceil(n_cells / n_cols))
    spacing = args_cli.spacing

    print(f"[INFO] {n_cells} configs → {n_rows}×{n_cols} grid (spacing={spacing:.1f}m)")

    # ---- Get USD stage ----
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # ---- Global scene elements ----
    grid_extent = max(n_rows, n_cols) * spacing
    _spawn_ground_plane(stage, grid_extent)
    _spawn_dome_light(stage)

    # ---- Mesh cache (avoid re-loading the same OBJ file) ----
    mesh_cache: dict[str, trimesh.Trimesh] = {}

    def _load_mesh(path: str) -> trimesh.Trimesh:
        if path not in mesh_cache:
            mesh_cache[path] = trimesh.load(path, force="mesh", process=False)
        return mesh_cache[path].copy()

    # ---- Spawn each cell ----
    for cell_idx, pt_path in enumerate(pt_files):
        row = cell_idx // n_cols
        col = cell_idx % n_cols
        cx = col * spacing
        cy = row * spacing

        print(f"\n[Cell {cell_idx}] row={row} col={col}  {os.path.basename(pt_path)}")

        data = load_pt(pt_path)
        n_total = data["tool_translations"].shape[0]
        n_show = min(args_cli.num_tools_per_cell, n_total)

        obj_path = data.get("object_mesh_path")
        tool_path = data.get("tool_mesh_path")
        if obj_path is None or tool_path is None:
            print(f"  ⚠ Skipping: mesh paths not found in {pt_path}")
            continue
        if not os.path.isfile(obj_path):
            print(f"  ⚠ Object mesh missing: {obj_path}")
            continue
        if not os.path.isfile(tool_path):
            print(f"  ⚠ Tool mesh missing: {tool_path}")
            continue

        # Load and transform object
        obj_mesh = _load_mesh(obj_path)
        R_obj = data["object_rotation"]
        obj_mesh.vertices = obj_mesh.vertices @ R_obj.T
        obj_mesh.vertices[:, 2] -= obj_mesh.vertices[:, 2].min()
        # Shift to grid cell
        obj_mesh.vertices[:, 0] += cx
        obj_mesh.vertices[:, 1] += cy

        cell_prim_base = f"/World/Cell_{cell_idx:04d}"
        _spawn_trimesh_as_usd(
            stage,
            f"{cell_prim_base}/Object",
            obj_mesh,
            OBJECT_COLOUR,
            mat_path=f"{cell_prim_base}/Object_Mat",
        )

        # Select diverse tool indices
        if "contact_loss" in data:
            order = np.argsort(data["contact_loss"])
        else:
            order = np.arange(n_total)
        indices = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

        tool_scale = data.get("tool_scale", 1.0)

        for ti, idx in enumerate(indices):
            tool_mesh = _load_mesh(tool_path)
            tool_mesh.vertices *= tool_scale

            R_tool = data["tool_rotations"][idx]
            t_tool = data["tool_translations"][idx]
            tool_mesh.vertices = tool_mesh.vertices @ R_tool.T + t_tool
            # Shift to grid cell
            tool_mesh.vertices[:, 0] += cx
            tool_mesh.vertices[:, 1] += cy

            c = TOOL_COLOURS[ti % len(TOOL_COLOURS)]
            _spawn_trimesh_as_usd(
                stage,
                f"{cell_prim_base}/Tool_{ti:03d}",
                tool_mesh,
                c,
                mat_path=f"{cell_prim_base}/Tool_{ti:03d}_Mat",
            )

        # Add a small text label (Xform with meaningful name, visible in viewport)
        label_xform = UsdGeom.Xform.Define(stage, f"{cell_prim_base}/Label")
        label_xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0))

        print(f"  Spawned: 1 object + {n_show} tools")

    print(f"\n[INFO] Scene built: {n_cells} cells in {n_rows}×{n_cols} grid")

    # ---- Screenshot or interactive ----
    if args_cli.save:
        # Let the renderer settle
        for _ in range(args_cli.settle_steps):
            simulation_app.update()

        # Use Omniverse's built-in capture
        try:
            import omni.kit.viewport.utility as vp_util
            viewport = vp_util.get_active_viewport()
            if viewport is not None:
                from omni.kit.viewport.utility import capture_viewport_to_file
                capture_viewport_to_file(viewport, args_cli.save)
                print(f"\n✓ Screenshot saved to {args_cli.save}")
            else:
                print("⚠ No active viewport for screenshot (try without --headless)")
        except Exception as e:
            print(f"⚠ Screenshot capture failed: {e}")
            print("  The scene is still built — re-run without --save for interactive.")
    else:
        print("\n[INFO] Interactive mode — close the viewport window to exit.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

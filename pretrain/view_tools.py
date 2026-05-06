#!/usr/bin/env python3
"""IsaacLab viewer that lays generated tool USD assets on a ground plane."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tools-root",
        type=str,
        default="eef/objects_usd",
        help="Directory containing tool USD folders/files.",
    )
    parser.add_argument("--columns", type=int, default=0, help="Grid columns. 0 = auto.")
    parser.add_argument("--spacing", type=float, default=1.6, help="Grid spacing in meters.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale for every tool.")
    parser.add_argument("--max-tools", type=int, default=0, help="Only show the first N tools. 0 = all.")
    parser.add_argument("--ground-margin", type=float, default=2.0)
    parser.add_argument(
        "--tools-metadata",
        type=str,
        default="eef/tools_adjusted.json",
        help="Metadata JSON containing tool head_area annotations.",
    )
    parser.add_argument("--no-tool-bboxes", action="store_true", help="Do not render tool-end bounding boxes.")
    parser.add_argument("--bbox-line-width", type=float, default=0.012)
    parser.add_argument("--save-stage", type=str, default="", help="Optional output .usd path.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to keep running. 0 = until closed.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def safe_prim_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not cleaned:
        cleaned = "tool"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def discover_tool_usds(tools_root: Path) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []

    for usd in sorted(tools_root.glob("*.usd")):
        results.append((usd.stem, usd.resolve()))

    for child in sorted(path for path in tools_root.iterdir() if path.is_dir()):
        same_name = child / f"{child.name}.usd"
        if same_name.is_file():
            results.append((child.name, same_name.resolve()))
            continue
        usds = sorted(child.glob("*.usd"))
        if len(usds) == 1:
            results.append((child.name, usds[0].resolve()))

    return results


def load_tool_head_areas(metadata_path: Path) -> dict[str, list[list[float]]]:
    if not metadata_path.is_file():
        print(f"[WARN] Tool metadata not found: {metadata_path}")
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        entry["name"]: entry["head_area"]
        for entry in data
        if isinstance(entry, dict) and entry.get("name") and entry.get("head_area")
    }


def create_preview_material(stage: Usd.Stage, prim_path: str, color: tuple[float, float, float]):
    mat = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, prim_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def create_line_material(stage: Usd.Stage, prim_path: str, color: tuple[float, float, float]):
    mat = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, prim_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def spawn_ground(stage: Usd.Stage, extent: float) -> None:
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    e = float(extent)
    ground.CreatePointsAttr().Set(
        [
            Gf.Vec3f(-e, -e, 0.0),
            Gf.Vec3f(e, -e, 0.0),
            Gf.Vec3f(e, e, 0.0),
            Gf.Vec3f(-e, e, 0.0),
        ]
    )
    ground.CreateFaceVertexCountsAttr().Set([4])
    ground.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    mat = create_preview_material(stage, "/World/Materials/Ground", (0.78, 0.78, 0.74))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat)


def spawn_lights(stage: Usd.Stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(800.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.95, 0.96, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    sun.CreateIntensityAttr().Set(2200.0)
    sun.CreateAngleAttr().Set(0.45)
    xform = UsdGeom.Xformable(sun.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))


def set_xform(prim, translation: tuple[float, float, float], scale: float):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*translation))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale, scale, scale))
    return translate_op


def aligned_world_bbox(stage: Usd.Stage, prim) -> Gf.Range3d:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()


def local_bbox(prim) -> Gf.Range3d:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return bbox_cache.ComputeLocalBound(prim).ComputeAlignedBox()


def resolve_tool_bbox_frame(tool_prim):
    """Return the prim whose local coordinates match tool geometry metadata."""
    preferred = tool_prim.GetStage().GetPrimAtPath(tool_prim.GetPath().AppendChild("link_coacd_convex_piece_0"))
    if preferred.IsValid():
        return preferred

    for prim in Usd.PrimRange(tool_prim):
        if prim == tool_prim:
            continue
        if prim.GetName() == "link_coacd_convex_piece_0":
            return prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim

    return tool_prim


def render_tool_bbox(
    stage: Usd.Stage,
    tool_prim,
    tool_name: str,
    head_areas: dict[str, list[list[float]]],
    material,
) -> bool:
    head_area = head_areas.get(tool_name)
    if head_area is None:
        return False

    bbox_frame = resolve_tool_bbox_frame(tool_prim)
    bbox = local_bbox(bbox_frame)
    bbox_min = bbox.GetMin()
    bbox_max = bbox.GetMax()
    size = Gf.Vec3d(
        bbox_max[0] - bbox_min[0],
        bbox_max[1] - bbox_min[1],
        bbox_max[2] - bbox_min[2],
    )
    if size[0] == 0.0 or size[1] == 0.0 or size[2] == 0.0:
        return False

    hmin = head_area[0]
    hmax = head_area[1]
    min_pt = Gf.Vec3d(
        bbox_min[0] + hmin[0] * size[0],
        bbox_min[1] + hmin[1] * size[1],
        bbox_min[2] + hmin[2] * size[2],
    )
    max_pt = Gf.Vec3d(
        bbox_min[0] + hmax[0] * size[0],
        bbox_min[1] + hmax[1] * size[1],
        bbox_min[2] + hmax[2] * size[2],
    )

    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt
    corners = [
        Gf.Vec3f(x0, y0, z0),
        Gf.Vec3f(x1, y0, z0),
        Gf.Vec3f(x1, y1, z0),
        Gf.Vec3f(x0, y1, z0),
        Gf.Vec3f(x0, y0, z1),
        Gf.Vec3f(x1, y0, z1),
        Gf.Vec3f(x1, y1, z1),
        Gf.Vec3f(x0, y1, z1),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    points = []
    for a, b in edges:
        points.extend([corners[a], corners[b]])

    curves = UsdGeom.BasisCurves.Define(stage, str(bbox_frame.GetPath().AppendChild("FunctionalEndBBox")))
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr().Set([2] * len(edges))
    curves.CreatePointsAttr().Set(points)
    curves.CreateWidthsAttr().Set([args_cli.bbox_line_width] * len(points))
    curves.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)
    return True


def add_camera(stage: Usd.Stage, center_x: float, center_y: float, grid_width: float, grid_height: float) -> None:
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    distance = max(grid_width, grid_height, 6.0)
    eye = Gf.Vec3d(center_x - distance * 0.55, center_y - distance * 0.85, distance * 0.65)
    target = Gf.Vec3d(center_x, center_y, 0.0)
    direction = (target - eye).GetNormalized()

    up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(direction, up).GetNormalized()
    true_up = Gf.Cross(right, direction).GetNormalized()
    rot = Gf.Matrix3d(
        right[0],
        right[1],
        right[2],
        true_up[0],
        true_up[1],
        true_up[2],
        -direction[0],
        -direction[1],
        -direction[2],
    )
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(rot)
    transform.SetTranslateOnly(eye)
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(transform)
    camera.CreateFocalLengthAttr().Set(18.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, max(distance * 4.0, 100.0)))

    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = "/World/Camera"
    except Exception as exc:
        print(f"[WARN] Could not set active viewport camera: {exc}")


def main() -> None:
    tools_root = Path(args_cli.tools_root).resolve()
    if not tools_root.is_dir():
        raise SystemExit(f"tools root does not exist: {tools_root}")

    tool_usds = discover_tool_usds(tools_root)
    if args_cli.max_tools > 0:
        tool_usds = tool_usds[: args_cli.max_tools]
    if not tool_usds:
        raise SystemExit(f"no USD files found under: {tools_root}")
    head_areas = {} if args_cli.no_tool_bboxes else load_tool_head_areas(Path(args_cli.tools_metadata).resolve())

    columns = args_cli.columns if args_cli.columns > 0 else max(1, math.ceil(math.sqrt(len(tool_usds))))
    rows = math.ceil(len(tool_usds) / columns)
    spacing = float(args_cli.spacing)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Tools")
    UsdGeom.Xform.Define(stage, "/World/Materials")

    grid_width = max(1, columns - 1) * spacing
    grid_height = max(1, rows - 1) * spacing
    extent = max(grid_width, grid_height) * 0.5 + args_cli.ground_margin
    spawn_ground(stage, extent)
    spawn_lights(stage)
    bbox_material = create_line_material(stage, "/World/Materials/FunctionalEndBBox", (1.0, 0.05, 0.02))

    translate_ops = []
    for index, (name, usd_path) in enumerate(tool_usds):
        row = index // columns
        col = index % columns
        x = (col - (columns - 1) * 0.5) * spacing
        y = ((rows - 1) * 0.5 - row) * spacing

        prim_name = f"Tool_{index:04d}_{safe_prim_name(name)}"
        prim = UsdGeom.Xform.Define(stage, f"/World/Tools/{prim_name}").GetPrim()
        prim.GetReferences().AddReference(str(usd_path))
        translate_op = set_xform(prim, (x, y, 0.0), args_cli.scale)
        translate_ops.append((prim, translate_op, x, y, name))

    # Let references compose, then push every tool up/down so its world bbox rests on z=0.
    for _ in range(3):
        simulation_app.update()

    rendered_bbox_count = 0
    for prim, translate_op, x, y, name in translate_ops:
        bbox = aligned_world_bbox(stage, prim)
        min_z = bbox.GetMin()[2]
        if math.isfinite(min_z):
            translate_op.Set(Gf.Vec3d(x, y, -min_z))
        else:
            print(f"[WARN] Could not compute bbox for {name}; leaving z=0")
        if head_areas and render_tool_bbox(stage, prim, name, head_areas, bbox_material):
            rendered_bbox_count += 1

    add_camera(stage, 0.0, 0.0, grid_width + spacing, grid_height + spacing)

    if args_cli.save_stage:
        out_path = Path(args_cli.save_stage).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out_path))
        print(f"[INFO] Saved stage: {out_path}")

    print(f"[INFO] Spawned {len(tool_usds)} tools on a {rows}x{columns} grid from {tools_root}")
    if not args_cli.no_tool_bboxes:
        print(f"[INFO] Rendered {rendered_bbox_count} functional-end bounding boxes")
    print("[INFO] Close the Isaac Sim window to exit.")

    if args_cli.duration > 0:
        steps = max(1, int(args_cli.duration * 60))
        for _ in range(steps):
            simulation_app.update()
    else:
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
#!/usr/bin/env python3
"""Minimal asset visibility test — follows view_tools.py pattern exactly.

Spawns a single USD at the origin, records a short video.

Usage:
    # Object:
    python pretrain/test_asset_visibility.py \
        --usd /path/to/coacd_usd/stem/stem.usd --scale 0.06 \
        --out object_test.mp4 --headless --enable_cameras

    # Tool:
    python pretrain/test_asset_visibility.py \
        --usd /path/to/objects_usd/name/name.usd --scale 0.1 \
        --out tool_test.mp4 --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="Path to .usd asset")
parser.add_argument("--scale", type=float, default=0.1)
parser.add_argument("--out", default="asset_test.mp4", help="Output video path")
parser.add_argument("--steps", type=int, default=60, help="Frames to capture")
AppLauncher.add_app_launcher_args(parser)
cli_args = parser.parse_args()

app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app

# ── Imports after AppLauncher ─────────────────────────────────────────────────
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
import numpy as np


def main():
    usd_path = Path(cli_args.usd).resolve()
    if not usd_path.exists():
        raise SystemExit(f"USD not found: {usd_path}")

    # ── Create stage (exactly like view_tools.py) ─────────────────────────────
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    # ── Ground plane (visible mesh, like view_tools.py) ───────────────────────
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    e = 3.0
    ground.CreatePointsAttr().Set([
        Gf.Vec3f(-e, -e, 0), Gf.Vec3f(e, -e, 0),
        Gf.Vec3f(e, e, 0),   Gf.Vec3f(-e, e, 0),
    ])
    ground.CreateFaceVertexCountsAttr().Set([4])
    ground.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr().Set("none")
    # Ground material
    gnd_mat = UsdShade.Material.Define(stage, "/World/Materials/Ground")
    gnd_shd = UsdShade.Shader.Define(stage, "/World/Materials/Ground/Shader")
    gnd_shd.CreateIdAttr("UsdPreviewSurface")
    gnd_shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.78, 0.74))
    gnd_shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    gnd_mat.CreateSurfaceOutput().ConnectToSource(gnd_shd.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(gnd_mat)

    # ── Lighting (exactly like view_tools.py) ─────────────────────────────────
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(800.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.95, 0.96, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    sun.CreateIntensityAttr().Set(2200.0)
    sun.CreateAngleAttr().Set(0.45)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    # ── Spawn asset via AddReference (exactly like view_tools.py) ─────────────
    s = cli_args.scale
    prim = UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim()
    prim.GetReferences().AddReference(str(usd_path))
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(s, s, s))

    # Let references compose
    for _ in range(3):
        simulation_app.update()

    # Ground the asset (push z so bbox min sits on z=0)
    import math
    bbox_cache = UsdGeom.BBoxCache(
        stage.GetTimeCodesPerSecond(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_z = bbox.GetMin()[2]
    if math.isfinite(min_z):
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, -min_z))
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(s, s, s))
        print(f"  Grounded asset: shifted z by {-min_z:.4f}")

    # Print prim tree
    print(f"\nPrim tree under /World/Asset:")
    for child in prim.GetAllChildren():
        print(f"  {child.GetPath()}  type={child.GetTypeName()}")

    # ── Camera (Replicator, for offscreen capture) ────────────────────────────
    import omni.replicator.core as rep
    rep_cam = rep.create.camera(
        position=(0.35, -0.35, 0.25),
        look_at=(0.0, 0.0, 0.05),
        focal_length=24.0,
        clipping_range=(0.01, 100.0),
    )
    render_prod = rep.create.render_product(rep_cam, (1280, 720))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_prod)

    # ── Capture frames ────────────────────────────────────────────────────────
    import imageio

    # Warm up renderer
    for _ in range(5):
        simulation_app.update()

    frames = []
    for i in range(cli_args.steps):
        simulation_app.update()
        rgba = annotator.get_data()
        if rgba is not None and rgba.size > 0 and rgba.max() > 0:
            frames.append(rgba[:, :, :3])

    if frames:
        Path(cli_args.out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(cli_args.out, frames, fps=30, quality=8)
        print(f"\n✓ Wrote {len(frames)} frames → {cli_args.out}")
    else:
        print("\n⚠ No frames captured!")

    import os
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

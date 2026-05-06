"""Minimal asset visibility test.

Spawns a single USD asset at canonical pose in Isaac Sim and records a short video.
Use to verify that tool / object USD assets are renderable.

Usage:
    # Test object only:
    python pretrain/test_asset_visibility.py \
        --usd /path/to/object.usd --scale 0.06 --out object_test.mp4

    # Test tool only:
    python pretrain/test_asset_visibility.py \
        --usd /path/to/tool.usd --scale 0.1 --out tool_test.mp4
"""

# ── Isaac Sim bootstrap ──────────────────────────────────────────────────────
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--usd", required=True, help="Path to .usd asset")
p.add_argument("--scale", type=float, default=0.1)
p.add_argument("--out", default="asset_test.mp4", help="Output video path")
p.add_argument("--steps", type=int, default=60, help="Frames to capture")
args = p.parse_args()

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True, "offscreen_render": True})

# Monkey-patch missing rendering preset (Isaac Sim 4.5+ standalone issue)
try:
    from omni.isaac.lab.sim import converters as _conv
    if not hasattr(_conv, "RenderingPreset"):
        import enum
        class _RP(enum.Enum):
            PERFORMANCE = 0
            QUALITY = 1
        _conv.RenderingPreset = _RP
except Exception:
    pass

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# ── Build scene ───────────────────────────────────────────────────────────────
sim_cfg = sim_utils.SimulationCfg(dt=1/60, gravity=(0.0, 0.0, -9.81))
sim_ctx = SimulationContext(sim_cfg)

from pxr import UsdGeom, UsdPhysics, UsdLux
stage = sim_utils.get_current_stage()

# Ground plane
gnd = "/World/GroundPlane"
UsdGeom.Xform.Define(stage, gnd)
plane = UsdGeom.Plane.Define(stage, f"{gnd}/Collision")
plane.CreateAxisAttr("Z")
plane.CreateDoubleSidedAttr(False)
UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

# Dome light
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(500.0)
dome.CreateColorAttr((1.0, 1.0, 1.0))

# Spawn asset
s = args.scale
cfg = sim_utils.UsdFileCfg(usd_path=args.usd, scale=(s, s, s))
cfg.func("/World/Asset", cfg, translation=(0.0, 0.0, 0.1))

sim_ctx.reset()

# Camera
import omni.replicator.core as rep
rep_cam = rep.create.camera(
    position=(0.3, -0.3, 0.25),
    look_at=(0.0, 0.0, 0.05),
    focal_length=24.0,
    clipping_range=(0.01, 100.0),
)
render_prod = rep.create.render_product(rep_cam, (1280, 720))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach(render_prod)

# ── Capture frames ───────────────────────────────────────────────────────────
import imageio
from pathlib import Path

frames = []
for i in range(args.steps):
    sim_ctx.step()
    sim_ctx.render()
    rgba = annotator.get_data()
    if rgba is not None and rgba.size > 0 and rgba.max() > 0:
        frames.append(rgba[:, :, :3])

if frames:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=8)
    print(f"✓ Wrote {len(frames)} frames → {args.out}")
else:
    print("⚠ No frames captured!")

# List prim children for debugging
asset_prim = stage.GetPrimAtPath("/World/Asset")
if asset_prim:
    print(f"\nPrim tree under /World/Asset:")
    for child in asset_prim.GetAllChildren():
        print(f"  {child.GetPath()}  type={child.GetTypeName()}")

import os
os._exit(0)

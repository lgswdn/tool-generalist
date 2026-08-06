#!/usr/bin/env python3
"""Render a DGN object using Tool-Generalist's asset-loading convention only."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_DGN_ROOT = Path("/mnt/project/world_model/tool_generalist/assets/DGN")
DEFAULT_OUTPUT_DIR = Path("/mnt/project/world_model/tool_generalist/grasp_result_dgn_visibility_tool_generalist")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure DGN asset visibility probe. This does not import experiment configs "
            "or RL runtime specs; it mirrors Tool-Generalist's DGN asset paths and "
            "spawns the object with IsaacLab RigidObjectCfg/MultiAssetSpawnerCfg."
        )
    )
    parser.add_argument("--dgn-root", type=Path, default=DEFAULT_DGN_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--entry", default=None, help="Full manifest entry, e.g. core-bottle-...-0.060")
    parser.add_argument("--index", type=int, default=0, help="Manifest index used when --entry is omitted")
    parser.add_argument(
        "--scale",
        default="0.10",
        help="'manifest' uses the entry suffix, 'spawn-default' uses RL's initial 0.01 spawn scale, or pass a numeric fixed scale.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-x", type=float, default=0.5)
    parser.add_argument("--object-y", type=float, default=0.0)
    parser.add_argument("--object-z", type=float, default=0.12)
    parser.add_argument("--disable-gravity", action="store_true")
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Opt into an interactive Isaac UI instead of headless rendering.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.gui:
        args.headless = True
    if not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


def _read_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    entries: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            entries.append(str(item.get("name", item.get("object_id", item.get("id")))))
        else:
            entries.append(str(item))
    return entries


def _base_and_scale(entry: str) -> tuple[str, float]:
    base, scale = str(entry).rsplit("-", 1)
    return base, float(scale)


def _resolve_entry(manifest: Path, entry: str | None, index: int) -> str:
    entries = _read_json_list(manifest)
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest}")
    if entry is None:
        if index < 0 or index >= len(entries):
            raise IndexError(f"--index {index} is outside manifest length {len(entries)}")
        return entries[index]
    if entry in entries:
        return entry
    base = entry.rsplit("-", 1)[0] if "-" in entry else entry
    matches = [candidate for candidate in entries if candidate.rsplit("-", 1)[0] == base]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Entry {entry!r} matched multiple manifest entries: {matches[:8]}")
    raise ValueError(f"Entry {entry!r} was not found in {manifest}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scale_from_arg(value: str, manifest_scale: float) -> float:
    lower = str(value).lower()
    if lower == "manifest":
        return manifest_scale
    if lower in {"spawn-default", "tool-generalist-spawn-default"}:
        return 0.01
    scale = float(value)
    if scale <= 0.0:
        raise ValueError("--scale must be positive")
    return scale


def _make_dgn_usd_cfg(usd_path: Path, scale: float) -> sim_utils.UsdFileCfg:
    # Mirrors the object portion of env_tool.load_object_candidates(), with scale
    # supplied directly instead of going through RL's prestartup scale event.
    return sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=(float(scale), float(scale), float(scale)),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=bool(args_cli.disable_gravity),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
    )


def _rgb_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    frame = rgb.detach().cpu().numpy()[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame


def _stage_scale_for_object(scene: InteractiveScene) -> list[list[float]]:
    import isaacsim.core.utils.prims as prim_utils

    prim_paths = sim_utils.find_matching_prim_paths(scene["object"].cfg.prim_path)
    scales = []
    for prim_path in prim_paths:
        prim = prim_utils.get_prim_at_path(prim_path)
        attr = prim.GetAttribute("xformOp:scale")
        value = attr.Get() if attr.IsValid() else None
        scales.append([float(v) for v in value] if value is not None else [])
    return scales


def _make_scene_cfg(object_cfg: sim_utils.UsdFileCfg):
    @configclass
    class DgnVisibilitySceneCfg(InteractiveSceneCfg):
        replicate_physics: bool = False

        if not args_cli.no_ground:
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
                debug_vis=False,
            )

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2800.0),
        )

        object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[object_cfg],
                random_choice=False,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=bool(args_cli.disable_gravity),
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(args_cli.object_x), float(args_cli.object_y), float(args_cli.object_z)),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        visibility_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/TGVisibilityCamera",
            update_period=0.0,
            height=int(args_cli.height),
            width=int(args_cli.width),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=1.50,
                horizontal_aperture=28.0,
                clipping_range=(0.02, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.25, 0.0, 0.85),
                rot=(-0.3337, 0.6234, 0.6234, -0.3337),
                convention="ros",
            ),
        )

    return DgnVisibilitySceneCfg(
        num_envs=int(args_cli.num_envs),
        env_spacing=float(args_cli.env_spacing),
        lazy_sensor_update=False,
    )


def main() -> int:
    dgn_root = args_cli.dgn_root.expanduser().resolve()
    manifest = args_cli.manifest.expanduser() if args_cli.manifest else dgn_root / "full_yes.json"
    manifest = manifest if manifest.is_absolute() else manifest.resolve()
    entry = _resolve_entry(manifest, args_cli.entry, int(args_cli.index))
    base, manifest_scale = _base_and_scale(entry)
    scale = _scale_from_arg(args_cli.scale, manifest_scale)
    usd_path = dgn_root / "coacd_usd" / base / f"{base}.usd"
    obj_path = dgn_root / "coacd_normalized" / f"{base}.obj"
    missing = [str(path) for path in (usd_path, obj_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DGN object {entry} is missing assets: {missing}")

    run_name = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{_safe_name(entry)}_scale_{scale:.3f}"
    )
    out_dir = args_cli.output_dir.expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_device = getattr(args_cli, "device", None) or "cuda:0"
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=sim_device))
    sim.set_camera_view(eye=[1.25, 0.0, 0.85], target=[0.5, 0.0, 0.02])

    scene = InteractiveScene(_make_scene_cfg(_make_dgn_usd_cfg(usd_path, scale)))
    sim.reset()
    scene.reset()

    sim_dt = sim.get_physics_dt()
    for _ in range(max(0, int(args_cli.warmup_steps))):
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)

    sim.render()
    scene["visibility_camera"].update(dt=0.0, force_recompute=True)
    rgb = scene["visibility_camera"].data.output["rgb"][0]
    frame_rgb = _rgb_to_uint8(rgb)
    image_path = out_dir / "tool_generalist_asset_dgn_visibility.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    object_asset = scene["object"]
    object_pos_env = (object_asset.data.root_pos_w[:, :3] - scene.env_origins[:, :3]).detach().cpu().numpy()
    object_quat = object_asset.data.root_quat_w.detach().cpu().numpy()
    object_lin_vel = object_asset.data.root_lin_vel_w.detach().cpu().numpy()
    object_ang_vel = object_asset.data.root_ang_vel_w.detach().cpu().numpy()
    summary = {
        "entry": entry,
        "base": base,
        "manifest": str(manifest),
        "manifest_scale": manifest_scale,
        "requested_scale": scale,
        "usd_path": str(usd_path),
        "obj_path": str(obj_path),
        "image": str(image_path),
        "disable_gravity": bool(args_cli.disable_gravity),
        "stage_object_scale_xyz": _stage_scale_for_object(scene),
        "object_pos_env": object_pos_env.tolist(),
        "object_quat_wxyz": object_quat.tolist(),
        "object_lin_vel_w": object_lin_vel.tolist(),
        "object_ang_vel_w": object_ang_vel.tolist(),
        "camera": {
            "pos": [1.25, 0.0, 0.85],
            "lookat": [0.5, 0.0, 0.02],
            "width": int(args_cli.width),
            "height": int(args_cli.height),
        },
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"[INFO] entry={entry}", flush=True)
    print(f"[INFO] requested_scale={scale}", flush=True)
    print(f"[INFO] stage_object_scale_xyz={summary['stage_object_scale_xyz']}", flush=True)
    print(f"[INFO] object_pos_env={summary['object_pos_env']}", flush=True)
    print(f"[INFO] image={image_path}", flush=True)
    print(f"[INFO] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

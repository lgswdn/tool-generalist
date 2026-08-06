#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record exactly the first fixed generated-gripper episodes for one checkpoint.

Example:
    python scripts/record_fixed_episodes.py \
        --runtime_spec /path/to/rl_runtime_spec.json \
        --checkpoint /path/to/model_500.pt \
        --video_dir /path/to/output/model_500 \
        --num_episodes 15 --seed 0 --object_random_seed 0

This script must be run on a machine with a usable Isaac/IsaacLab runtime.
It records env IDs 0..N-1 once each, keeps the first completed episode
regardless of outcome, and never replaces an episode with a later reset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from scripts.fixed_episode_runtime import backfill_legacy_fixed_episode_fields
from utils.experiment.rl_runtime_spec import RUNTIME_SPEC_ENV_VAR, validate_runtime_spec


FFMPEG_PATH = "/usr/bin/ffmpeg"
DEFAULT_RECORD_RESOLUTION = (512, 512)
MODE_NAME = "record_fixed_episodes"
SUMMARY_NAME = "record_fixed_episodes_manifest.json"
COMMAND_NAME = "target_object_pose"


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) == 0:
        raise ValueError(f"Could not create a safe filename from: {name}")
    return safe


def _backfill_runtime_spec_defaults(spec: dict[str, Any]) -> None:
    """Fill policy fields that older runtime specs omitted but current configs default."""

    backfill_legacy_fixed_episode_fields(spec)
    policy = spec.get("policy_params")
    if not isinstance(policy, dict):
        return
    observation = spec.get("observation_params")
    if not isinstance(observation, dict):
        observation = {}
    policy.setdefault("model_input_centering", observation.get("model_input_centering", "bbox_center"))
    policy.setdefault("relative_translation_query_tokens", 2)
    policy.setdefault("reuse_pretrain_pose_cross_attn", False)


parser = argparse.ArgumentParser(
    description=(
        "Record exactly the first completed episode from generated-gripper env IDs 0..N-1."
    )
)
parser.add_argument("--runtime_spec", type=str, required=True, help="Exact rl_runtime_spec.json to replay.")
parser.add_argument("--checkpoint", type=str, required=True, help="Exact RSL-RL checkpoint to load.")
parser.add_argument("--num_episodes", type=int, default=15, help="Number of fixed first episodes/envs.")
parser.add_argument("--seed", type=int, default=0, help="Fixed environment and agent seed.")
parser.add_argument(
    "--object_random_seed",
    type=int,
    default=0,
    help="Fixed generated-gripper/object assignment seed.",
)
parser.add_argument("--task", type=str, default=None, help="Task name. Defaults to runtime_spec['task_id'].")
parser.add_argument("--video_dir", type=str, required=True, help="Directory for MP4s and metadata.")
parser.add_argument(
    "--checkpoint_label",
    type=str,
    default=None,
    help="Short label shown in overlays. Defaults to checkpoint stem.",
)
parser.add_argument("--video_width", type=int, default=DEFAULT_RECORD_RESOLUTION[0])
parser.add_argument("--video_height", type=int, default=DEFAULT_RECORD_RESOLUTION[1])
parser.add_argument("--video_fps", type=int, default=10)
parser.add_argument(
    "--disable_recording_visual_overrides",
    action="store_true",
    default=False,
    help="Disable recording-only visual overrides.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be positive")
if args_cli.seed < 0:
    parser.error("--seed must be >= 0")
if args_cli.object_random_seed < 0:
    parser.error("--object_random_seed must be >= 0")
if args_cli.video_width <= 0 or args_cli.video_height <= 0:
    parser.error("--video_width and --video_height must be positive")
if args_cli.video_fps <= 0:
    parser.error("--video_fps must be positive")
if not os.path.isfile(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found: {FFMPEG_PATH}")

runtime_spec_path = os.path.abspath(os.path.normpath(args_cli.runtime_spec))
checkpoint_arg = os.path.abspath(os.path.normpath(args_cli.checkpoint))
video_dir = os.path.abspath(os.path.normpath(args_cli.video_dir))
if not os.path.isfile(runtime_spec_path):
    raise FileNotFoundError(f"runtime_spec does not exist: {runtime_spec_path}")
if not os.path.isfile(checkpoint_arg):
    raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_arg}")
os.makedirs(video_dir, exist_ok=True)

with open(runtime_spec_path, "r", encoding="utf-8") as stream:
    runtime_spec = json.load(stream)
if not isinstance(runtime_spec, dict):
    raise RuntimeError(f"Runtime spec must be a JSON object: {runtime_spec_path}")

runtime_env_params = runtime_spec.get("env_params")
if not isinstance(runtime_env_params, dict):
    raise RuntimeError("runtime_spec env_params must be a JSON object")
runtime_robot_mode = str(runtime_env_params.get("robot_mode", ""))
if runtime_robot_mode != "generated_gripper":
    raise RuntimeError(
        "record_fixed_episodes.py requires runtime_spec env_params.robot_mode='generated_gripper'"
    )
if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id")
if not args_cli.task:
    parser.error("--task is required when runtime_spec does not contain task_id")

paths_yaml = runtime_spec.get("paths_yaml")
if not isinstance(paths_yaml, str) or not paths_yaml.strip():
    raise RuntimeError("runtime_spec must contain paths_yaml")
paths_yaml = os.path.abspath(os.path.normpath(paths_yaml))
if not os.path.isfile(paths_yaml):
    raise FileNotFoundError(f"paths_yaml does not exist: {paths_yaml}")

checkpoint_label = args_cli.checkpoint_label or Path(checkpoint_arg).stem

os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = "0"
os.environ["TOOL_GENERALIST_LOCAL_RANK"] = "0"
os.environ["TOOL_GENERALIST_WORLD_SIZE"] = "1"
os.environ["TOOL_GENERALIST_PATHS_YAML"] = paths_yaml
os.environ["TOOL_GENERALIST_OBJECT_ASSIGNMENT_SEED"] = str(args_cli.object_random_seed)

eval_runtime_spec = copy.deepcopy(runtime_spec)
eval_runtime_spec["mode"] = MODE_NAME
eval_runtime_spec["seed"] = int(args_cli.seed)
eval_runtime_spec["num_envs"] = int(args_cli.num_episodes)
eval_runtime_spec["paths_yaml"] = paths_yaml
eval_env_params = eval_runtime_spec.setdefault("env_params", {})
if not isinstance(eval_env_params, dict):
    raise RuntimeError("runtime_spec env_params must be a JSON object")
eval_env_params["num_envs"] = int(args_cli.num_episodes)
eval_env_params["robot_mode"] = "generated_gripper"
launch_params = eval_runtime_spec.setdefault("launch_params", {})
if not isinstance(launch_params, dict):
    raise RuntimeError("runtime_spec launch_params must be a JSON object")
launch_params["init_at_random_ep_len"] = False
asset_assignment = eval_runtime_spec.setdefault("asset_assignment_params", {})
if not isinstance(asset_assignment, dict):
    raise RuntimeError("runtime_spec asset_assignment_params must be a JSON object")
asset_assignment["seed"] = int(args_cli.object_random_seed)
asset_assignment["randomize_tool_assignment"] = True
asset_assignment["randomize_object_assignment"] = True
_backfill_runtime_spec_defaults(eval_runtime_spec)

eval_runtime_spec_path = os.path.join(video_dir, "effective_rl_runtime_spec.json")
validate_runtime_spec(eval_runtime_spec, eval_runtime_spec_path)
with open(eval_runtime_spec_path, "w", encoding="utf-8") as stream:
    json.dump(eval_runtime_spec, stream, ensure_ascii=True, indent=2, sort_keys=True)
    stream.write("\n")
os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path

args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below requires Isaac Sim to be launched."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import env_tool as env_tool_module
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.asset_assignment import (
    GENERATED_GRIPPER_ASSIGNMENT_SALT,
    OBJECT_ASSIGNMENT_SALT,
    asset_indices_for_rank,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    GENERATED_GRIPPER_DATA,
    OBJECT_ASSET_CFGS,
    get_generated_gripper_data_for_env,
    get_generated_gripper_index_for_env,
    get_object_asset_cfg_for_env,
    get_object_index_for_env,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.events import (
    get_rigid_body_scale,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
    phys_params,
)
from scripts.video_diagnostics import (
    format_recording_diagnostics,
    overlay_recording_diagnostics,
    recording_debug_metrics,
)


def _set_deterministic_seeds(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _object_names_from_loaded_data() -> list[str]:
    names = []
    for index, cfg in enumerate(OBJECT_ASSET_CFGS):
        obj_path = getattr(cfg, "obj_path", None)
        usd_path = getattr(cfg, "usd_path", None)
        if obj_path:
            names.append(Path(str(obj_path)).stem)
        elif usd_path:
            names.append(Path(str(usd_path)).stem)
        else:
            raise RuntimeError(f"Object asset config {index} has neither obj_path nor usd_path")
    if not names:
        raise RuntimeError("No objects were loaded into OBJECT_ASSET_CFGS.")
    return names


def _generated_gripper_names_from_loaded_data() -> list[str]:
    names = [
        str(getattr(asset, "gripper_id", ""))
        for asset in GENERATED_GRIPPER_DATA
    ]
    if not names or any(not name for name in names):
        raise RuntimeError("Generated gripper manifest did not provide non-empty gripper_id values.")
    return names


def _apply_asset_assignment_seed(
    seed: int,
    num_envs: int,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> tuple[list[int], list[int]]:
    gripper_indices = asset_indices_for_rank(
        int(num_envs),
        0,
        len(env_tool_module.GENERATED_GRIPPER_DATA),
        randomize=True,
        seed=int(seed),
        salt=GENERATED_GRIPPER_ASSIGNMENT_SALT,
    )
    gripper_usd_paths = [
        env_tool_module.GENERATED_GRIPPER_USD_PATHS[index] for index in gripper_indices
    ]
    env_tool_module.GENERATED_GRIPPER_ASSET_INDICES_BY_ENV[:] = gripper_indices
    env_tool_module.GENERATED_GRIPPER_USD_PATHS_BY_ENV[:] = gripper_usd_paths
    env_tool_module.GENERATED_GRIPPER_SPAWN_ASSET_INDICES[:] = gripper_indices
    env_tool_module.GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN[:] = gripper_usd_paths
    if hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot.spawn.usd_path = env_tool_module.GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN

    object_indices = asset_indices_for_rank(
        int(num_envs),
        0,
        len(env_tool_module.OBJECT_ASSET_CFGS),
        randomize=True,
        seed=int(seed),
        salt=OBJECT_ASSIGNMENT_SALT,
    )
    object_spawn_cfgs = [env_tool_module.OBJECT_ASSET_CFGS[index] for index in object_indices]
    env_tool_module.OBJECT_ASSET_INDICES_BY_ENV[:] = object_indices
    env_tool_module.OBJECT_ASSET_CFGS_BY_ENV[:] = object_spawn_cfgs
    env_tool_module.OBJECT_SPAWN_ASSET_INDICES[:] = object_indices
    env_tool_module.OBJECT_ASSET_CFGS_FOR_SPAWN[:] = object_spawn_cfgs
    if hasattr(env_cfg.scene, "object"):
        env_cfg.scene.object.spawn.assets_cfg = env_tool_module.OBJECT_ASSET_CFGS_FOR_SPAWN
    return gripper_indices, object_indices


def _make_record_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/EvalRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.25, 0.0, 0.85),
            rot=(-0.3337, 0.6234, 0.6234, -0.3337),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.50,
            horizontal_aperture=28.0,
            clipping_range=(0.05, 20.0),
        ),
        width=args_cli.video_width,
        height=args_cli.video_height,
    )


def _disable_debug_pointcloud_rendering(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    for attr in (
        "visualize_current_object_pose",
        "visualize_object_pointcloud",
        "visualize_tool_pointcloud",
        "visualize_tool1_pointcloud",
        "visualize_tool2_pointcloud",
        "visualize_tool_head_area",
        "visualize_head_area_center",
        "visualize_eef_position",
        "visualize_object_velocity_mass",
        "visualize_tool_velocity_mass",
    ):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)


def _apply_recording_visual_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    light = getattr(scene, "light", None)
    light_spawn = getattr(light, "spawn", None)
    if light_spawn is not None:
        if hasattr(light_spawn, "color"):
            light_spawn.color = (0.75, 0.75, 0.75)
        if hasattr(light_spawn, "intensity"):
            light_spawn.intensity = 1500.0
    if hasattr(scene, "terrain") and getattr(scene, "terrain") is not None:
        terrain = getattr(scene, "terrain")
        if hasattr(terrain, "visual_material"):
            terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
    if hasattr(scene, "table") and getattr(scene, "table") is not None:
        table_spawn = getattr(scene.table, "spawn", None)
        if table_spawn is not None and hasattr(table_spawn, "visual_material"):
            table_spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45))


def _start_ffmpeg_writer(path: str) -> subprocess.Popen:
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args_cli.video_width}x{args_cli.video_height}",
        "-r",
        str(args_cli.video_fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _close_ffmpeg_writer(writer: subprocess.Popen) -> None:
    if writer.stdin is not None:
        writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def _discard_video_tmp(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")


def _write_failure_summary(stage: str, exc: BaseException, *, overwrite: bool) -> None:
    summary_path = Path(video_dir) / SUMMARY_NAME
    if summary_path.is_file() and not overwrite:
        return
    payload = {
        "schema_version": "record_fixed_episodes_manifest_v1",
        "status": "failed",
        "stage": stage,
        "task": args_cli.task,
        "checkpoint": checkpoint_arg,
        "checkpoint_label": checkpoint_label,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "paths_yaml": paths_yaml,
        "num_episodes": int(args_cli.num_episodes),
        "seed": int(args_cli.seed),
        "object_random_seed": int(args_cli.object_random_seed),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }
    try:
        _json_dump(summary_path, payload)
    except Exception:
        print(f"[ERROR] Could not write failed recorder manifest: {summary_path}", file=sys.stderr)
        traceback.print_exc()


def _require_complete_summary() -> None:
    summary_path = Path(video_dir) / SUMMARY_NAME
    if not summary_path.is_file():
        raise RuntimeError(
            "Hydra recorder entry returned without producing the required manifest: "
            f"{summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        error = payload.get("error") if isinstance(payload, dict) else None
        raise RuntimeError(f"Recorder did not complete: manifest={summary_path} error={error!r}")


def _require_scene_asset(base_env: Any, name: str) -> Any:
    try:
        return base_env.scene[name]
    except Exception as exc:
        raise RuntimeError(f"Environment scene is missing required asset {name!r}") from exc


def _require_data_tensor(data: Any, attr: str, env_id: int, width: int | None = None) -> list[float]:
    tensor = getattr(data, attr, None)
    if tensor is None:
        raise RuntimeError(f"Required tensor is missing: {data!r}.{attr}")
    if int(tensor.shape[0]) <= int(env_id):
        raise RuntimeError(f"Tensor {attr} has no row for env_id={env_id}")
    row = tensor[int(env_id)].detach().cpu().reshape(-1).tolist()
    if width is not None and len(row) != width:
        raise RuntimeError(f"Tensor {attr} row width {len(row)} != expected {width}")
    return [float(value) for value in row]


def _require_vector_from_tensor(tensor: torch.Tensor, env_id: int, label: str, width: int) -> list[float]:
    if int(tensor.shape[0]) <= int(env_id):
        raise RuntimeError(f"{label} has no row for env_id={env_id}")
    row = tensor[int(env_id)].detach().cpu().reshape(-1).tolist()
    if len(row) != width:
        raise RuntimeError(f"{label} row width {len(row)} != expected {width}")
    return [float(value) for value in row]


def _tensor_scalar(value: Any) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().item())
    return int(value)


def _object_asset_name(cfg: Any, index: int) -> str:
    obj_path = getattr(cfg, "obj_path", None)
    usd_path = getattr(cfg, "usd_path", None)
    if obj_path:
        return Path(str(obj_path)).stem
    if usd_path:
        return Path(str(usd_path)).stem
    raise RuntimeError(f"Object asset config {index} has neither obj_path nor usd_path")


def _asset_scale_xyz(cfg: Any) -> list[float]:
    scale = getattr(cfg, "scale", None)
    if scale is None:
        raise RuntimeError(f"Object asset config is missing scale: {cfg!r}")
    values = list(scale) if isinstance(scale, (list, tuple)) else [float(scale)] * 3
    if len(values) != 3:
        raise RuntimeError(f"Object asset config scale must have 3 values, got {values!r}")
    return [float(value) for value in values]


def _command_task_metadata(base_env: Any, env_id: int) -> dict[str, Any]:
    try:
        term = base_env.command_manager.get_term(COMMAND_NAME)
    except Exception as exc:
        raise RuntimeError(f"Could not access command term {COMMAND_NAME!r}") from exc
    task_index = getattr(term, "target_pose_task_index", None)
    task_names = getattr(term, "target_pose_task_names", None)
    task_is_stable = getattr(term, "target_pose_task_is_stable", None)
    if task_index is None or task_names is None or task_is_stable is None:
        raise RuntimeError(f"Command term {COMMAND_NAME!r} is missing target task metadata")
    task_idx = _tensor_scalar(task_index[int(env_id)])
    names = [str(name) for name in task_names]
    if task_idx < 0 or task_idx >= len(names):
        raise RuntimeError(f"Command task index {task_idx} is outside task names {names!r}")
    return {
        "task_index": task_idx,
        "task_name": names[task_idx],
        "is_stable": bool(task_is_stable[int(env_id)].detach().cpu().item()),
    }


def _physics_metadata(base_env: Any, env_id: int) -> dict[str, Any]:
    field_names = runtime_spec.get("physics_observation_fields")
    if not isinstance(field_names, list):
        raise RuntimeError("runtime_spec physics_observation_fields must be a list")
    fields = tuple(str(name) for name in field_names)
    physics = phys_params(base_env, field_names=fields)
    if int(physics.shape[0]) <= int(env_id):
        raise RuntimeError(f"phys_params has no row for env_id={env_id}")
    if int(physics.shape[1]) != len(fields):
        raise RuntimeError(
            f"phys_params width {int(physics.shape[1])} does not match fields {len(fields)}"
        )
    row = physics[int(env_id)].detach().cpu().reshape(-1).tolist()
    return {
        "field_names": list(fields),
        "values": {field: float(row[index]) for index, field in enumerate(fields)},
    }


def _capture_episode_identity(base_env: Any, env_id: int) -> dict[str, Any]:
    robot = _require_scene_asset(base_env, "robot")
    obj = _require_scene_asset(base_env, "object")
    env_origins = getattr(base_env.scene, "env_origins", None)
    if env_origins is None:
        raise RuntimeError("Environment scene is missing env_origins")

    gripper_index = int(get_generated_gripper_index_for_env(env_id))
    gripper = get_generated_gripper_data_for_env(env_id)
    object_index = int(get_object_index_for_env(env_id))
    object_cfg = get_object_asset_cfg_for_env(env_id)
    runtime_scale = get_rigid_body_scale(base_env, SceneEntityCfg("object"), [env_id])[0]

    command = base_env.command_manager.get_command(COMMAND_NAME)
    target_pose = _require_vector_from_tensor(command, env_id, f"command {COMMAND_NAME}", 7)
    object_root_pos = torch.as_tensor(obj.data.root_pos_w[int(env_id), :3])
    object_pose_env = object_root_pos - env_origins[int(env_id), :3]

    return {
        "episode_index": int(env_id),
        "env_id": int(env_id),
        "robot_mode": "generated_gripper",
        "task": str(args_cli.task),
        "seeds": {
            "env_seed": int(args_cli.seed),
            "object_random_seed": int(args_cli.object_random_seed),
        },
        "generated_gripper": {
            "index": gripper_index,
            "gripper_id": str(gripper.gripper_id),
            "usd_path": str(gripper.usd_path),
            "root_dir": str(gripper.root_dir),
            "finger_joint_names": [str(name) for name in gripper.finger_joint_names],
            "open_joint_pos": float(gripper.open_joint_pos),
            "ee_body_name": str(gripper.ee_body_name),
        },
        "object": {
            "index": object_index,
            "name": _object_asset_name(object_cfg, object_index),
            "usd_path": str(getattr(object_cfg, "usd_path")),
            "obj_path": str(getattr(object_cfg, "obj_path")),
            "asset_config_scale_xyz": _asset_scale_xyz(object_cfg),
            "runtime_scale_xyz": [float(value) for value in runtime_scale.detach().cpu().tolist()],
        },
        "initial_robot": {
            "joint_names": [str(name) for name in robot.data.joint_names],
            "joint_pos": _require_data_tensor(robot.data, "joint_pos", env_id),
            "joint_vel": _require_data_tensor(robot.data, "joint_vel", env_id),
            "root_pos_w": _require_data_tensor(robot.data, "root_pos_w", env_id, 3),
            "root_quat_w": _require_data_tensor(robot.data, "root_quat_w", env_id, 4),
        },
        "initial_object": {
            "root_pos_w": _require_data_tensor(obj.data, "root_pos_w", env_id, 3),
            "root_quat_w": _require_data_tensor(obj.data, "root_quat_w", env_id, 4),
            "pose_env": [
                *[float(value) for value in object_pose_env.detach().cpu().tolist()],
                *_require_data_tensor(obj.data, "root_quat_w", env_id, 4),
            ],
        },
        "target_pose": {
            "command_name": COMMAND_NAME,
            "pose_env": target_pose,
            **_command_task_metadata(base_env, env_id),
        },
        "physics": _physics_metadata(base_env, env_id),
    }


def _output_gate_metrics_for_env(policy: Any, env_id: int) -> dict[str, Any] | None:
    gate = getattr(policy, "_last_actor_gate", None) if policy is not None else None
    if gate is None:
        return None
    if int(gate.shape[0]) <= int(env_id):
        return None
    gate_row = gate[int(env_id)].detach().float().reshape(-1).cpu()
    if gate_row.numel() == 0:
        return None
    expert_a = float(gate_row.mean().item())
    return {
        "output_gate_expert_a_weight": expert_a,
        "output_gate_expert_b_weight": 1.0 - expert_a,
        "output_gate_expert_a_min": float(gate_row.min().item()),
        "output_gate_expert_a_max": float(gate_row.max().item()),
        "output_gate_selected_expert": "model_a" if expert_a >= 0.5 else "model_b",
    }


def _overlay_checkpoint_label(frame: Any, *, episode_index: int, env_id: int) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV/cv2 is required for checkpoint labels in fixed episode videos") from exc
    label = f"{checkpoint_label}  episode {episode_index:02d}  env {env_id:02d}"
    height, width = frame.shape[:2]
    font_scale = max(0.48, min(width, height) / 900.0)
    thickness = 1 if min(width, height) < 700 else 2
    y0 = max(0, height - int(36 * font_scale) - 12)
    cv2.rectangle(frame, (6, y0), (width - 6, height - 6), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        label,
        (14, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def _make_episode_video_record(env_id: int, identity: dict[str, Any]) -> dict[str, Any]:
    episode_index = int(env_id)
    stem = f"episode_{episode_index:03d}_env_{int(env_id):03d}"
    tmp_path = os.path.join(video_dir, f"{stem}.tmp.mp4")
    video_path = os.path.join(video_dir, f"{stem}.mp4")
    metadata_path = os.path.join(video_dir, f"{stem}.json")
    writer = _start_ffmpeg_writer(tmp_path)
    return {
        "episode_index": episode_index,
        "env_id": int(env_id),
        "identity": identity,
        "tmp_path": tmp_path,
        "video_path": video_path,
        "metadata_path": metadata_path,
        "writer": writer,
        "frames": 0,
        "last_diag": None,
    }


def _capture_video_frames(
    env: Any,
    active_records: dict[int, dict[str, Any]],
    *,
    env_ids: set[int] | None = None,
    policy: Any = None,
    step: int | None = None,
) -> None:
    if not active_records:
        return
    env.unwrapped.sim.render()
    env.unwrapped.scene["eval_record_camera"].update(dt=0.0, force_recompute=True)
    rgb_all = env.unwrapped.scene["eval_record_camera"].data.output["rgb"]
    for record in list(active_records.values()):
        env_id = int(record["env_id"])
        if env_ids is not None and env_id not in env_ids:
            continue
        writer = record.get("writer")
        if writer is None or writer.stdin is None:
            raise RuntimeError(f"Episode {record['episode_index']} has no active ffmpeg stdin")
        frame_tensor = rgb_all[env_id, ..., :3].detach().cpu()
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = torch.clamp(frame_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
        frame = frame_tensor.contiguous().numpy().copy()
        metrics = recording_debug_metrics(
            env.unwrapped,
            env_id,
            runtime_spec.get("reward_params", {}),
        )
        gate_metrics = _output_gate_metrics_for_env(policy, env_id)
        if gate_metrics is not None:
            metrics.update(gate_metrics)
        record["last_diag"] = format_recording_diagnostics(metrics, step=step)
        frame = overlay_recording_diagnostics(frame, metrics, step=step)
        frame = _overlay_checkpoint_label(
            frame,
            episode_index=int(record["episode_index"]),
            env_id=env_id,
        )
        writer.stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1


def _finish_episode_record(
    active_records: dict[int, dict[str, Any]],
    env_id: int,
    *,
    success: bool,
    step: int,
) -> dict[str, Any]:
    record = active_records.pop(int(env_id), None)
    if record is None:
        raise RuntimeError(f"No active recording for env_id={env_id}")
    writer = record.get("writer")
    if writer is not None:
        _close_ffmpeg_writer(writer)
        record["writer"] = None
    if int(record["frames"]) <= 0:
        raise RuntimeError(f"Episode {record['episode_index']} recorded zero frames")
    os.replace(record["tmp_path"], record["video_path"])

    payload = {
        "schema_version": "fixed_episode_record_v1",
        "status": "complete",
        "episode_index": int(record["episode_index"]),
        "env_id": int(record["env_id"]),
        "checkpoint": checkpoint_arg,
        "checkpoint_label": checkpoint_label,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "paths_yaml": paths_yaml,
        "identity": record["identity"],
        "outcome": {
            "success": bool(success),
            "ended_step": int(step),
            "frames": int(record["frames"]),
            "last_diagnostics": record.get("last_diag"),
        },
        "video": {
            "path": record["video_path"],
            "fps": int(args_cli.video_fps),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
        },
    }
    _json_dump(record["metadata_path"], payload)
    return {
        "episode_index": int(record["episode_index"]),
        "env_id": int(record["env_id"]),
        "metadata_path": record["metadata_path"],
        "video_path": record["video_path"],
        "success": bool(success),
        "ended_step": int(step),
        "frames": int(record["frames"]),
    }


def _close_active_records(active_records: dict[int, dict[str, Any]]) -> None:
    for record in list(active_records.values()):
        writer = record.get("writer")
        if writer is not None:
            _close_ffmpeg_writer(writer)
        _discard_video_tmp(str(record["tmp_path"]))
    active_records.clear()


def _write_summary(
    *,
    resume_path: str,
    gripper_indices: list[int],
    object_indices: list[int],
    episodes: list[dict[str, Any]],
    steps_completed: int,
) -> None:
    if len(episodes) != args_cli.num_episodes:
        raise RuntimeError(
            f"Cannot write complete summary with {len(episodes)} of {args_cli.num_episodes} episodes"
        )
    payload = {
        "schema_version": "record_fixed_episodes_manifest_v1",
        "status": "complete",
        "task": args_cli.task,
        "robot_mode": runtime_robot_mode,
        "checkpoint": resume_path,
        "checkpoint_label": checkpoint_label,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "paths_yaml": paths_yaml,
        "rank": 0,
        "world_size": 1,
        "num_episodes": int(args_cli.num_episodes),
        "num_envs_per_rank": int(args_cli.num_episodes),
        "seed": int(args_cli.seed),
        "object_random_seed": int(args_cli.object_random_seed),
        "video_fps": int(args_cli.video_fps),
        "video_width": int(args_cli.video_width),
        "video_height": int(args_cli.video_height),
        "steps_completed": int(steps_completed),
        "asset_assignment": {
            "generated_gripper_indices_by_env": list(gripper_indices),
            "object_indices_by_env": list(object_indices),
        },
        "episodes": sorted(episodes, key=lambda item: int(item["episode_index"])),
    }
    summary_path = os.path.join(video_dir, SUMMARY_NAME)
    _json_dump(summary_path, payload)
    print(f"[INFO] Saved fixed-episode summary: {summary_path}", flush=True)


def _record_main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
    _set_deterministic_seeds(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_episodes
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.disable_obs_noise = True
    env_cfg.scene.eval_record_camera = _make_record_camera_cfg()
    _disable_debug_pointcloud_rendering(env_cfg)
    if not args_cli.disable_recording_visual_overrides:
        _apply_recording_visual_overrides(env_cfg)

    resume_path = retrieve_file_path(checkpoint_arg)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {resume_path}")

    gripper_names = _generated_gripper_names_from_loaded_data()
    object_names = _object_names_from_loaded_data()
    gripper_indices, object_indices = _apply_asset_assignment_seed(
        args_cli.object_random_seed,
        args_cli.num_episodes,
        env_cfg,
    )
    print(
        f"[INFO] Recording fixed first episodes: checkpoint={resume_path} "
        f"num_episodes={args_cli.num_episodes} seed={args_cli.seed} "
        f"object_random_seed={args_cli.object_random_seed}",
        flush=True,
    )
    print(
        f"[INFO] Loaded generated_grippers={len(gripper_names)} objects={len(object_names)} "
        f"video_dir={video_dir}",
        flush=True,
    )

    env = None
    active_records: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    steps = 0
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)
        policy_module = ppo_runner.alg.policy

        if not hasattr(env.unwrapped, "episode_success_buf"):
            raise AttributeError("Environment does not have episode_success_buf.")
        if int(env.unwrapped.num_envs) != int(args_cli.num_episodes):
            raise RuntimeError(
                f"Environment num_envs={env.unwrapped.num_envs} does not match num_episodes={args_cli.num_episodes}"
            )

        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
        identities = {
            env_id: _capture_episode_identity(env.unwrapped, env_id)
            for env_id in range(args_cli.num_episodes)
        }
        for env_id in range(args_cli.num_episodes):
            active_records[env_id] = _make_episode_video_record(env_id, identities[env_id])
        pending = set(range(args_cli.num_episodes))
        _capture_video_frames(env, active_records, env_ids=pending, policy=policy_module, step=0)

        while pending and simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = inference_policy(obs)
                obs, _, dones, _ = env.step(actions)
            steps += 1
            _capture_video_frames(env, active_records, env_ids=pending, policy=policy_module, step=steps)

            ended = dones.bool()
            if torch.any(ended):
                ended_env_ids = set(int(env_id) for env_id in torch.where(ended)[0].tolist())
                for env_id in sorted(ended_env_ids.intersection(pending)):
                    if not hasattr(env.unwrapped, "_episode_success_before_reset"):
                        raise AttributeError("Environment does not have _episode_success_before_reset.")
                    success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                    completed.append(
                        _finish_episode_record(
                            active_records,
                            env_id,
                            success=success,
                            step=steps,
                        )
                    )
                    pending.remove(env_id)
                    print(
                        f"[PROGRESS] saved episode={env_id} success={success} "
                        f"completed={len(completed)}/{args_cli.num_episodes} step={steps}",
                        flush=True,
                    )

            elapsed = time.time() - start_time
            print(
                f"[PROGRESS] step={steps} pending={len(pending)} "
                f"completed={len(completed)}/{args_cli.num_episodes} "
                f"step_time={elapsed:.4f}s",
                flush=True,
            )
            if args_cli.real_time and dt is not None:
                sleep_time = float(dt) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if pending:
            raise RuntimeError(
                f"Simulation stopped before fixed first episodes completed; pending env IDs: {sorted(pending)}"
            )
    finally:
        _close_active_records(active_records)
        if env is not None:
            env.close()

    _write_summary(
        resume_path=resume_path,
        gripper_indices=gripper_indices,
        object_indices=object_indices,
        episodes=completed,
        steps_completed=steps,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
    try:
        _record_main(env_cfg, agent_cfg)
    except BaseException as exc:
        _write_failure_summary("recorder_body", exc, overwrite=True)
        raise


if __name__ == "__main__":
    try:
        main()
        _require_complete_summary()
    except BaseException as exc:
        _write_failure_summary("entrypoint", exc, overwrite=False)
        raise
    finally:
        simulation_app.close()

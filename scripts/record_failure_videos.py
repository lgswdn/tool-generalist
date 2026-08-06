# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record full failed policy episodes with random tools and random objects."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from utils.artifacts.resolver import resolve_artifacts
from utils.config.loader import load_exp_cfg
from utils.experiment.rl_runtime_spec import RUNTIME_SPEC_ENV_VAR, validate_runtime_spec


FFMPEG_PATH = "/usr/bin/ffmpeg"
PANDA_GRIPPER_RECORD_CAMERA_EYE = (1.5, 0.5, 0.7)
PANDA_GRIPPER_RECORD_CAMERA_LOOKAT = (0.4, 0.0, 0.1)
PANDA_GRIPPER_RECORD_CAMERA_ROT_ROS = (-0.2853, 0.4601, 0.7146, -0.4430)
PANDA_GRIPPER_RECORD_RESOLUTION = (1280, 720)
DEFAULT_TOOL_RECORD_RESOLUTION = (512, 512)


def _distributed_rank_info(distributed: bool) -> tuple[int, int]:
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size
    return 0, 1


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) == 0:
        raise ValueError(f"Could not create a safe filename from: {name}")
    return safe


def _backfill_runtime_spec_defaults(spec: dict) -> None:
    """Fill policy fields that older runtime specs omitted but current configs default."""

    policy = spec.get("policy_params")
    if not isinstance(policy, dict):
        return

    observation = spec.get("observation_params")
    if not isinstance(observation, dict):
        observation = {}

    policy.setdefault(
        "model_input_centering",
        observation.get("model_input_centering", "bbox_center"),
    )
    policy.setdefault("relative_translation_query_tokens", 2)
    policy.setdefault("reuse_pretrain_pose_cross_attn", False)


def _resolve_checkpoint_arg(runtime_spec_path: str, checkpoint: str | None) -> str:
    if checkpoint is not None:
        return checkpoint

    spec_dir = Path(runtime_spec_path).parent
    best_path = spec_dir / "model_best.pt"
    if best_path.is_file():
        return str(best_path)

    candidates = []
    for path in spec_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])

    for name in ("model.pt", "best.pt"):
        path = spec_dir / name
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        f"--checkpoint was not provided and no model_best.pt/model_*.pt/model.pt/best.pt was found in {spec_dir}"
    )


def _latest_runtime_spec_for_config(config: str) -> str:
    cfg = load_exp_cfg(config)
    rl_refs = [ref for ref in resolve_artifacts(cfg).stages if ref.stage == "rl"]
    if not rl_refs:
        raise FileNotFoundError(f"No RL artifact location could be resolved for config: {config}")
    run_root = rl_refs[0].directory.parent
    candidates = []
    if run_root.is_dir():
        for path in run_root.iterdir():
            spec_path = path / "rl_runtime_spec.json"
            manifest_path = path / "manifest.json"
            if not path.is_dir() or not spec_path.is_file() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if manifest.get("config_hash") == rl_refs[0].config_hash:
                candidates.append((path.stat().st_mtime, path.name, spec_path))
    if not candidates:
        raise FileNotFoundError(
            "No rl_runtime_spec.json matching the current RL asset/config hash "
            f"was found for {config}: {run_root}. Train the current config first, "
            "or pass an older artifact's manifest/runtime spec explicitly."
        )
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


parser = argparse.ArgumentParser(
    description=(
        "Record full failed episodes. Tool and object assets are randomly assigned to envs."
    )
)
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--config",
    type=str,
    default=None,
    help="Experiment config exposing EXP_CFG. Uses the latest RL run/checkpoint for this config.",
)
source_group.add_argument(
    "--runtime_spec",
    type=str,
    help="Path to the rl_runtime_spec.json written with the checkpoint.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the RSL-RL checkpoint. Defaults to model_best.pt beside the runtime spec.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task. Defaults to runtime_spec['task_id'].")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate per rank.")
parser.add_argument(
    "--num_failure_videos",
    type=int,
    default=32,
    help="Number of failed episode videos to keep per rank.",
)
parser.add_argument(
    "--num_success_videos",
    type=int,
    default=0,
    help="Number of successful episode videos to keep per rank.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--object_random_seed",
    type=int,
    default=None,
    help="Base seed for random tool/object assignment. Defaults to a time-derived seed.",
)
parser.add_argument(
    "--object_rerandomize_interval_steps",
    type=int,
    default=0,
    help=(
        "When positive, rebuild the Isaac env with fresh per-env random tool/object assignment "
        "after this many policy steps. Incomplete temporary videos from the previous env are discarded."
    ),
)
parser.add_argument(
    "--video_width",
    type=int,
    default=None,
    help="Video width. Defaults to 1280 for official Panda gripper and 512 for welded-tool recording.",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=None,
    help="Video height. Defaults to 720 for official Panda gripper and 512 for welded-tool recording.",
)
parser.add_argument("--video_fps", type=int, default=10, help="Output video frames per second.")
parser.add_argument(
    "--video_max_active_episodes",
    type=int,
    default=2,
    help="Maximum number of complete episode videos recorded concurrently per rank.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help=(
        "Directory to write MP4 files. Defaults to <runtime_spec_dir>/failure_videos, "
        "outcome_videos, or success_videos depending on requested outcomes."
    ),
)
parser.add_argument(
    "--disable_recording_visual_overrides",
    action="store_true",
    default=False,
    help="Disable recording-only color/material overrides for isolating renderer memory regressions.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run recording across multiple GPUs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.num_failure_videos < 0:
    parser.error("--num_failure_videos must be >= 0")
if args_cli.num_success_videos < 0:
    parser.error("--num_success_videos must be >= 0")
if args_cli.num_failure_videos == 0 and args_cli.num_success_videos == 0:
    parser.error("At least one of --num_failure_videos or --num_success_videos must be positive")
if args_cli.object_random_seed is not None and args_cli.object_random_seed < 0:
    parser.error("--object_random_seed must be >= 0")
if args_cli.object_rerandomize_interval_steps < 0:
    parser.error("--object_rerandomize_interval_steps must be >= 0")
if args_cli.video_fps <= 0:
    parser.error("--video_fps must be positive")
if args_cli.video_max_active_episodes <= 0:
    parser.error("--video_max_active_episodes must be positive")

runtime_spec_source = args_cli.runtime_spec or _latest_runtime_spec_for_config(args_cli.config)
runtime_spec_path = os.path.abspath(os.path.normpath(runtime_spec_source))
with open(runtime_spec_path, "r", encoding="utf-8") as f:
    runtime_spec = json.load(f)

requested_runtime_robot_mode = str(
    runtime_spec.get("env_params", {}).get("robot_mode", "tool")
)
if requested_runtime_robot_mode not in {
    "tool",
    "official_panda_gripper",
    "generated_gripper",
    "one_dof_gripper",
    "cross_embodiment_gripper",
}:
    parser.error(
        "record_failure_videos.py requires runtime_spec env_params.robot_mode "
        "to be 'tool', 'official_panda_gripper', 'generated_gripper', "
        "'one_dof_gripper', or 'cross_embodiment_gripper'"
    )
if args_cli.video_width is None:
    args_cli.video_width = (
        PANDA_GRIPPER_RECORD_RESOLUTION[0]
        if requested_runtime_robot_mode == "official_panda_gripper"
        else DEFAULT_TOOL_RECORD_RESOLUTION[0]
    )
if args_cli.video_height is None:
    args_cli.video_height = (
        PANDA_GRIPPER_RECORD_RESOLUTION[1]
        if requested_runtime_robot_mode == "official_panda_gripper"
        else DEFAULT_TOOL_RECORD_RESOLUTION[1]
    )
if args_cli.video_width <= 0:
    parser.error("--video_width must be positive")
if args_cli.video_height <= 0:
    parser.error("--video_height must be positive")

if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id")
if not args_cli.task:
    parser.error("--task is required when runtime_spec does not contain task_id")

paths_yaml = runtime_spec.get("paths_yaml")
if not paths_yaml:
    parser.error("runtime_spec must contain paths_yaml")
paths_yaml = os.path.abspath(os.path.normpath(paths_yaml))
checkpoint_arg = _resolve_checkpoint_arg(runtime_spec_path, args_cli.checkpoint)

rank, world_size = _distributed_rank_info(args_cli.distributed)
local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args_cli.distributed else 0
runtime_robot_mode = requested_runtime_robot_mode
if requested_runtime_robot_mode == "cross_embodiment_gripper":
    if not args_cli.distributed or world_size < 2 or world_size % 2 != 0:
        parser.error(
            "cross_embodiment_gripper recording requires --distributed with "
            "an even world size >= 2; use record.bash, which defaults CE runs to 2 GPUs"
        )
    runtime_robot_mode = (
        "generated_gripper" if rank < world_size // 2 else "one_dof_gripper"
    )
    print(
        f"[record] cross_embodiment rank={rank}/{world_size} "
        f"effective_robot_mode={runtime_robot_mode}",
        flush=True,
    )
os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = str(rank)
os.environ["TOOL_GENERALIST_LOCAL_RANK"] = str(local_rank)
os.environ["TOOL_GENERALIST_WORLD_SIZE"] = str(world_size)
os.environ["TOOL_GENERALIST_PATHS_YAML"] = paths_yaml

object_random_seed = (
    int(args_cli.object_random_seed)
    if args_cli.object_random_seed is not None
    else int(time.time() * 1000.0) % (2**31 - 1)
)
os.environ["TOOL_GENERALIST_OBJECT_ASSIGNMENT_SEED"] = str(object_random_seed)

eval_runtime_spec = copy.deepcopy(runtime_spec)
eval_runtime_spec["num_envs"] = args_cli.num_envs
if isinstance(eval_runtime_spec.get("env_params"), dict):
    eval_runtime_spec["env_params"]["num_envs"] = args_cli.num_envs
eval_runtime_spec["paths_yaml"] = paths_yaml
asset_assignment = eval_runtime_spec.get("asset_assignment_params")
if not isinstance(asset_assignment, dict):
    parser.error("runtime_spec must contain asset_assignment_params")
asset_assignment["randomize_tool_assignment"] = runtime_robot_mode in {
    "tool",
    "generated_gripper",
    "one_dof_gripper",
}
asset_assignment["randomize_object_assignment"] = True
_backfill_runtime_spec_defaults(eval_runtime_spec)
eval_runtime_spec_path = os.path.join(
    tempfile.gettempdir(),
    "tool_generalist_record_failure_videos",
    f"rl_runtime_spec_rank_{rank}_of_{world_size}_envs_{args_cli.num_envs}.json",
)
os.makedirs(os.path.dirname(eval_runtime_spec_path), exist_ok=True)
validate_runtime_spec(eval_runtime_spec, eval_runtime_spec_path)
with open(eval_runtime_spec_path, "w", encoding="utf-8") as f:
    json.dump(eval_runtime_spec, f, ensure_ascii=False, indent=2)
os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path

args_cli.enable_cameras = True
if runtime_robot_mode == "official_panda_gripper":
    args_cli.video = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.distributed as dist

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
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
    ONE_DOF_GRIPPER_ASSIGNMENT_SALT,
    OBJECT_ASSIGNMENT_SALT,
    TOOL_ASSIGNMENT_SALT,
    asset_indices_for_rank,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    OBJECT_ASSET_CFGS,
    TOOL_DATA,
    get_generated_gripper_index_for_env,
    get_object_index_for_env,
    get_one_dof_gripper_index_for_env,
    get_tool_index_for_env,
)
from scripts.video_diagnostics import (
    format_recording_diagnostics,
    overlay_recording_diagnostics,
    recording_debug_metrics,
)


def _tool_names_from_loaded_data() -> list[str]:
    if runtime_robot_mode == "official_panda_gripper":
        return ["official_panda_gripper"]
    if runtime_robot_mode == "generated_gripper":
        names = [
            str(getattr(asset, "gripper_id", f"generated_gripper_{index:04d}"))
            for index, asset in enumerate(env_tool_module.GENERATED_GRIPPER_DATA)
        ]
        if len(names) == 0:
            raise ValueError("No generated grippers were loaded into GENERATED_GRIPPER_DATA.")
        return names
    if runtime_robot_mode == "one_dof_gripper":
        names = [
            str(getattr(asset, "gripper_id", f"one_dof_gripper_{index:04d}"))
            for index, asset in enumerate(env_tool_module.ONE_DOF_GRIPPER_DATA)
        ]
        if len(names) == 0:
            raise ValueError("No one-DoF grippers were loaded into ONE_DOF_GRIPPER_DATA.")
        return names
    tool_names = [tool_data["name"] for tool_data in TOOL_DATA]
    if len(tool_names) == 0:
        raise ValueError("No tools were loaded into TOOL_DATA.")
    return tool_names


def _object_names_from_loaded_data() -> list[str]:
    object_names = []
    for index, cfg in enumerate(OBJECT_ASSET_CFGS):
        obj_path = getattr(cfg, "obj_path", None)
        usd_path = getattr(cfg, "usd_path", None)
        if obj_path:
            object_names.append(Path(str(obj_path)).stem)
        elif usd_path:
            object_names.append(Path(str(usd_path)).stem)
        else:
            object_names.append(f"object_{index:04d}")
    if len(object_names) == 0:
        raise ValueError("No objects were loaded into OBJECT_ASSET_CFGS.")
    return object_names


def _apply_asset_assignment_seed(
    seed: int,
    num_envs: int,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> tuple[list[int], list[int]]:
    if runtime_robot_mode == "tool":
        tool_indices = asset_indices_for_rank(
            int(num_envs),
            rank,
            len(env_tool_module.TOOL_DATA),
            randomize=True,
            seed=int(seed),
            salt=TOOL_ASSIGNMENT_SALT,
        )
        tool_usd_paths = [env_tool_module.TOOL_USD_PATHS[index] for index in tool_indices]
        env_tool_module.TOOL_ASSET_INDICES_BY_ENV[:] = tool_indices
        env_tool_module.TOOL_USD_PATHS_BY_ENV[:] = tool_usd_paths
        env_tool_module.TOOL_SPAWN_ASSET_INDICES[:] = tool_indices
        env_tool_module.TOOL_USD_PATHS_FOR_SPAWN[:] = tool_usd_paths
        if hasattr(env_cfg.scene, "robot"):
            env_cfg.scene.robot.spawn.usd_path = env_tool_module.TOOL_USD_PATHS_FOR_SPAWN
    elif runtime_robot_mode == "generated_gripper":
        tool_indices = asset_indices_for_rank(
            int(num_envs),
            rank,
            len(env_tool_module.GENERATED_GRIPPER_DATA),
            randomize=True,
            seed=int(seed),
            salt=GENERATED_GRIPPER_ASSIGNMENT_SALT,
        )
        gripper_usd_paths = [
            env_tool_module.GENERATED_GRIPPER_USD_PATHS[index] for index in tool_indices
        ]
        env_tool_module.GENERATED_GRIPPER_ASSET_INDICES_BY_ENV[:] = tool_indices
        env_tool_module.GENERATED_GRIPPER_USD_PATHS_BY_ENV[:] = gripper_usd_paths
        env_tool_module.GENERATED_GRIPPER_SPAWN_ASSET_INDICES[:] = tool_indices
        env_tool_module.GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN[:] = gripper_usd_paths
        if hasattr(env_cfg.scene, "robot"):
            env_cfg.scene.robot.spawn.usd_path = env_tool_module.GENERATED_GRIPPER_USD_PATHS_FOR_SPAWN
    elif runtime_robot_mode == "one_dof_gripper":
        tool_indices = asset_indices_for_rank(
            int(num_envs),
            rank,
            len(env_tool_module.ONE_DOF_GRIPPER_DATA),
            randomize=True,
            seed=int(seed),
            salt=ONE_DOF_GRIPPER_ASSIGNMENT_SALT,
        )
        gripper_usd_paths = [
            str(env_tool_module.ONE_DOF_GRIPPER_DATA[index].usd_path)
            for index in tool_indices
        ]
        env_tool_module.ONE_DOF_GRIPPER_ASSET_INDICES_BY_ENV[:] = tool_indices
        env_tool_module.ONE_DOF_GRIPPER_SPAWN_ASSET_INDICES[:] = tool_indices
        env_tool_module.ONE_DOF_GRIPPER_USD_PATHS_FOR_SPAWN[:] = gripper_usd_paths
        if hasattr(env_cfg.scene, "robot"):
            env_cfg.scene.robot.spawn.usd_path = (
                env_tool_module.ONE_DOF_GRIPPER_USD_PATHS_FOR_SPAWN
            )
    else:
        tool_indices = [0 for _ in range(int(num_envs))]

    object_indices = asset_indices_for_rank(
        int(num_envs),
        rank,
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
    return tool_indices, object_indices


def _make_record_camera_cfg() -> TiledCameraCfg:
    if runtime_robot_mode == "official_panda_gripper":
        camera_pos = PANDA_GRIPPER_RECORD_CAMERA_EYE
        camera_rot = PANDA_GRIPPER_RECORD_CAMERA_ROT_ROS
    else:
        camera_pos = (1.25, 0.0, 0.85)
        camera_rot = (-0.3337, 0.6234, 0.6234, -0.3337)

    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/EvalRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_pos,
            rot=camera_rot,
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


def _apply_panda_gripper_viewer_camera(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if runtime_robot_mode != "official_panda_gripper":
        return
    viewer = getattr(env_cfg, "viewer", None)
    if viewer is None:
        return
    viewer.eye = PANDA_GRIPPER_RECORD_CAMERA_EYE
    viewer.lookat = PANDA_GRIPPER_RECORD_CAMERA_LOOKAT
    viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
    viewer.origin_type = "env"
    viewer.env_index = 0


def _default_video_subdir() -> str:
    if args_cli.num_success_videos > 0 and args_cli.num_failure_videos == 0:
        return "success_videos"
    if args_cli.num_success_videos > 0:
        return "outcome_videos"
    return "failure_videos"


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


def _apply_recording_visual_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Use a minimal deterministic scene for recording without changing physics."""

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


def _close_ffmpeg_writer(writer: subprocess.Popen) -> None:
    if writer.stdin is not None:
        writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def _init_video_state(video_dir: str) -> dict:
    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG_PATH}")
    os.makedirs(video_dir, exist_ok=True)
    return {
        "video_dir": video_dir,
        "active": {},
        "waiting": {},
        "next_id": 0,
        "next_env_cursor": 0,
        "failure_saved": 0,
        "success_saved": 0,
        "kept_paths": [],
        "current_step": None,
    }


def _video_quota_remaining(video_state: dict) -> bool:
    return (
        int(video_state["failure_saved"]) < args_cli.num_failure_videos
        or int(video_state["success_saved"]) < args_cli.num_success_videos
    )


def _outcome_quota_remaining(video_state: dict, episode_success: bool) -> bool:
    if episode_success:
        return int(video_state["success_saved"]) < args_cli.num_success_videos
    return int(video_state["failure_saved"]) < args_cli.num_failure_videos


def _video_slots_used(video_state: dict) -> int:
    return len(video_state["active"]) + len(video_state["waiting"])


def _output_gate_metrics_for_env(policy, env_id: int) -> dict | None:
    gate = getattr(policy, "_last_actor_gate", None) if policy is not None else None
    if gate is None:
        return None
    if int(gate.shape[0]) <= int(env_id):
        return None
    gate_row = gate[int(env_id)].detach().float().reshape(-1).cpu()
    if gate_row.numel() == 0:
        return None
    expert_a = float(gate_row.mean().item())
    expert_a_min = float(gate_row.min().item())
    expert_a_max = float(gate_row.max().item())
    expert_b = 1.0 - expert_a
    return {
        "output_gate_expert_a_weight": expert_a,
        "output_gate_expert_b_weight": expert_b,
        "output_gate_expert_a_min": expert_a_min,
        "output_gate_expert_a_max": expert_a_max,
        "output_gate_selected_expert": "model_a" if expert_a >= 0.5 else "model_b",
    }


def _make_episode_video_record(
    video_state: dict,
    env_id: int,
    tool_name: str,
    object_name: str,
) -> dict:
    record_id = int(video_state["next_id"])
    video_state["next_id"] = record_id + 1
    tmp_path = os.path.join(
        video_state["video_dir"],
        (
            f"rank_{rank:03d}_pending_{record_id:06d}_"
            f"{_safe_filename(tool_name)}__{_safe_filename(object_name)}.tmp.mp4"
        ),
    )
    return {
        "record_id": record_id,
        "env_id": int(env_id),
        "tool_name": tool_name,
        "object_name": object_name,
        "tmp_path": tmp_path,
        "writer": None,
        "frames": 0,
    }


def _start_episode_video(
    video_state: dict,
    env_id: int,
    env_to_tool_idx: torch.Tensor,
    env_to_object_idx: torch.Tensor,
    tool_names: list[str],
    object_names: list[str],
) -> None:
    tool_name = tool_names[int(env_to_tool_idx[env_id].item())]
    object_name = object_names[int(env_to_object_idx[env_id].item())]
    record = _make_episode_video_record(video_state, env_id, tool_name, object_name)
    record["writer"] = _start_ffmpeg_writer(record["tmp_path"])
    video_state["active"][int(env_id)] = record


def _queue_episode_video(
    video_state: dict,
    env_id: int,
    env_to_tool_idx: torch.Tensor,
    env_to_object_idx: torch.Tensor,
    tool_names: list[str],
    object_names: list[str],
) -> None:
    tool_name = tool_names[int(env_to_tool_idx[env_id].item())]
    object_name = object_names[int(env_to_object_idx[env_id].item())]
    video_state["waiting"][int(env_id)] = _make_episode_video_record(video_state, env_id, tool_name, object_name)


def _start_waiting_videos_on_episode_start(video_state: dict, episode_start_env_ids: set[int]) -> set[int]:
    started_env_ids = set()
    for env_id in sorted(episode_start_env_ids):
        record = video_state["waiting"].pop(int(env_id), None)
        if record is None:
            continue
        if not _video_quota_remaining(video_state):
            _discard_video_tmp(record["tmp_path"])
            continue
        record["writer"] = _start_ffmpeg_writer(record["tmp_path"])
        record["frames"] = 0
        video_state["active"][int(env_id)] = record
        started_env_ids.add(int(env_id))
    return started_env_ids


def _activate_video_slots(
    video_state: dict,
    num_envs: int,
    env_to_tool_idx: torch.Tensor,
    env_to_object_idx: torch.Tensor,
    tool_names: list[str],
    object_names: list[str],
    episode_start_env_ids: set[int],
) -> set[int]:
    started_env_ids = set()
    if not _video_quota_remaining(video_state):
        return started_env_ids
    cursor = int(video_state["next_env_cursor"])
    attempts = 0
    while (
        _video_slots_used(video_state) < args_cli.video_max_active_episodes
        and _video_quota_remaining(video_state)
        and attempts < num_envs
    ):
        env_id = cursor % num_envs
        cursor += 1
        attempts += 1
        if env_id in video_state["active"] or env_id in video_state["waiting"]:
            continue
        if env_id in episode_start_env_ids:
            _start_episode_video(video_state, env_id, env_to_tool_idx, env_to_object_idx, tool_names, object_names)
            started_env_ids.add(int(env_id))
        else:
            _queue_episode_video(video_state, env_id, env_to_tool_idx, env_to_object_idx, tool_names, object_names)
    video_state["next_env_cursor"] = cursor
    return started_env_ids


def _capture_video_frames(env, video_state: dict, env_ids: set[int] | None = None, policy=None) -> None:
    if len(video_state["active"]) == 0:
        return
    if runtime_robot_mode == "official_panda_gripper":
        _capture_viewport_video_frames(env, video_state, env_ids=env_ids, policy=policy)
        return
    env.unwrapped.sim.render()
    env.unwrapped.scene["eval_record_camera"].update(dt=0.0, force_recompute=True)
    rgb_all = env.unwrapped.scene["eval_record_camera"].data.output["rgb"]
    for record in list(video_state["active"].values()):
        env_id = int(record["env_id"])
        if env_ids is not None and env_id not in env_ids:
            continue
        writer = record.get("writer")
        if writer is None:
            continue
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
        step = video_state.get("current_step")
        record["last_diag"] = format_recording_diagnostics(metrics, step=step)
        frame = overlay_recording_diagnostics(frame, metrics, step=step)
        writer.stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1


def _capture_viewport_video_frames(env, video_state: dict, env_ids: set[int] | None = None, policy=None) -> None:
    base_env = env.unwrapped
    camera_controller = getattr(base_env, "viewport_camera_controller", None)
    if camera_controller is None:
        raise RuntimeError(
            "Official Panda gripper outcome recording requires the IsaacLab viewport camera controller. "
            "Run with cameras/video enabled."
        )
    for record in list(video_state["active"].values()):
        env_id = int(record["env_id"])
        if env_ids is not None and env_id not in env_ids:
            continue
        writer = record.get("writer")
        if writer is None:
            continue

        camera_controller.set_view_env_index(env_id)
        camera_controller.update_view_to_env()
        frame = base_env.render()
        if frame is None:
            raise RuntimeError("Official Panda gripper viewport recording expected env.render() to return RGB data.")
        frame = frame[:, :, :3].copy()

        metrics = recording_debug_metrics(
            base_env,
            env_id,
            runtime_spec.get("reward_params", {}),
        )
        gate_metrics = _output_gate_metrics_for_env(policy, env_id)
        if gate_metrics is not None:
            metrics.update(gate_metrics)
        step = video_state.get("current_step")
        record["last_diag"] = format_recording_diagnostics(metrics, step=step)
        frame = overlay_recording_diagnostics(frame, metrics, step=step)
        writer.stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1


def _active_diag_summary(video_state: dict) -> str:
    for record in video_state["active"].values():
        diag = record.get("last_diag")
        if diag:
            return str(diag)
    return "pending"


def _discard_video_tmp(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _finish_video_on_done(video_state: dict, env_id: int, episode_success: bool) -> str | None:
    record = video_state["active"].pop(int(env_id), None)
    if record is None:
        return None
    writer = record.get("writer")
    if writer is not None:
        _close_ffmpeg_writer(writer)
        record["writer"] = None

    if not _outcome_quota_remaining(video_state, episode_success):
        _discard_video_tmp(record["tmp_path"])
        return None

    outcome = "success" if episode_success else "failure"
    counter_key = "success_saved" if episode_success else "failure_saved"
    slot = int(video_state[counter_key])
    video_state[counter_key] = slot + 1
    final_path = os.path.join(
        video_state["video_dir"],
        (
            f"rank_{rank:03d}_{outcome}_{slot:03d}_"
            f"{_safe_filename(record['tool_name'])}__{_safe_filename(record['object_name'])}.mp4"
        ),
    )
    os.replace(record["tmp_path"], final_path)
    video_state["kept_paths"].append(final_path)
    return final_path


def _close_video_state(video_state: dict) -> None:
    for record in list(video_state["active"].values()):
        writer = record.get("writer")
        if writer is not None:
            _close_ffmpeg_writer(writer)
        _discard_video_tmp(record["tmp_path"])
    for record in list(video_state["waiting"].values()):
        _discard_video_tmp(record["tmp_path"])
    video_state["active"].clear()
    video_state["waiting"].clear()


def _write_summary(
    video_dir: str,
    resume_path: str,
    video_state: dict,
    *,
    total_episodes: int,
    failures_seen: int,
    successes_seen: int,
    asset_assignment_cycles: list[dict],
) -> None:
    summary_path = os.path.join(video_dir, f"record_failure_videos_rank_{rank}.json")
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "paths_yaml": paths_yaml,
        "rank": rank,
        "world_size": world_size,
        "num_envs_per_rank": args_cli.num_envs,
        "object_random_seed": object_random_seed,
        "object_rerandomize_interval_steps": args_cli.object_rerandomize_interval_steps,
        "asset_assignment_cycles": asset_assignment_cycles,
        "requested_failure_videos": args_cli.num_failure_videos,
        "requested_success_videos": args_cli.num_success_videos,
        "failure_videos_saved": int(video_state["failure_saved"]),
        "success_videos_saved": int(video_state["success_saved"]),
        "episodes_seen": total_episodes,
        "failures_seen": failures_seen,
        "successes_seen": successes_seen,
        "videos": list(video_state["kept_paths"]),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO][rank {rank}]: Saved summary: {summary_path}", flush=True)


def _distributed_barrier(label: str) -> None:
    if not args_cli.distributed:
        return
    if not dist.is_initialized():
        raise RuntimeError(f"Distributed barrier requested before torch.distributed init: {label}")
    print(
        f"[INFO][rank {rank}]: Waiting at distributed barrier: {label}; "
        f"saved_failure_videos={video_state_for_logging['failure_saved']}/"
        f"{args_cli.num_failure_videos} saved_success_videos="
        f"{video_state_for_logging['success_saved']}/{args_cli.num_success_videos}",
        flush=True,
    )
    dist.barrier()
    print(f"[INFO][rank {rank}]: Passed distributed barrier: {label}", flush=True)


def _any_rank_needs_videos(video_state: dict) -> bool:
    local_remaining = 1 if _video_quota_remaining(video_state) else 0
    if not args_cli.distributed:
        return bool(local_remaining)
    if not dist.is_initialized():
        return bool(local_remaining)
    device = torch.device(f"cuda:{app_launcher.local_rank}" if torch.cuda.is_available() else "cpu")
    remaining = torch.tensor([local_remaining], device=device, dtype=torch.int32)
    dist.all_reduce(remaining, op=dist.ReduceOp.SUM)
    return int(remaining.item()) > 0


video_state_for_logging = {"failure_saved": 0, "success_saved": 0}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        if agent_cfg.seed is not None:
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

    resume_path = retrieve_file_path(checkpoint_arg)
    runtime_spec_dir = os.path.dirname(runtime_spec_path)
    video_dir = (
        args_cli.video_dir
        if args_cli.video_dir is not None
        else os.path.join(runtime_spec_dir, _default_video_subdir())
    )
    _apply_panda_gripper_viewer_camera(env_cfg)
    _disable_debug_pointcloud_rendering(env_cfg)
    if runtime_robot_mode in {"tool", "generated_gripper", "one_dof_gripper"}:
        env_cfg.scene.eval_record_camera = _make_record_camera_cfg()
    if not args_cli.disable_recording_visual_overrides:
        _apply_recording_visual_overrides(env_cfg)

    tool_names = _tool_names_from_loaded_data()
    object_names = _object_names_from_loaded_data()
    if args_cli.num_envs < len(tool_names):
        print(
            f"[WARNING][rank {rank}]: num_envs={args_cli.num_envs} is smaller than "
            f"loaded tools={len(tool_names)}; this rank can only record tools assigned to active envs.",
            flush=True,
        )

    def _create_env_and_policy(cycle_index: int, cycle_seed: int):
        tool_indices, object_indices = _apply_asset_assignment_seed(cycle_seed, args_cli.num_envs, env_cfg)
        render_mode = "rgb_array" if runtime_robot_mode == "official_panda_gripper" else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO][rank {rank}]: Loading model checkpoint from: {resume_path}", flush=True)
        print(
            f"[INFO][rank {rank}]: Asset assignment cycle={cycle_index} "
            f"seed={cycle_seed} interval_steps={args_cli.object_rerandomize_interval_steps} "
            f"saved_failure_videos={video_state['failure_saved']}/{args_cli.num_failure_videos} "
            f"saved_success_videos={video_state['success_saved']}/{args_cli.num_success_videos}",
            flush=True,
        )
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(
            f"[INFO][rank {rank}]: Inference runner initialized; loading policy weights "
            "(optimizer state skipped).",
            flush=True,
        )
        ppo_runner.load(resume_path, load_optimizer=False)
        print(f"[INFO][rank {rank}]: Policy checkpoint loaded.", flush=True)
        inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)
        policy_module = ppo_runner.alg.policy

        if not hasattr(env.unwrapped, "episode_success_buf"):
            raise AttributeError("Environment does not have episode_success_buf.")

        num_envs = env.unwrapped.num_envs
        if runtime_robot_mode == "tool":
            env_to_tool_idx = torch.tensor(
                [get_tool_index_for_env(env_id) for env_id in range(num_envs)],
                dtype=torch.long,
            )
        elif runtime_robot_mode == "generated_gripper":
            env_to_tool_idx = torch.tensor(
                [get_generated_gripper_index_for_env(env_id) for env_id in range(num_envs)],
                dtype=torch.long,
            )
        elif runtime_robot_mode == "one_dof_gripper":
            env_to_tool_idx = torch.tensor(
                [get_one_dof_gripper_index_for_env(env_id) for env_id in range(num_envs)],
                dtype=torch.long,
            )
        else:
            env_to_tool_idx = torch.zeros(num_envs, dtype=torch.long)
        env_to_object_idx = torch.tensor(
            [get_object_index_for_env(env_id) for env_id in range(num_envs)],
            dtype=torch.long,
        )
        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
        return env, inference_policy, policy_module, obs, dt, env_to_tool_idx, env_to_object_idx, tool_indices, object_indices

    video_state = _init_video_state(video_dir)
    video_state_for_logging["failure_saved"] = int(video_state["failure_saved"])
    video_state_for_logging["success_saved"] = int(video_state["success_saved"])
    print(f"[INFO][rank {rank}]: Recording outcome videos to: {video_dir}", flush=True)

    total_episodes = 0
    failures_seen = 0
    successes_seen = 0
    asset_assignment_cycles = []
    env = None
    try:
        cycle_index = 0
        while _any_rank_needs_videos(video_state) and simulation_app.is_running():
            cycle_seed = object_random_seed + cycle_index
            (
                env,
                inference_policy,
                policy_module,
                obs,
                dt,
                env_to_tool_idx,
                env_to_object_idx,
                tool_indices,
                object_indices,
            ) = _create_env_and_policy(
                cycle_index,
                cycle_seed,
            )
            asset_assignment_cycles.append(
                {
                    "cycle": cycle_index,
                    "seed": cycle_seed,
                    "tool_indices_by_env": list(tool_indices),
                    "object_indices_by_env": list(object_indices),
                }
            )
            num_envs = env.unwrapped.num_envs
            cycle_steps = 0
            started_env_ids = _activate_video_slots(
                video_state,
                num_envs,
                env_to_tool_idx,
                env_to_object_idx,
                tool_names,
                object_names,
                set(range(num_envs)),
            )
            _capture_video_frames(env, video_state, started_env_ids)

            while _any_rank_needs_videos(video_state) and simulation_app.is_running():
                start_time = time.time()
                video_state["current_step"] = cycle_steps + 1
                with torch.inference_mode():
                    actions = inference_policy(obs)
                    obs, _, dones, _ = env.step(actions)
                cycle_steps += 1

                _capture_video_frames(env, video_state, policy=policy_module)

                ended = dones.bool()
                if torch.any(ended):
                    ended_ids = torch.where(ended)[0]
                    ended_env_ids = set(int(env_id) for env_id in ended_ids.tolist())
                    for env_id in ended_ids.tolist():
                        if not hasattr(env.unwrapped, "_episode_success_before_reset"):
                            raise AttributeError("Environment does not have _episode_success_before_reset.")
                        episode_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                        total_episodes += 1
                        if episode_success:
                            successes_seen += 1
                        else:
                            failures_seen += 1
                        saved_path = _finish_video_on_done(video_state, env_id, episode_success)
                        video_state_for_logging["failure_saved"] = int(video_state["failure_saved"])
                        video_state_for_logging["success_saved"] = int(video_state["success_saved"])
                        if saved_path is not None:
                            print(
                                f"[PROGRESS][rank {rank}] saved_video={saved_path} "
                                f"saved_failure_videos={video_state['failure_saved']}/{args_cli.num_failure_videos} "
                                f"saved_success_videos={video_state['success_saved']}/{args_cli.num_success_videos} "
                                f"cycle={cycle_index} cycle_step={cycle_steps} "
                                f"episodes_seen={total_episodes} failures_seen={failures_seen} "
                                f"successes_seen={successes_seen} active_recordings={len(video_state['active'])} ",
                                flush=True,
                            )

                    started_env_ids = _start_waiting_videos_on_episode_start(video_state, ended_env_ids)
                    started_env_ids.update(
                        _activate_video_slots(
                            video_state,
                            num_envs,
                            env_to_tool_idx,
                            env_to_object_idx,
                            tool_names,
                            object_names,
                            ended_env_ids,
                        )
                    )
                    if started_env_ids:
                        _capture_video_frames(env, video_state, started_env_ids)

                if args_cli.real_time and dt is not None:
                    elapsed = time.time() - start_time
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    elapsed = time.time() - start_time

                print(
                    f"[PROGRESS][rank {rank}] step cycle={cycle_index} cycle_step={cycle_steps} "
                    f"saved_failure_videos={video_state['failure_saved']}/{args_cli.num_failure_videos} "
                    f"saved_success_videos={video_state['success_saved']}/{args_cli.num_success_videos} "
                    f"episodes_seen={total_episodes} failures_seen={failures_seen} "
                    f"successes_seen={successes_seen} active_recordings={len(video_state['active'])} "
                    f"waiting_recordings={len(video_state['waiting'])} "
                    f"diag={_active_diag_summary(video_state)} "
                    f"step_time={elapsed:.4f}s",
                    flush=True,
                )

                if (
                    args_cli.object_rerandomize_interval_steps > 0
                    and cycle_steps >= args_cli.object_rerandomize_interval_steps
                    and _any_rank_needs_videos(video_state)
                ):
                    print(
                        f"[INFO][rank {rank}]: Reached {cycle_steps} steps in asset assignment "
                        f"cycle {cycle_index}; saved_failure_videos="
                        f"{video_state['failure_saved']}/{args_cli.num_failure_videos}; "
                        f"saved_success_videos={video_state['success_saved']}/{args_cli.num_success_videos}; "
                        f"waiting for all ranks before rebuilding env with new random tools/objects.",
                        flush=True,
                    )
                    break

            if (
                args_cli.object_rerandomize_interval_steps > 0
                and _any_rank_needs_videos(video_state)
                and simulation_app.is_running()
            ):
                _distributed_barrier(f"before_object_rebuild_cycle_{cycle_index}")
            _close_video_state(video_state)
            env.close()
            env = None
            if (
                args_cli.object_rerandomize_interval_steps > 0
                and _any_rank_needs_videos(video_state)
                and simulation_app.is_running()
            ):
                _distributed_barrier(f"after_object_rebuild_cycle_{cycle_index}")
            cycle_index += 1
    finally:
        _close_video_state(video_state)
        if env is not None:
            env.close()

    _write_summary(
        video_dir,
        resume_path,
        video_state,
        total_episodes=total_episodes,
        failures_seen=failures_seen,
        successes_seen=successes_seen,
        asset_assignment_cycles=asset_assignment_cycles,
    )
    print(
        f"[INFO][rank {rank}]: Saved {video_state['failure_saved']}/"
        f"{args_cli.num_failure_videos} failure videos and {video_state['success_saved']}/"
        f"{args_cli.num_success_videos} success videos after {total_episodes} episodes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    if args_cli.distributed and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

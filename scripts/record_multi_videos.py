# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record completed policy episodes during a fixed-step rollout window."""

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
        f"--checkpoint was not provided and no model_*.pt/model.pt/best.pt was found in {spec_dir}"
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
            if path.is_dir() and spec_path.is_file():
                candidates.append((path.stat().st_mtime, path.name, spec_path))
    if not candidates:
        raise FileNotFoundError(
            f"No rl_runtime_spec.json found under latest-run root for config {config}: {run_root}"
        )
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


parser = argparse.ArgumentParser(
    description=(
        "Run every env for a fixed number of policy steps and save every completed episode video."
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
    help="Path to the RSL-RL checkpoint. Defaults to the latest model_*.pt beside the runtime spec.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task. Defaults to runtime_spec['task_id'].")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate per rank.")
parser.add_argument(
    "--num_steps",
    type=int,
    required=True,
    help="Number of policy steps to run in every environment. Completed episodes during this window are saved.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--object_random_seed",
    type=int,
    default=None,
    help="Base seed for random tool/object assignment. Defaults to a time-derived seed.",
)
parser.add_argument("--video_width", type=int, default=512, help="Per-env tiled-camera video width.")
parser.add_argument("--video_height", type=int, default=512, help="Per-env tiled-camera video height.")
parser.add_argument("--video_fps", type=int, default=10, help="Output video frames per second.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Directory to write MP4 files. Defaults to <runtime_spec_dir>/multi_videos.",
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
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be positive")
if args_cli.object_random_seed is not None and args_cli.object_random_seed < 0:
    parser.error("--object_random_seed must be >= 0")
if args_cli.video_width <= 0:
    parser.error("--video_width must be positive")
if args_cli.video_height <= 0:
    parser.error("--video_height must be positive")
if args_cli.video_fps <= 0:
    parser.error("--video_fps must be positive")

runtime_spec_source = args_cli.runtime_spec or _latest_runtime_spec_for_config(args_cli.config)
runtime_spec_path = os.path.abspath(os.path.normpath(runtime_spec_source))
with open(runtime_spec_path, "r", encoding="utf-8") as f:
    runtime_spec = json.load(f)

runtime_robot_mode = str(runtime_spec.get("env_params", {}).get("robot_mode", "tool"))
if runtime_robot_mode != "tool":
    parser.error("record_multi_videos.py requires runtime_spec env_params.robot_mode='tool'")

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
asset_assignment["randomize_tool_assignment"] = True
asset_assignment["randomize_object_assignment"] = True
_backfill_runtime_spec_defaults(eval_runtime_spec)
eval_runtime_spec_path = os.path.join(
    tempfile.gettempdir(),
    "tool_generalist_record_multi_videos",
    f"rl_runtime_spec_rank_{rank}_of_{world_size}_envs_{args_cli.num_envs}.json",
)
os.makedirs(os.path.dirname(eval_runtime_spec_path), exist_ok=True)
validate_runtime_spec(eval_runtime_spec, eval_runtime_spec_path)
with open(eval_runtime_spec_path, "w", encoding="utf-8") as f:
    json.dump(eval_runtime_spec, f, ensure_ascii=False, indent=2)
os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path

args_cli.enable_cameras = True
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
    OBJECT_ASSIGNMENT_SALT,
    TOOL_ASSIGNMENT_SALT,
    asset_indices_for_rank,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    OBJECT_ASSET_CFGS,
    TOOL_DATA,
    get_object_index_for_env,
    get_tool_index_for_env,
)


def _tool_names_from_loaded_data() -> list[str]:
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
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/EvalRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, -1.45, 0.85),
            rot=(0.5557194, -0.8313699, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.55,
            horizontal_aperture=28.0,
            clipping_range=(0.05, 20.0),
        ),
        width=args_cli.video_width,
        height=args_cli.video_height,
    )


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
    """Improve contrast for recording without changing training configs."""

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return

    if hasattr(scene, "table") and getattr(scene, "table") is not None:
        table_spawn = getattr(scene.table, "spawn", None)
        if table_spawn is not None and hasattr(table_spawn, "visual_material"):
            table_spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.13, 0.15))

    if hasattr(scene, "object") and getattr(scene, "object") is not None:
        object_spawn = getattr(scene.object, "spawn", None)
        assets_cfg = getattr(object_spawn, "assets_cfg", None)
        if assets_cfg is not None:
            for asset_cfg in assets_cfg:
                if hasattr(asset_cfg, "visual_material"):
                    asset_cfg.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.42, 0.95))

    for robot_name, color in (
        ("robot", (0.95, 0.80, 0.35)),
        ("robot_1", (0.95, 0.78, 0.28)),
        ("robot_2", (0.35, 0.90, 0.65)),
    ):
        robot = getattr(scene, robot_name, None)
        spawn = getattr(robot, "spawn", None)
        if spawn is not None and hasattr(spawn, "visual_material"):
            spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)


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
        "next_id": 0,
        "failure_saved": 0,
        "success_saved": 0,
        "kept_paths": [],
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


def _start_episode_videos(
    video_state: dict,
    env_ids: set[int],
    env_to_tool_idx: torch.Tensor,
    env_to_object_idx: torch.Tensor,
    tool_names: list[str],
    object_names: list[str],
) -> set[int]:
    started_env_ids = set()
    for env_id in sorted(env_ids):
        if env_id in video_state["active"]:
            continue
        _start_episode_video(video_state, env_id, env_to_tool_idx, env_to_object_idx, tool_names, object_names)
        started_env_ids.add(int(env_id))
    return started_env_ids


def _capture_video_frames(env, video_state: dict, env_ids: set[int] | None = None) -> None:
    if len(video_state["active"]) == 0:
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
        frame = frame_tensor.contiguous().numpy()
        writer.stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1


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
    video_state["active"].clear()


def _write_summary(
    video_dir: str,
    resume_path: str,
    video_state: dict,
    *,
    total_episodes: int,
    failures_seen: int,
    successes_seen: int,
    asset_assignment: dict,
    steps_completed: int,
) -> None:
    summary_path = os.path.join(video_dir, f"record_multi_videos_rank_{rank}.json")
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "paths_yaml": paths_yaml,
        "rank": rank,
        "world_size": world_size,
        "num_envs_per_rank": args_cli.num_envs,
        "num_steps": args_cli.num_steps,
        "steps_completed": steps_completed,
        "object_random_seed": object_random_seed,
        "asset_assignment": asset_assignment,
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
        else os.path.join(runtime_spec_dir, "multi_videos")
    )
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

    def _create_env_and_policy(asset_seed: int):
        tool_indices, object_indices = _apply_asset_assignment_seed(asset_seed, args_cli.num_envs, env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO][rank {rank}]: Loading model checkpoint from: {resume_path}", flush=True)
        print(
            f"[INFO][rank {rank}]: Asset assignment seed={asset_seed} "
            f"num_steps={args_cli.num_steps}",
            flush=True,
        )
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)

        if not hasattr(env.unwrapped, "episode_success_buf"):
            raise AttributeError("Environment does not have episode_success_buf.")

        num_envs = env.unwrapped.num_envs
        env_to_tool_idx = torch.tensor(
            [get_tool_index_for_env(env_id) for env_id in range(num_envs)],
            dtype=torch.long,
        )
        env_to_object_idx = torch.tensor(
            [get_object_index_for_env(env_id) for env_id in range(num_envs)],
            dtype=torch.long,
        )
        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
        return env, inference_policy, obs, dt, env_to_tool_idx, env_to_object_idx, tool_indices, object_indices

    video_state = _init_video_state(video_dir)
    print(f"[INFO][rank {rank}]: Recording outcome videos to: {video_dir}", flush=True)

    total_episodes = 0
    failures_seen = 0
    successes_seen = 0
    asset_assignment = {}
    env = None
    try:
        (
            env,
            inference_policy,
            obs,
            dt,
            env_to_tool_idx,
            env_to_object_idx,
            tool_indices,
            object_indices,
        ) = _create_env_and_policy(object_random_seed)
        asset_assignment = {
            "seed": object_random_seed,
            "tool_indices_by_env": list(tool_indices),
            "object_indices_by_env": list(object_indices),
        }
        num_envs = env.unwrapped.num_envs
        steps = 0
        started_env_ids = _start_episode_videos(
            video_state,
            set(range(num_envs)),
            env_to_tool_idx,
            env_to_object_idx,
            tool_names,
            object_names,
        )
        _capture_video_frames(env, video_state, started_env_ids)

        while steps < args_cli.num_steps and simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = inference_policy(obs)
                obs, _, dones, _ = env.step(actions)
            steps += 1

            _capture_video_frames(env, video_state)

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
                    if saved_path is not None:
                        print(
                            f"[PROGRESS][rank {rank}] saved_video={saved_path} "
                            f"saved_failure_videos={video_state['failure_saved']} "
                            f"saved_success_videos={video_state['success_saved']} "
                            f"step={steps}/{args_cli.num_steps} episodes_seen={total_episodes} failures_seen={failures_seen} "
                            f"successes_seen={successes_seen} active_recordings={len(video_state['active'])} ",
                            flush=True,
                        )

                if steps < args_cli.num_steps:
                    started_env_ids = _start_episode_videos(
                        video_state,
                        ended_env_ids,
                        env_to_tool_idx,
                        env_to_object_idx,
                        tool_names,
                        object_names,
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
                f"[PROGRESS][rank {rank}] step={steps}/{args_cli.num_steps} "
                f"saved_failure_videos={video_state['failure_saved']} "
                f"saved_success_videos={video_state['success_saved']} "
                f"episodes_seen={total_episodes} failures_seen={failures_seen} "
                f"successes_seen={successes_seen} active_recordings={len(video_state['active'])} "
                f"step_time={elapsed:.4f}s",
                flush=True,
            )
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
        asset_assignment=asset_assignment,
        steps_completed=steps,
    )
    print(
        f"[INFO][rank {rank}]: Saved {video_state['failure_saved']} failure videos "
        f"and {video_state['success_saved']} success videos after {steps} steps "
        f"and {total_episodes} completed episodes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    if args_cli.distributed and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

#!/usr/bin/env python3
"""Visualize the official Panda-gripper env with random actions.

This script is intended for machines with Isaac/IsaacLab runtime available. It
materializes the normal config-derived runtime spec, launches ``panda-gripper-v0``,
and enables the gripper debug visualizers already wired into the environment:

- red fingertip-midpoint/tool-center marker,
- blue policy gripper point cloud after openness-bucket selection,
- periodic bucket/opening logs for env 0.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from scripts.train import build_rl_runtime_spec
from utils.artifacts.manifest import manifest_is_complete
from utils.artifacts.resolver import resolve_artifacts
from utils.config.loader import load_exp_cfg
from utils.config.paths import load_project_paths
from utils.experiment.effective_paths import apply_experiment_path_overrides
from utils.experiment.rl_runtime_spec import RUNTIME_SPEC_ENV_VAR, validate_runtime_spec
from utils.experiment.runner import (
    _resolve_initial_encoder_checkpoint,
    _resolve_stage_encoder_checkpoint_from_manifest,
)


DEFAULT_CONFIG = "configs/experiments/panda_gripper_diff_post.py"
DEFAULT_TASK = "panda-gripper-v0"
UNUSED_ENCODER_CHECKPOINT = "/tmp/tool_generalist_unused_random_visualization_encoder.pt"
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
GRIPPER_OPEN_JOINT_POS = 0.04
GRIPPER_NUM_BUCKETS = 64


def _resolve_encoder_checkpoint_for_config(cfg, config: str) -> str | None:
    resolved = _resolve_initial_encoder_checkpoint(cfg, config_source=config)
    if resolved:
        return resolved

    for ref in resolve_artifacts(cfg).stages:
        if ref.stage != "pretrain":
            continue
        if manifest_is_complete(ref.manifest_path):
            resolved = _resolve_stage_encoder_checkpoint_from_manifest(None, ref)
            if resolved:
                return resolved
        best_checkpoint = ref.directory / "best.pt"
        if best_checkpoint.exists():
            return str(best_checkpoint)
    return None


def _build_runtime_spec_from_config(
    config: str,
    *,
    num_envs: int,
    seed: int | None,
    encoder_checkpoint: str | None,
) -> dict:
    cfg = load_exp_cfg(config)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifact_dir = Path(tempfile.gettempdir()) / "tool_generalist_panda_gripper_random" / Path(config).stem

    checkpoint = (
        encoder_checkpoint
        or _resolve_encoder_checkpoint_for_config(cfg, config)
        or UNUSED_ENCODER_CHECKPOINT
    )
    spec = build_rl_runtime_spec(
        cfg,
        paths,
        artifact_dir,
        mode="visualize_panda_gripper_random",
        encoder_checkpoint_override=checkpoint,
    )
    payload = asdict(spec)
    payload["source_config"] = str(config)
    return _runtime_spec_for_visualization(payload, num_envs=num_envs, seed=seed)


def _load_runtime_spec(path: str, *, num_envs: int, seed: int | None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime spec must be a JSON object: {path}")
    return _runtime_spec_for_visualization(payload, num_envs=num_envs, seed=seed)


def _runtime_spec_for_visualization(spec: dict, *, num_envs: int, seed: int | None) -> dict:
    spec = copy.deepcopy(spec)
    spec["mode"] = "visualize_panda_gripper_random"
    spec["num_envs"] = int(num_envs)
    if seed is not None:
        spec["seed"] = int(seed)

    env_params = spec.setdefault("env_params", {})
    env_params["num_envs"] = int(num_envs)
    env_params["robot_mode"] = "official_panda_gripper"

    spec["task_id"] = spec.get("task_id") or DEFAULT_TASK
    observation = spec.setdefault("observation_params", {})
    observation["include_tool_cloud"] = True
    observation["tool_cloud_source"] = "official_panda_gripper_kinematic_mesh"
    return spec


def _materialize_runtime_spec(spec: dict) -> str:
    spec_dir = Path(tempfile.gettempdir()) / "tool_generalist_panda_gripper_random"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"rl_runtime_spec_panda_gripper_random_envs_{spec['num_envs']}.json"
    validate_runtime_spec(spec, spec_path)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return str(spec_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run panda-gripper-v0 with random actions and debug visualizers."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Experiment config exposing EXP_CFG. Defaults to {DEFAULT_CONFIG}.",
    )
    source_group.add_argument("--runtime_spec", type=str, help="Existing rl_runtime_spec.json.")
    parser.add_argument("--task", type=str, default=None, help="Task name. Defaults to runtime spec task_id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument("--num_steps", type=int, default=600, help="Number of policy steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for actions and env spec.")
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional encoder checkpoint for runtime-spec validation. The random agent does not load a policy; "
            "a placeholder is used if no checkpoint can be resolved."
        ),
    )
    parser.add_argument(
        "--action_mode",
        choices=("random", "zero"),
        default="random",
        help="Arm action source.",
    )
    parser.add_argument(
        "--gripper_action_mode",
        choices=("random", "sweep", "open", "closed"),
        default="random",
        help="How to drive action index 7. Use sweep for visual bucket/cloud inspection.",
    )
    parser.add_argument("--print_every", type=int, default=10, help="Print gripper debug every N steps.")
    parser.add_argument("--video", action="store_true", default=False, help="Record an RGB video.")
    parser.add_argument("--video_dir", type=str, default="videos/panda_gripper_random", help="Video output directory.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args_cli.num_steps <= 0:
        parser.error("--num_steps must be positive")
    if args_cli.print_every <= 0:
        parser.error("--print_every must be positive")
    if args_cli.config is None and args_cli.runtime_spec is None:
        args_cli.config = DEFAULT_CONFIG
    return args_cli, hydra_args


args_cli, hydra_args = _parse_args()

if args_cli.config:
    runtime_spec = _build_runtime_spec_from_config(
        args_cli.config,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        encoder_checkpoint=args_cli.encoder_checkpoint,
    )
else:
    runtime_spec = _load_runtime_spec(
        args_cli.runtime_spec,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
    )

if runtime_spec.get("env_params", {}).get("robot_mode") != "official_panda_gripper":
    raise RuntimeError("This visualizer requires env_params.robot_mode=official_panda_gripper")

if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id") or DEFAULT_TASK
if args_cli.task != DEFAULT_TASK:
    raise RuntimeError(f"This visualizer expects task {DEFAULT_TASK!r}, got {args_cli.task!r}")

runtime_spec_path = _materialize_runtime_spec(runtime_spec)
os.environ[RUNTIME_SPEC_ENV_VAR] = runtime_spec_path
os.environ["TOOL_GENERALIST_PATHS_YAML"] = os.path.abspath(os.path.normpath(runtime_spec["paths_yaml"]))
os.environ.setdefault("TOOL_GENERALIST_GLOBAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_LOCAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_WORLD_SIZE", "1")

if runtime_spec.get("encoder_checkpoint") == UNUSED_ENCODER_CHECKPOINT:
    print(
        "[INFO] No encoder checkpoint resolved; using an unused placeholder for random-action visualization.",
        flush=True,
    )

args_cli.enable_cameras = bool(args_cli.video or getattr(args_cli, "enable_cameras", False))
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_actions(env, step: int) -> torch.Tensor:
    shape = env.action_space.shape
    device = env.unwrapped.device
    if args_cli.action_mode == "zero":
        actions = torch.zeros(shape, device=device)
    else:
        actions = 2.0 * torch.rand(shape, device=device) - 1.0

    if actions.shape[-1] >= 8:
        if args_cli.gripper_action_mode == "sweep":
            phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * float(step) / 120.0)
            actions[:, 7] = 2.0 * phase - 1.0
        elif args_cli.gripper_action_mode == "open":
            actions[:, 7] = 1.0
        elif args_cli.gripper_action_mode == "closed":
            actions[:, 7] = -1.0
    return actions


def _resolve_finger_joint_ids(base_env) -> list[int]:
    robot = base_env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    missing = [name for name in FINGER_JOINT_NAMES if name not in joint_names]
    if missing:
        raise RuntimeError(
            f"Missing official Panda finger joints {missing!r}; available joints are {tuple(joint_names)!r}"
        )
    return [joint_names.index(name) for name in FINGER_JOINT_NAMES]


def _print_gripper_debug(base_env, step: int, finger_joint_ids: list[int]) -> None:
    robot = base_env.scene["robot"]
    env_i = 0
    finger_pos = robot.data.joint_pos[env_i, finger_joint_ids].detach()
    opening = torch.clamp(finger_pos.mean() / GRIPPER_OPEN_JOINT_POS, 0.0, 1.0)
    bucket = int(torch.round(opening * (GRIPPER_NUM_BUCKETS - 1)).item())

    origin = base_env.scene.env_origins[env_i]
    center = None
    try:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
            get_official_panda_fingertip_center_pos_w,
        )

        center = get_official_panda_fingertip_center_pos_w(base_env)[env_i] - origin
    except Exception as exc:
        center_msg = f"unavailable:{type(exc).__name__}"
    else:
        center_msg = [round(float(v), 5) for v in center.detach().cpu().tolist()]

    bbox_center = getattr(base_env, "_obs_tool_bbox_center", None)
    if bbox_center is not None:
        bbox_msg = [round(float(v), 5) for v in bbox_center[env_i].detach().cpu().tolist()]
    else:
        bbox_msg = "not-populated"
    bbox_extent = getattr(base_env, "_obs_tool_bbox_extent", None)
    if bbox_extent is not None:
        bbox_extent_msg = [round(float(v), 5) for v in bbox_extent[env_i].detach().cpu().tolist()]
    else:
        bbox_extent_msg = "not-populated"
    cloud_source = getattr(base_env, "_obs_gripper_cloud_source", "not-populated")

    print(
        "[GRIPPER] "
        f"step={step} env=0 finger_joint_pos={[round(float(v), 5) for v in finger_pos.cpu().tolist()]} "
        f"opening={float(opening):.5f} bucket={bucket:02d}/{GRIPPER_NUM_BUCKETS - 1} "
        f"fingertip_center_E={center_msg} tool_cloud_bbox_E={bbox_msg} "
        f"tool_cloud_bbox_size_E={bbox_extent_msg} cloud_source={cloud_source}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    _seed_everything(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.visualize_head_area_center = True
    env_cfg.visualize_tool_pointcloud = True
    env_cfg.disable_obs_noise = True
    env_cfg.viewer.eye = (1.5, 0.5, 0.7)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.1)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            name_prefix=f"panda_gripper_random_{timestamp}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.num_steps,
            disable_logger=True,
        )

    print(f"[INFO] Runtime spec: {runtime_spec_path}", flush=True)
    print(f"[INFO] Running {args_cli.task} for {args_cli.num_steps} steps", flush=True)
    obs, _ = env.reset()
    base_env = env.unwrapped
    finger_joint_ids = _resolve_finger_joint_ids(base_env)
    _print_gripper_debug(base_env, 0, finger_joint_ids)

    for step in range(args_cli.num_steps):
        actions = _make_actions(env, step)
        obs, _, terminated, truncated, _ = env.step(actions)

        if step % args_cli.print_every == 0:
            _print_gripper_debug(base_env, step, finger_joint_ids)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

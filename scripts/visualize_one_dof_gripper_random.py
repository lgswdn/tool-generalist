#!/usr/bin/env python3
"""Visualize the one-DoF gripper signals consumed by RL.

The blue spheres are the exact 512-point tool cloud used by the policy. The red
sphere is the exact manifest-defined interaction center used by rewards and the
hand-state observation. A slow sweep uses the semantic convention -1=open and
+1=closed while leaving the Panda arm still by default.
"""

from __future__ import annotations

import argparse
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


DEFAULT_CONFIG = "configs/experiments/robotiq_2f140_diff_post.py"
DEFAULT_TASK = "one-dof-gripper-v0"
MODE_NAME = "visualize_one_dof_gripper_random"
TMP_ROOT = Path(tempfile.gettempdir()) / MODE_NAME
UNUSED_ENCODER_CHECKPOINT = "/tmp/tool_generalist_unused_one_dof_visualization_encoder.pt"


def _resolve_encoder_checkpoint(cfg, config: str) -> str | None:
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
        candidate = ref.directory / "best.pt"
        if candidate.exists():
            return str(candidate)
    return None


def _build_visualization_spec(
    config: str,
    *,
    num_envs: int,
    seed: int,
    encoder_checkpoint: str | None,
) -> dict:
    cfg = load_exp_cfg(config)
    if cfg.rl.env.robot_mode != "one_dof_gripper":
        raise RuntimeError(f"{config} must configure robot_mode=one_dof_gripper")
    if cfg.rl.isaac_task_id != DEFAULT_TASK:
        raise RuntimeError(f"{config} must configure task {DEFAULT_TASK!r}")
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    checkpoint = (
        encoder_checkpoint
        or _resolve_encoder_checkpoint(cfg, config)
        or UNUSED_ENCODER_CHECKPOINT
    )
    spec = asdict(
        build_rl_runtime_spec(
            cfg,
            paths,
            TMP_ROOT / Path(config).stem,
            mode=MODE_NAME,
            encoder_checkpoint_override=checkpoint,
        )
    )
    spec["source_config"] = str(config)
    spec["mode"] = MODE_NAME
    spec["num_envs"] = int(num_envs)
    spec["seed"] = int(seed)
    spec["task_id"] = DEFAULT_TASK
    spec["env_params"]["num_envs"] = int(num_envs)
    spec["env_params"]["robot_mode"] = "one_dof_gripper"
    spec["env_params"]["visualize_tool_pointcloud"] = True
    spec["observation_params"]["include_tool_cloud"] = True
    spec["observation_params"]["tool_cloud_source"] = "gripper_cloud_cache_v1"
    return spec


def _materialize_runtime_spec(spec: dict) -> str:
    spec_dir = TMP_ROOT / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    path = spec_dir / f"rl_runtime_spec_envs_{spec['num_envs']}.json"
    validate_runtime_spec(spec, path)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=600)
    parser.add_argument("--sweep_period", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--encoder_checkpoint", default=None)
    parser.add_argument(
        "--arm_action_mode",
        choices=("zero", "random"),
        default="zero",
        help="Keep zero to isolate gripper kinematics.",
    )
    parser.add_argument(
        "--gripper_action_mode",
        choices=("sweep", "open", "half", "closed", "random"),
        default="sweep",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_dir", default="videos/one_dof_gripper_random")
    parser.add_argument(
        "--no_debug_markers",
        action="store_true",
        help="Hide the blue RL-cloud and red interaction-center markers.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    for name in ("num_envs", "num_steps", "sweep_period", "print_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    args.video_dir = str(Path(args.video_dir).expanduser().resolve())
    return args, hydra_args


args_cli, hydra_args = _parse_args()
runtime_spec = _build_visualization_spec(
    args_cli.config,
    num_envs=args_cli.num_envs,
    seed=args_cli.seed,
    encoder_checkpoint=args_cli.encoder_checkpoint,
)
runtime_spec_path = _materialize_runtime_spec(runtime_spec)
os.environ[RUNTIME_SPEC_ENV_VAR] = runtime_spec_path
os.environ["TOOL_GENERALIST_PATHS_YAML"] = os.path.abspath(runtime_spec["paths_yaml"])
os.environ.setdefault("TOOL_GENERALIST_GLOBAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_LOCAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_WORLD_SIZE", "1")
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _semantic_gripper_command(step: int) -> float:
    mode = args_cli.gripper_action_mode
    if mode == "open":
        return -1.0
    if mode == "half":
        return 0.0
    if mode == "closed":
        return 1.0
    if mode == "random":
        return 2.0 * random.random() - 1.0
    # Start fully open, reach fully closed halfway through the period, and return.
    return -math.cos(2.0 * math.pi * float(step) / float(args_cli.sweep_period))


def _make_actions(env, step: int) -> torch.Tensor:
    shape = env.action_space.shape
    device = env.unwrapped.device
    if args_cli.arm_action_mode == "random":
        actions = 2.0 * torch.rand(shape, device=device) - 1.0
    else:
        actions = torch.zeros(shape, device=device)
    if actions.shape[-1] != 8:
        raise RuntimeError(f"Expected 8D arm+gripper action, got {shape}")
    actions[:, 7] = _semantic_gripper_command(step)
    return actions


def _resolve_debug_ids(base_env, gripper):
    robot = base_env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    missing_joints = [
        name for name in gripper.actuated_joint_names if name not in joint_names
    ]
    if missing_joints:
        raise RuntimeError(
            f"Missing actuated joints {missing_joints!r}; available={tuple(joint_names)!r}"
        )
    body_names = list(robot.data.body_names)
    cloud_body_names = tuple(dict.fromkeys(part.body_name for part in gripper.cloud_parts))
    missing = [name for name in cloud_body_names if name not in body_names]
    if missing:
        raise RuntimeError(f"Missing cloud bodies {missing!r}; available={tuple(body_names)!r}")
    return (
        [joint_names.index(name) for name in gripper.actuated_joint_names],
        [body_names.index(name) for name in cloud_body_names],
    )


def _vec(value: torch.Tensor) -> list[float]:
    return [round(float(item), 5) for item in value.detach().cpu().tolist()]


def _print_debug(
    base_env,
    step: int,
    semantic_command: float,
    joint_ids: list[int],
    body_ids: list[int],
    initial_body_pos,
) -> None:
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_one_dof_gripper_data_for_env,
    )
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
        get_head_area_pos_w,
    )

    gripper = get_one_dof_gripper_data_for_env(0)
    robot = base_env.scene["robot"]
    q = robot.data.joint_pos[0, joint_ids]
    qd = robot.data.joint_vel[0, joint_ids]
    arm_joint_ids = [
        index for index, name in enumerate(robot.data.joint_names) if name.startswith("panda_joint")
    ]
    max_arm_qd = torch.abs(robot.data.joint_vel[0, arm_joint_ids]).max()
    open_pos = torch.tensor(
        gripper.open_joint_positions,
        dtype=q.dtype,
        device=q.device,
    )
    closed_pos = torch.tensor(
        gripper.closed_joint_positions,
        dtype=q.dtype,
        device=q.device,
    )
    joint_closure = torch.clamp(
        (q - open_pos) / (closed_pos - open_pos),
        0.0,
        1.0,
    )
    measured_closure = joint_closure.mean()
    commanded = getattr(base_env, "_one_dof_gripper_commanded_closure", None)
    commanded_value = float(measured_closure) if commanded is None else float(commanded[0, 0])
    joint_tracking_error = commanded_value - joint_closure
    max_tracking_error = torch.abs(joint_tracking_error).max()
    synchronization_error = joint_closure.max() - joint_closure.min()
    center_e = get_head_area_pos_w(base_env)[0] - base_env.scene.env_origins[0]

    cloud = getattr(base_env, "_obs_tool_cloud_E", None)
    if cloud is None:
        cloud_msg = "not-populated"
        finite_msg = "n/a"
    else:
        cloud_msg = str(tuple(cloud[0].shape))
        finite_msg = str(bool(torch.isfinite(cloud[0]).all().item()))
    bbox = getattr(base_env, "_obs_tool_bbox_center", None)
    extent = getattr(base_env, "_obs_tool_bbox_extent", None)
    cache_bins = getattr(base_env, "_obs_gripper_bucket_ids", None)
    if cache_bins is None:
        raise RuntimeError("RL observation did not publish its gripper cache bin")
    cache_bin = int(cache_bins[0].detach().cpu())
    cloud_source = getattr(
        base_env, "_obs_gripper_cloud_source", "not-populated"
    )
    if cloud_source != "gripper_cloud_cache_v1":
        raise RuntimeError(
            f"Unexpected RL gripper cloud source: {cloud_source!r}"
        )
    body_pos = robot.data.body_state_w[0, body_ids, :3]
    max_body_motion = torch.linalg.vector_norm(body_pos - initial_body_pos, dim=1).max()

    print(
        "[ONE_DOF_GRIPPER] "
        f"step={step} id={gripper.gripper_id} command={semantic_command:+.4f} "
        f"commanded_closure={commanded_value:.4f} measured_closure={float(measured_closure):.4f} "
        f"tracking_error={commanded_value - float(measured_closure):+.4f} "
        f"max_joint_tracking_error={float(max_tracking_error):.4f} "
        f"joint_sync_error={float(synchronization_error):.4f} "
        f"joint_closure={_vec(joint_closure)} max_gripper_qd={float(torch.abs(qd).max()):.5f} "
        f"max_arm_qd={float(max_arm_qd):.5f} "
        f"rl_center_E={_vec(center_e)} cloud={cloud_msg} finite={finite_msg} "
        f"cache_bin={cache_bin:03d}/127 "
        f"cloud_source={cloud_source} "
        f"cloud_bbox_E={'not-populated' if bbox is None else _vec(bbox[0])} "
        f"cloud_extent_E={'not-populated' if extent is None else _vec(extent[0])} "
        f"max_cloud_link_motion={float(max_body_motion):.5f}",
        flush=True,
    )


@hydra_task_config(DEFAULT_TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    _seed_everything(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.visualize_tool_pointcloud = not args_cli.no_debug_markers
    env_cfg.visualize_head_area_center = not args_cli.no_debug_markers
    env_cfg.disable_obs_noise = True
    env_cfg.viewer.eye = (1.25, 0.55, 0.65)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.18)

    env = None
    try:
        print("[ONE_DOF_STARTUP] creating Isaac Lab environment", flush=True)
        env = gym.make(DEFAULT_TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        print("[ONE_DOF_STARTUP] Isaac Lab environment created", flush=True)
        if args_cli.video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print("[ONE_DOF_STARTUP] attaching video recorder", flush=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=args_cli.video_dir,
                name_prefix=f"one_dof_gripper_{timestamp}",
                step_trigger=lambda step: step == 0,
                video_length=args_cli.num_steps,
                disable_logger=True,
            )
            print("[ONE_DOF_STARTUP] video recorder attached", flush=True)
        print(f"[INFO] Runtime spec: {runtime_spec_path}", flush=True)
        print("[INFO] Blue=RL tool cloud; red=RL interaction center.", flush=True)
        print("[ONE_DOF_STARTUP] resetting environment", flush=True)
        env.reset()
        print("[ONE_DOF_STARTUP] environment reset complete", flush=True)
        base_env = env.unwrapped
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_one_dof_gripper_data_for_env,
        )

        gripper = get_one_dof_gripper_data_for_env(0)
        joint_ids, body_ids = _resolve_debug_ids(base_env, gripper)
        initial_body_pos = base_env.scene["robot"].data.body_state_w[0, body_ids, :3].clone()
        _print_debug(base_env, 0, -1.0, joint_ids, body_ids, initial_body_pos)
        for step in range(args_cli.num_steps):
            actions = _make_actions(env, step)
            semantic_command = float(actions[0, 7].detach().cpu())
            env.step(actions)
            if step % args_cli.print_every == 0:
                _print_debug(
                    base_env,
                    step,
                    semantic_command,
                    joint_ids,
                    body_ids,
                    initial_body_pos,
                )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

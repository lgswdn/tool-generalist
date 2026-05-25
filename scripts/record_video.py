#!/usr/bin/env python3
"""Record an Isaac task video without loading a policy."""

import argparse
import copy
import json
import os
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


def _build_runtime_spec_from_config(config: str, num_envs: int) -> dict:
    cfg = load_exp_cfg(config)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifact_dir = Path(tempfile.gettempdir()) / "tool_generalist_record_video" / Path(config).stem
    spec = build_rl_runtime_spec(
        cfg,
        paths,
        artifact_dir,
        mode="record_video",
        encoder_checkpoint_override=_resolve_encoder_checkpoint_for_config(cfg, config),
    )
    payload = asdict(spec)
    payload["source_config"] = str(config)
    return _runtime_spec_for_recording(payload, num_envs)


def _load_runtime_spec(path: str, num_envs: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime spec must be a JSON object: {path}")
    return _runtime_spec_for_recording(payload, num_envs)


def _runtime_spec_for_recording(spec: dict, num_envs: int) -> dict:
    spec = copy.deepcopy(spec)
    spec["num_envs"] = int(num_envs)
    if isinstance(spec.get("env_params"), dict):
        spec["env_params"]["num_envs"] = int(num_envs)
    return spec


def _materialize_runtime_spec(spec: dict) -> str:
    spec_dir = Path(tempfile.gettempdir()) / "tool_generalist_record_video"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"rl_runtime_spec_record_video_envs_{spec['num_envs']}.json"
    validate_runtime_spec(spec, spec_path)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return str(spec_path)


parser = argparse.ArgumentParser(description="Record a task video with zero or random actions.")
source_group = parser.add_mutually_exclusive_group()
source_group.add_argument("--config", type=str, help="Experiment config exposing EXP_CFG.")
source_group.add_argument("--runtime_spec", type=str, help="Existing rl_runtime_spec.json.")
parser.add_argument("--task", type=str, default=None, help="Task name. Defaults to the runtime spec task_id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--video_length", type=int, default=300, help="Video length in steps.")
parser.add_argument("--video_dir", type=str, default="videos", help="Video output directory.")
parser.add_argument("--action_mode", choices=("zero", "random"), default="zero", help="Actions to apply while recording.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.config:
    runtime_spec = _build_runtime_spec_from_config(args_cli.config, args_cli.num_envs)
elif args_cli.runtime_spec:
    runtime_spec = _load_runtime_spec(args_cli.runtime_spec, args_cli.num_envs)
else:
    runtime_spec_path = os.environ.get(RUNTIME_SPEC_ENV_VAR)
    if not runtime_spec_path:
        parser.error(f"Provide --config, --runtime_spec, or set {RUNTIME_SPEC_ENV_VAR}.")
    runtime_spec = _load_runtime_spec(runtime_spec_path, args_cli.num_envs)

if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id")
if not args_cli.task:
    parser.error("--task is required when the runtime spec does not contain task_id.")

runtime_spec_path = _materialize_runtime_spec(runtime_spec)
os.environ[RUNTIME_SPEC_ENV_VAR] = runtime_spec_path
os.environ["TOOL_GENERALIST_PATHS_YAML"] = os.path.abspath(os.path.normpath(runtime_spec["paths_yaml"]))
os.environ.setdefault("TOOL_GENERALIST_GLOBAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_LOCAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_WORLD_SIZE", "1")

args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
import IsaacLab_nonPrehensile.tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


def _make_actions(env) -> torch.Tensor:
    if args_cli.action_mode == "zero":
        return torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    return torch.tensor(env.action_space.sample(), device=env.unwrapped.device)


def _print_distance_debug(base_env, step: int) -> None:
    obj_pos_w = base_env.scene["object"].data.root_pos_w
    if getattr(base_env.cfg, "bimanual", False):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp import (
            get_head_area_pos_w_for_slot,
        )

        ee1_pos_w = get_head_area_pos_w_for_slot(
            base_env,
            ee_frame_name="ee_frame_1",
            offsets_attr="_head_area_offsets_1",
        )
        ee2_pos_w = get_head_area_pos_w_for_slot(
            base_env,
            ee_frame_name="ee_frame_2",
            offsets_attr="_head_area_offsets_2",
        )
        dist1 = torch.norm(ee1_pos_w - obj_pos_w, dim=1)
        dist2 = torch.norm(ee2_pos_w - obj_pos_w, dim=1)
        print(
            f"[DEBUG] Step {step} | EE1-obj distance: {dist1.cpu().numpy()} "
            f"| EE2-obj distance: {dist2.cpu().numpy()}",
            flush=True,
        )
        return

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp import get_head_area_pos_w

    ee_pos_w = get_head_area_pos_w(base_env)
    ee_obj_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)
    print(f"[DEBUG] Step {step} | EE-obj distance: {ee_obj_dist.cpu().numpy()}", flush=True)


def _fmt_tensor(values: torch.Tensor) -> list[float]:
    return [round(float(v), 6) for v in values.detach().cpu().tolist()]


def _print_layout_debug(base_env) -> None:
    env_i = 0
    origin = base_env.scene.env_origins[env_i]
    obj = base_env.scene["object"]
    object_pos_e = obj.data.root_pos_w[env_i, :3] - origin
    object_quat = obj.data.root_quat_w[env_i]
    command = base_env.command_manager.get_command("target_object_pose")
    target_pos_e = command[env_i, :3]
    target_quat = command[env_i, 3:7]

    print(
        "[LAYOUT] "
        f"env={env_i} origin_W={_fmt_tensor(origin)} "
        f"object_pos_E={_fmt_tensor(object_pos_e)} "
        f"object_quat_wxyz={_fmt_tensor(object_quat)} "
        f"target_pos_E={_fmt_tensor(target_pos_e)} "
        f"target_quat_wxyz={_fmt_tensor(target_quat)}",
        flush=True,
    )

    robot_names = ["robot_1", "robot_2"] if getattr(base_env.cfg, "bimanual", False) else ["robot"]
    for name in robot_names:
        robot = base_env.scene[name]
        root_pos_e = robot.data.root_pos_w[env_i, :3] - origin
        root_quat = robot.data.root_quat_w[env_i]
        print(
            "[LAYOUT] "
            f"{name}_base_pos_E={_fmt_tensor(root_pos_e)} "
            f"{name}_base_quat_wxyz={_fmt_tensor(root_quat)}",
            flush=True,
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(env_cfg, "bimanual", False):
        env_cfg.viewer.eye = (0.5, -1.45, 0.85)
        env_cfg.viewer.lookat = (0.5, 0.0, 0.25)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_kwargs = {
        "video_folder": args_cli.video_dir,
        "name_prefix": f"video_{timestamp}",
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO] Recording {args_cli.video_length} steps")
    obs, _ = env.reset()
    print("[DEBUG] Reset complete")
    _print_layout_debug(env.unwrapped)

    for step in range(args_cli.video_length):
        print(f"[DEBUG] Starting step {step}")
        actions = _make_actions(env)
        print(f"[DEBUG] About to call env.step() for step {step}")
        obs, _, terminated, truncated, _ = env.step(actions)
        print(f"[DEBUG] env.step() completed for step {step}")

        _print_distance_debug(env.unwrapped, step)

        if terminated.any() or truncated.any():
            obs, _ = env.reset()
        if step % 50 == 0:
            print(f"[INFO] Step {step}/{args_cli.video_length}")

    print("[INFO] Video complete")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

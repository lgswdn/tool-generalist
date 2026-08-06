#!/usr/bin/env python3
"""Visualize the generated-gripper env with random actions.

This mirrors ``visualize_panda_gripper_random.py`` but selects
``generated-gripper-v0`` and reads generated-gripper metadata from the runtime
manifest. It is intended for machines with Isaac/IsaacLab runtime available.

Useful debug signals:

- red fingertip-midpoint/tool-center marker from generated metadata,
- blue policy gripper point cloud after openness-bucket selection,
- periodic env-0 logs with gripper id, manifest finger joints, opening bucket,
  fingertip center, and tool-cloud AABB.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
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


DEFAULT_CONFIG = "configs/experiments/panda_general_dpoc_gg.py"
DEFAULT_TASK = "generated-gripper-v0"
MODE_NAME = "visualize_generated_gripper_random"
TMP_ROOT = Path(tempfile.gettempdir()) / "tool_generalist_generated_gripper_random"
UNUSED_ENCODER_CHECKPOINT = "/tmp/tool_generalist_unused_generated_gripper_visualization_encoder.pt"
DEFAULT_GENERATED_GRIPPER_ROOT = Path("/mnt/project/world_model/tool_generalist/gripper")
DEFAULT_GENERATED_GRIPPER_MANIFEST = DEFAULT_GENERATED_GRIPPER_ROOT / "generated_grippers.json"


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
    paths_yaml: str | None,
    generated_gripper_manifest: str | None,
    include_tool_cloud: bool,
) -> dict:
    cfg = load_exp_cfg(config)
    if paths_yaml is not None:
        cfg.paths_yaml = paths_yaml

    # Pretrain-only configs retain the default tool-mode RL shell.  This
    # visualizer still needs the generated-gripper task's action, observation,
    # table, and physics contract even though it never loads an RL policy.
    cfg.rl.isaac_task_id = DEFAULT_TASK
    cfg.rl.env.robot_mode = "generated_gripper"
    cfg.rl.action.action_dim = 8
    cfg.rl.observation.previous_action_dim = 8
    cfg.rl.observation.robot_state_dim = 18
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.table.enabled = True
    cfg.rl.domain_randomization.ground.material.enabled = False

    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifact_dir = TMP_ROOT / Path(config).stem

    checkpoint = (
        encoder_checkpoint
        or _resolve_encoder_checkpoint_for_config(cfg, config)
        or UNUSED_ENCODER_CHECKPOINT
    )
    spec = build_rl_runtime_spec(
        cfg,
        paths,
        artifact_dir,
        mode=MODE_NAME,
        encoder_checkpoint_override=checkpoint,
    )
    payload = asdict(spec)
    payload["source_config"] = str(config)
    return _runtime_spec_for_visualization(
        payload,
        num_envs=num_envs,
        seed=seed,
        paths_yaml=paths_yaml,
        generated_gripper_manifest=generated_gripper_manifest,
        include_tool_cloud=include_tool_cloud,
    )


def _load_runtime_spec(
    path: str,
    *,
    num_envs: int,
    seed: int | None,
    paths_yaml: str | None,
    generated_gripper_manifest: str | None,
    include_tool_cloud: bool,
) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime spec must be a JSON object: {path}")
    return _runtime_spec_for_visualization(
        payload,
        num_envs=num_envs,
        seed=seed,
        paths_yaml=paths_yaml,
        generated_gripper_manifest=generated_gripper_manifest,
        include_tool_cloud=include_tool_cloud,
    )


def _runtime_spec_for_visualization(
    spec: dict,
    *,
    num_envs: int,
    seed: int | None,
    paths_yaml: str | None,
    generated_gripper_manifest: str | None,
    include_tool_cloud: bool,
) -> dict:
    spec = copy.deepcopy(spec)
    spec["mode"] = MODE_NAME
    spec["num_envs"] = int(num_envs)
    if seed is not None:
        spec["seed"] = int(seed)

    env_params = spec.setdefault("env_params", {})
    env_params["num_envs"] = int(num_envs)
    env_params["robot_mode"] = "generated_gripper"

    spec["task_id"] = DEFAULT_TASK
    observation = spec.setdefault("observation_params", {})
    observation["include_tool_cloud"] = True
    observation["tool_cloud_source"] = "gripper_cloud_cache_v1"
    if not include_tool_cloud:
        observation["include_tool_cloud"] = False
        layout = [
            name
            for name in observation.get("layout", [])
            if name not in {"tool_cloud_flat", "tool_bbox_center"}
        ]
        observation["layout"] = layout
        tool_cloud_dim = int(observation.get("num_points", 512)) * int(observation.get("point_dim", 3))
        bbox_dim = 3 if bool(observation.get("include_bbox_centers", False)) else 0
        spec["observation_dim"] = max(0, int(spec["observation_dim"]) - tool_cloud_dim - bbox_dim)

    if paths_yaml is not None:
        spec["paths_yaml"] = str(Path(paths_yaml).expanduser().resolve())
    manifest_to_inject = generated_gripper_manifest
    if manifest_to_inject is None and not _paths_yaml_has_generated_manifest(spec["paths_yaml"]):
        manifest_to_inject = str(DEFAULT_GENERATED_GRIPPER_MANIFEST)
    if manifest_to_inject is not None:
        spec["paths_yaml"] = _paths_yaml_with_generated_manifest(
            spec["paths_yaml"],
            manifest_to_inject,
        )
    return spec


def _paths_yaml_has_generated_manifest(paths_yaml: str) -> bool:
    import yaml

    source = Path(paths_yaml).expanduser().resolve()
    with source.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"paths.yaml must contain a mapping: {source}")
    section = raw.get("generated_grippers")
    return isinstance(section, dict) and bool(section.get("manifest"))


def _paths_yaml_with_generated_manifest(paths_yaml: str, manifest: str) -> str:
    import yaml

    source = Path(paths_yaml).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    with source.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"paths.yaml must contain a mapping: {source}")
    section = raw.setdefault("generated_grippers", {})
    if not isinstance(section, dict):
        raise RuntimeError(f"paths.yaml generated_grippers must be a mapping: {source}")
    section["manifest"] = str(manifest_path)

    out_dir = TMP_ROOT / "paths"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paths.generated_gripper_visualize.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return str(out_path)


def _materialize_runtime_spec(spec: dict) -> str:
    spec_dir = TMP_ROOT / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"rl_runtime_spec_generated_gripper_random_envs_{spec['num_envs']}.json"
    validate_runtime_spec(spec, spec_path)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return str(spec_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run generated-gripper-v0 with random actions and debug visualizers."
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
    parser.add_argument("--paths_yaml", type=str, default=None, help="Override runtime spec paths_yaml.")
    parser.add_argument(
        "--generated_gripper_manifest",
        type=str,
        default=None,
        help=(
            "Inject generated_grippers.manifest into a temporary runtime paths.yaml. "
            f"Defaults to {DEFAULT_GENERATED_GRIPPER_MANIFEST} only if paths.yaml lacks the key."
        ),
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of policy steps to run. Defaults to 600.",
    )
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
    parser.add_argument(
        "--video_backend",
        choices=("record_video", "frames"),
        default="record_video",
        help="Video backend. Use frames only for render-hang diagnosis; default writes an mp4 via Gym RecordVideo.",
    )
    parser.add_argument(
        "--frame_every",
        type=int,
        default=1,
        help="When --video_backend=frames, save one rendered frame every N env steps.",
    )
    parser.add_argument(
        "--render_timeout_s",
        type=int,
        default=30,
        help="Fail frame dumping if one env.render() call takes longer than this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--no_debug_markers",
        action="store_true",
        default=False,
        help="Disable point-cloud/head-center debug markers for faster video recording.",
    )
    parser.add_argument(
        "--no_tool_cloud",
        action="store_true",
        default=False,
        help="Disable generated gripper point-cloud observation for hang diagnosis.",
    )
    parser.add_argument(
        "--trace_steps",
        action="store_true",
        default=False,
        help="Print before and after each env.step for hang diagnosis.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos/generated_gripper_random",
        help="Video output directory.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args_cli.num_steps is None:
        args_cli.num_steps = 100
    if args_cli.num_steps <= 0:
        parser.error("--num_steps must be positive")
    if args_cli.print_every <= 0:
        parser.error("--print_every must be positive")
    if args_cli.frame_every <= 0:
        parser.error("--frame_every must be positive")
    if args_cli.render_timeout_s < 0:
        parser.error("--render_timeout_s must be >= 0")
    if args_cli.config is None and args_cli.runtime_spec is None:
        args_cli.config = DEFAULT_CONFIG
    args_cli.video_dir = str(Path(args_cli.video_dir).expanduser().resolve())
    return args_cli, hydra_args


args_cli, hydra_args = _parse_args()

if args_cli.config:
    runtime_spec = _build_runtime_spec_from_config(
        args_cli.config,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        encoder_checkpoint=args_cli.encoder_checkpoint,
        paths_yaml=args_cli.paths_yaml,
        generated_gripper_manifest=args_cli.generated_gripper_manifest,
        include_tool_cloud=not args_cli.no_tool_cloud,
    )
else:
    runtime_spec = _load_runtime_spec(
        args_cli.runtime_spec,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        paths_yaml=args_cli.paths_yaml,
        generated_gripper_manifest=args_cli.generated_gripper_manifest,
        include_tool_cloud=not args_cli.no_tool_cloud,
    )

if runtime_spec.get("env_params", {}).get("robot_mode") != "generated_gripper":
    raise RuntimeError("This visualizer requires env_params.robot_mode=generated_gripper")

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


def _generated_gripper_for_env0():
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_generated_gripper_data_for_env,
    )

    return get_generated_gripper_data_for_env(0)


def _resolve_finger_joint_ids(base_env, gripper) -> list[int]:
    robot = base_env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    missing = [name for name in gripper.finger_joint_names if name not in joint_names]
    if missing:
        raise RuntimeError(
            f"Missing generated finger joints {missing!r}; available joints are {tuple(joint_names)!r}"
        )
    return [joint_names.index(name) for name in gripper.finger_joint_names]


def _print_gripper_debug(base_env, step: int, finger_joint_ids: list[int]) -> None:
    gripper = _generated_gripper_for_env0()
    robot = base_env.scene["robot"]
    env_i = 0
    finger_pos = robot.data.joint_pos[env_i, finger_joint_ids].detach()
    opening = torch.clamp(finger_pos.mean() / float(gripper.open_joint_pos), 0.0, 1.0)
    bucket = int(torch.round(opening * 127).item())

    origin = base_env.scene.env_origins[env_i]
    try:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
            get_generated_gripper_fingertip_center_pos_w,
        )

        center = get_generated_gripper_fingertip_center_pos_w(base_env)[env_i] - origin
    except Exception as exc:
        center_msg = f"unavailable:{type(exc).__name__}"
    else:
        center_msg = [round(float(v), 5) for v in center.detach().cpu().tolist()]

    observed_bucket = getattr(base_env, "_obs_gripper_bucket_ids", None)
    if observed_bucket is None:
        raise RuntimeError("RL observation did not publish its gripper cache bin")
    observed_bucket_msg = int(observed_bucket[env_i].detach().cpu().item())
    if observed_bucket_msg != bucket:
        raise RuntimeError(
            f"RL cache bin {observed_bucket_msg} != measured joint bin {bucket}"
        )

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
    if cloud_source != "gripper_cloud_cache_v1":
        raise RuntimeError(
            f"Unexpected RL gripper cloud source: {cloud_source!r}"
        )

    print(
        "[GENERATED_GRIPPER] "
        f"step={step} env=0 id={gripper.gripper_id} "
        f"finger_joints={tuple(gripper.finger_joint_names)!r} "
        f"finger_joint_pos={[round(float(v), 5) for v in finger_pos.cpu().tolist()]} "
        f"opening={float(opening):.5f} bucket={bucket:03d}/127 "
        f"observed_bucket={observed_bucket_msg} fingertip_center_E={center_msg} "
        f"tool_cloud_bbox_E={bbox_msg} tool_cloud_bbox_size_E={bbox_extent_msg} "
        f"cloud_source={cloud_source}",
        flush=True,
    )


def _save_rgb_frame(frame, path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("--video_backend=frames requires the optional dependency 'imageio'") from exc

    if frame is None:
        raise RuntimeError("env.render() returned None; Isaac RGB rendering is not producing frames")
    if isinstance(frame, tuple):
        if not frame:
            raise RuntimeError("env.render() returned an empty tuple")
        frame = frame[0]
    if isinstance(frame, dict):
        for key in ("rgb", "rgb_array", "image", "frame"):
            if key in frame:
                frame = frame[key]
                break
        else:
            raise RuntimeError(f"env.render() returned a dict without an RGB frame key: {sorted(frame)}")
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, list):
        if len(frame) != 1:
            raise RuntimeError(f"Expected one rendered frame for num_envs=1, got {len(frame)}")
        frame = frame[0]
    if hasattr(frame, "shape") and len(frame.shape) == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if hasattr(frame, "dtype") and str(frame.dtype).startswith("float"):
        frame = np.clip(frame, 0.0, 1.0)
        frame = (255.0 * frame).astype(np.uint8)
    imageio.imwrite(path, frame)


@contextlib.contextmanager
def _render_timeout(seconds: int, label: str):
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Timed out rendering {label!r} after {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _render_and_save_frame(env, frame_dir: Path, frame_index: int, label: str) -> None:
    frame_path = frame_dir / f"frame_{frame_index:06d}.png"
    if args_cli.trace_steps:
        print(f"[TRACE] before env.render {label}", flush=True)
    with _render_timeout(args_cli.render_timeout_s, label):
        frame = env.render()
    if args_cli.trace_steps:
        print(f"[TRACE] after env.render {label}", flush=True)
    _save_rgb_frame(frame, frame_path)
    if args_cli.trace_steps:
        print(f"[TRACE] saved frame {frame_path}", flush=True)


def _print_video_outputs(video_dir: str, started_at: float) -> None:
    root = Path(video_dir)
    if not root.exists():
        print(f"[WARNING] Video output directory was not created: {root}", flush=True)
        return
    outputs = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.stat().st_mtime >= started_at - 1.0
    ]
    if not outputs:
        print(f"[WARNING] No video/frame files were written under: {root}", flush=True)
        return
    print("[INFO] Video/frame outputs:", flush=True)
    for path in outputs:
        print(f"  {path}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    _seed_everything(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.visualize_head_area_center = not args_cli.no_debug_markers
    env_cfg.visualize_tool_pointcloud = not args_cli.no_debug_markers
    env_cfg.disable_obs_noise = True
    env_cfg.viewer.eye = (1.5, 0.5, 0.7)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.1)

    env = None
    video_started_at = time.time()
    try:
        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        frame_dir = None
        frame_index = 0
        if args_cli.video and args_cli.video_backend == "record_video":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=args_cli.video_dir,
                name_prefix=f"generated_gripper_random_{timestamp}",
                step_trigger=lambda step: step == 0,
                video_length=args_cli.num_steps,
                disable_logger=True,
            )
        elif args_cli.video and args_cli.video_backend == "frames":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_dir = Path(args_cli.video_dir) / f"generated_gripper_frames_{timestamp}"
            frame_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Runtime spec: {runtime_spec_path}", flush=True)
        print(f"[INFO] Paths YAML: {runtime_spec['paths_yaml']}", flush=True)
        print(f"[INFO] Running {args_cli.task} for {args_cli.num_steps} steps", flush=True)
        if args_cli.no_tool_cloud:
            print("[INFO] Generated gripper tool-cloud observation is disabled for diagnosis.", flush=True)
        if args_cli.video:
            print(
                f"[INFO] Video enabled: backend={args_cli.video_backend} output={args_cli.video_dir}",
                flush=True,
            )
        obs, _ = env.reset()
        base_env = env.unwrapped
        gripper = _generated_gripper_for_env0()
        finger_joint_ids = _resolve_finger_joint_ids(base_env, gripper)
        _print_gripper_debug(base_env, 0, finger_joint_ids)
        if frame_dir is not None:
            _render_and_save_frame(env, frame_dir, frame_index, "reset")
            frame_index += 1

        start_time = time.monotonic()
        for step in range(args_cli.num_steps):
            if args_cli.trace_steps:
                print(f"[TRACE] before env.step step={step}", flush=True)
            actions = _make_actions(env, step)
            obs, _, terminated, truncated, _ = env.step(actions)
            if args_cli.trace_steps:
                print(f"[TRACE] after env.step step={step}", flush=True)

            if step % args_cli.print_every == 0:
                if args_cli.trace_steps:
                    print(f"[TRACE] before debug print step={step}", flush=True)
                _print_gripper_debug(base_env, step, finger_joint_ids)
                if args_cli.trace_steps:
                    print(f"[TRACE] after debug print step={step}", flush=True)
                elapsed = time.monotonic() - start_time
                print(
                    f"[INFO] Progress: step {step + 1}/{args_cli.num_steps} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if frame_dir is not None and step % args_cli.frame_every == 0:
                _render_and_save_frame(env, frame_dir, frame_index, f"step={step}")
                frame_index += 1

        print("[INFO] Completed requested steps.", flush=True)
    finally:
        if env is not None:
            print("[INFO] Closing env and video writer.", flush=True)
            env.close()
            print("[INFO] Env closed.", flush=True)
        if args_cli.video:
            _print_video_outputs(args_cli.video_dir, video_started_at)
        print("[INFO] Closing Isaac app.", flush=True)
        simulation_app.close()
        print("[INFO] Visualization finished.", flush=True)


if __name__ == "__main__":
    main()

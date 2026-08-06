#!/usr/bin/env python3
"""Evaluate whether GraspGen Panda grasps can lift objects."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
DEFAULT_GRASPGEN_ROOT = "/mnt/project/world_model/tool_generalist/GraspGen"
DEFAULT_GRASPGEN_PORT = 5556
DEFAULT_ACTION_SCALE = 0.1
FFMPEG_PATH = "/usr/bin/ffmpeg"
MODE_NAME = "graspgen_lift_grasps_eval"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GraspGen candidates by trying every full-safe and hand-safe "
            "grasp for each reset object pose. A sample fails only if all such "
            "candidate grasps fail to lift the object."
        )
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Experiment config exposing EXP_CFG.")
    source_group.add_argument("--runtime_spec", type=str, help="Existing rl_runtime_spec.json.")
    parser.add_argument("--task", type=str, default=None, help="Task id. Defaults to runtime spec task_id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime seed override.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of reset-and-grasp rounds to run.")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker/rank id used for output naming and seed offset.")
    parser.add_argument("--num_workers", type=int, default=1, help="Total worker count recorded in summary metadata.")
    parser.add_argument(
        "--worker_seed_stride",
        type=int,
        default=100000,
        help="When --seed is set, add worker_id * worker_seed_stride to decorrelate workers.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/graspgen_lift_grasps_eval",
        help="Directory for summary JSON and failure JSONL outputs.",
    )
    parser.add_argument("--summary_json", type=str, default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--failures_jsonl", type=str, default=None, help="Optional explicit failure JSONL path.")
    parser.add_argument(
        "--all_results_jsonl",
        type=str,
        default=None,
        help="Optional JSONL path for every env attempt. By default only failures are written.",
    )
    parser.add_argument(
        "--paths_yaml",
        type=str,
        default=None,
        help=(
            "Optional paths.yaml override. By default config-built specs use "
            "configs/paths/panda_hand.yaml when present so official Panda assets resolve."
        ),
    )

    parser.add_argument("--graspgen_root", type=str, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--graspgen_host", type=str, default="localhost")
    parser.add_argument("--graspgen_port", type=int, default=DEFAULT_GRASPGEN_PORT)
    parser.add_argument("--graspgen_timeout_ms", type=int, default=120_000)
    parser.add_argument(
        "--graspgen_point_cloud_points",
        type=int,
        default=2048,
        help="Number of object cloud points sent to GraspGen per env.",
    )
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--topk_num_grasps", type=int, default=20)
    parser.add_argument("--grasp_threshold", type=float, default=-1.0)
    parser.add_argument("--min_grasps", type=int, default=1)
    parser.add_argument("--max_tries", type=int, default=6)
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help="Ask GraspGen server to remove point-cloud outliers. Disabled by default to avoid empty clouds.",
    )
    parser.add_argument(
        "--fail_on_graspgen_error",
        action="store_true",
        help="Stop the script if GraspGen fails for any env instead of using a no-op fallback for that env.",
    )

    parser.add_argument("--record_video", action="store_true", help="Record one MP4 per environment per episode.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory for per-env MP4 outputs. Defaults to <results_dir>/videos.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=0,
        help="Maximum frames to record per episode. 0 records the full scripted motion.",
    )
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument(
        "--record_warmup",
        action="store_true",
        help="Start recording at reset/open warmup instead of after GraspGen inference.",
    )

    parser.add_argument(
        "--visualize_grasps",
        dest="no_visualize_grasps",
        action="store_false",
        default=True,
        help="Render selected GraspGen grasp guides. Disabled by default for large-scale evaluation.",
    )
    parser.add_argument("--grasp_line_width", type=float, default=0.08, help="Distance between the two grasp guide lines.")
    parser.add_argument("--grasp_line_depth", type=float, default=0.10, help="Length of each grasp guide line.")
    parser.add_argument("--grasp_line_thickness", type=float, default=0.006, help="Rendered thickness of grasp guide lines.")
    parser.add_argument(
        "--visualize_candidate_grasps",
        action="store_true",
        help="Render all returned GraspGen candidates with table-filter color coding.",
    )
    parser.add_argument(
        "--candidate_grasp_vis_limit",
        type=int,
        default=0,
        help="Maximum candidate grasps to render per env. 0 renders every returned candidate.",
    )
    parser.add_argument(
        "--candidate_grasp_line_thickness",
        type=float,
        default=0.0025,
        help="Rendered thickness for non-selected candidate grasp guide lines.",
    )
    parser.add_argument(
        "--visualize_object_cloud",
        dest="no_visualize_object_cloud",
        action="store_false",
        default=True,
        help="Render the object point cloud sent to GraspGen. Disabled by default for large-scale evaluation.",
    )
    parser.add_argument("--object_cloud_vis_points", type=int, default=128, help="Maximum object cloud points to render per env.")
    parser.add_argument("--object_cloud_point_size", type=float, default=0.006, help="Rendered cube size for each object cloud point.")
    parser.add_argument(
        "--visualize_object_markers",
        action="store_true",
        help="Show current object pose and target object pose frame markers from the Isaac task.",
    )
    parser.add_argument(
        "--no_table_collision_filter",
        action="store_true",
        help="Disable filtering out GraspGen candidates whose Panda gripper proxy penetrates the table.",
    )
    parser.add_argument(
        "--table_collision_clearance",
        type=float,
        default=0.005,
        help="Required clearance above tabletop for gripper proxy points inside the table footprint.",
    )
    parser.add_argument(
        "--table_collision_xy_margin",
        type=float,
        default=0.0,
        help="Extra XY margin added to the table footprint during gripper/table filtering.",
    )
    parser.add_argument(
        "--table_collision_proxy_points",
        type=int,
        default=512,
        help="Maximum gripper proxy points used for table collision filtering.",
    )
    parser.add_argument(
        "--unsafe_grasp_fallback",
        choices=("noop", "best"),
        default="noop",
        help="Deprecated. When no full-safe grasp exists the script now falls back to hand-safe, then most-upward.",
    )

    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--open_steps", type=int, default=10)
    parser.add_argument("--approach_steps", type=int, default=80)
    parser.add_argument("--grasp_steps", type=int, default=50)
    parser.add_argument("--close_steps", type=int, default=35)
    parser.add_argument("--lift_steps", type=int, default=60)
    parser.add_argument("--goal_transport_steps", type=int, default=80)
    parser.add_argument("--hold_steps", type=int, default=20)
    parser.add_argument("--pregrasp_offset", type=float, default=0.10)
    parser.add_argument("--lift_distance", type=float, default=0.12)
    parser.add_argument(
        "--lift_success_height_delta",
        type=float,
        default=0.05,
        help="Object z increase over the initial pose required to count as a successful grasp lift.",
    )
    parser.add_argument(
        "--lift_success_stable_seconds",
        type=float,
        default=1.0,
        help=(
            "Seconds to keep the gripper closed after lifting. The object must stay above "
            "the lift-success height for this whole interval."
        ),
    )
    parser.add_argument(
        "--lift_success_stability_tolerance",
        type=float,
        default=0.005,
        help="Allowed z-height slack in meters during the stability hold.",
    )
    parser.add_argument(
        "--max_candidate_attempts",
        type=int,
        default=0,
        help="Maximum full-safe/hand-safe candidates to attempt per sample. 0 attempts all safe candidates.",
    )
    parser.add_argument(
        "--restore_settle_steps",
        type=int,
        default=2,
        help="Short open-gripper settling steps after restoring the initial state before each candidate attempt.",
    )
    parser.add_argument("--max_pos_step", type=float, default=0.035)
    parser.add_argument("--max_rot_step", type=float, default=0.25)
    parser.add_argument("--ik_damping", type=float, default=0.08)
    parser.add_argument("--ik_method", choices=("dls", "pinv", "svd", "trans"), default="dls")
    parser.add_argument(
        "--use_curobo_pregrasp",
        action="store_true",
        help="Use cuRobo MotionGen for the initial move from reset/open pose to pregrasp.",
    )
    parser.add_argument(
        "--use_curobo_goal_transport",
        action="store_true",
        help=(
            "When post-grasp motion is goal_transport, use cuRobo before falling back "
            "to differential IK."
        ),
    )
    parser.add_argument(
        "--post_grasp_motion",
        choices=("goal_transport", "lift"),
        default="lift",
        help=(
            "Motion after closing the gripper. Eval defaults to moving the grasped object "
            "toward target_object_pose; use lift to only raise the grasp."
        ),
    )
    parser.add_argument(
        "--pre_transport_lift",
        action="store_true",
        help=(
            "When post-grasp motion is goal_transport, lift the grasped object first, "
            "then re-measure the object-hand transform and transport to the target."
        ),
    )
    parser.add_argument(
        "--curobo_goal_object_z_offset",
        type=float,
        default=0.0,
        help="Optional z offset added to the commanded object goal before computing the carried hand pose.",
    )
    parser.add_argument("--curobo_robot_config", type=str, default="franka.yml")
    parser.add_argument("--curobo_interpolation_dt", type=float, default=0.02)
    parser.add_argument("--curobo_trajopt_tsteps", type=int, default=34)
    parser.add_argument("--curobo_interpolation_steps", type=int, default=5000)
    parser.add_argument("--curobo_ik_seeds", type=int, default=50)
    parser.add_argument("--curobo_trajopt_seeds", type=int, default=6)
    parser.add_argument("--curobo_graph_seeds", type=int, default=4)
    parser.add_argument("--curobo_grad_trajopt_iters", type=int, default=500)
    parser.add_argument("--curobo_max_attempts", type=int, default=6)
    parser.add_argument("--curobo_timeout", type=float, default=10.0)
    parser.add_argument("--curobo_collision_activation_distance", type=float, default=0.02)
    parser.add_argument("--curobo_table_padding", type=float, default=0.02)
    parser.add_argument("--curobo_object_padding", type=float, default=0.03)
    parser.add_argument(
        "--no_curobo_table_collision",
        action="store_true",
        help="Do not add the table/ground cuboid to the cuRobo pregrasp world.",
    )
    parser.add_argument(
        "--no_curobo_object_collision",
        action="store_true",
        help="Do not add an object AABB cuboid to the cuRobo pregrasp world.",
    )
    parser.add_argument(
        "--fail_on_curobo_error",
        action="store_true",
        help="Stop if cuRobo pregrasp planning fails instead of falling back to differential IK.",
    )
    parser.add_argument(
        "--grasp_pose_frame",
        choices=("base", "tcp"),
        default="base",
        help=(
            "Frame represented by GraspGen grasp matrices. GraspGen franka_panda returns the gripper "
            "base/link frame by default; use tcp only if your server returns tool-center poses."
        ),
    )
    parser.add_argument(
        "--grasp_to_hand_rotation",
        choices=("franka_panda", "identity"),
        default="franka_panda",
        help=(
            "Fixed rotation from GraspGen grasp convention to Isaac panda_hand. "
            "franka_panda maps GraspGen X finger-line to Isaac panda_hand Y finger-line."
        ),
    )
    parser.add_argument(
        "--panda_hand_to_tcp_z",
        type=float,
        default=0.107,
        help="panda_hand -> TCP local-z offset used when --grasp_pose_frame=tcp.",
    )
    parser.add_argument(
        "--grasp_index",
        type=int,
        default=0,
        help="Ranked table-safe GraspGen result to execute after sorting by confidence.",
    )
    parser.add_argument(
        "--approach_axis",
        choices=("x", "y", "z", "-x", "-y", "-z"),
        default="z",
        help="Gripper-local approach axis. Pregrasp is grasp_pos - axis * pregrasp_offset.",
    )
    parser.add_argument("--dry_run_graspgen", action="store_true", help="Skip motion after GraspGen inference.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def _resolve_encoder_checkpoint_for_config(cfg, config: str) -> str | None:
    try:
        resolved = _resolve_initial_encoder_checkpoint(cfg, config_source=config)
    except RuntimeError as exc:
        print(
            "[WARNING] could not resolve pretrained encoder checkpoint for scripted "
            f"GraspGen eval; continuing without one: {exc}",
            flush=True,
        )
        return None
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


def _apply_panda_gripper_overrides(cfg: Any, paths_yaml: str | None, seed: int | None) -> None:
    if paths_yaml is not None:
        cfg.paths_yaml = paths_yaml
    elif Path("configs/paths/panda_hand.yaml").is_file():
        cfg.paths_yaml = "configs/paths/panda_hand.yaml"

    if seed is not None:
        cfg.general.seed = int(seed)

    cfg.rl.isaac_task_id = "tool-unstable-v0"
    cfg.rl.env.robot_mode = "official_panda_gripper"
    cfg.rl.action.action_dim = 8
    cfg.rl.action.joint_names = ["panda_joint.*"]
    cfg.rl.observation.previous_action_dim = 8
    cfg.rl.observation.robot_state_dim = 18
    cfg.rl.observation.tool_cloud_source = "official_panda_gripper_kinematic_mesh"


def _runtime_spec_for_direct_grasp(spec: dict[str, Any], num_envs: int, seed: int | None) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec["mode"] = MODE_NAME
    spec["task_id"] = "tool-unstable-v0"
    spec["num_envs"] = int(num_envs)
    spec["action_dim"] = 8
    if seed is not None:
        spec["seed"] = int(seed)

    env_params = spec.setdefault("env_params", {})
    env_params["num_envs"] = int(num_envs)
    env_params["robot_mode"] = "official_panda_gripper"

    action_params = spec.setdefault("action_params", {})
    action_params["action_dim"] = 8
    action_params["joint_names"] = ["panda_joint.*"]

    obs_params = spec.setdefault("observation_params", {})
    obs_params["previous_action_dim"] = 8
    obs_params["robot_state_dim"] = 18
    obs_params["tool_cloud_source"] = "official_panda_gripper_kinematic_mesh"
    spec["observation_dim"] = _resolved_observation_dim(spec)
    return spec


def _resolved_observation_dim(spec: dict[str, Any]) -> int:
    obs = spec["observation_params"]
    layout = obs.get("layout", spec.get("observation_layout", []))
    physics_dim = int(spec.get("physics_dim", 0))
    action_dim = int(spec.get("action_dim", 8))
    num_points = int(obs.get("num_points", 512))
    point_dim = int(obs.get("point_dim", 3))
    include_cloud = bool(obs.get("include_object_cloud", True))
    include_tool_cloud = bool(obs.get("include_tool_cloud", True))
    include_bbox = bool(obs.get("include_bbox_centers", True))
    prev_dim = int(obs.get("previous_action_dim") or action_dim)
    dims = {
        "object_cloud_flat": num_points * point_dim if include_cloud else 0,
        "tool_cloud_flat": num_points * point_dim if include_tool_cloud else 0,
        "object_bbox_center": 3 if include_bbox else 0,
        "tool_bbox_center": 3 if include_bbox else 0,
        "hand_state": int(obs.get("hand_state_dim", 9)),
        "robot_state": int(obs.get("robot_state_dim", 18)),
        "previous_action": prev_dim,
        "relative_goal_pose": int(obs.get("relative_goal_dim", 9)),
        "object_velocity": int(obs.get("object_velocity_dim", 0)),
        "physics": physics_dim,
    }
    return sum(dims.get(name, 0) for name in layout)


def _build_runtime_spec_from_config(config: str, num_envs: int, seed: int | None, paths_yaml: str | None) -> dict[str, Any]:
    cfg = load_exp_cfg(config)
    _apply_panda_gripper_overrides(cfg, paths_yaml, seed)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifact_dir = Path(tempfile.gettempdir()) / "tool_generalist_graspgen_lift_grasps" / Path(config).stem
    spec = build_rl_runtime_spec(
        cfg,
        paths,
        artifact_dir,
        mode=MODE_NAME,
        encoder_checkpoint_override=_resolve_encoder_checkpoint_for_config(cfg, config),
        runtime_num_envs=num_envs,
    )
    payload = asdict(spec)
    payload["source_config"] = str(config)
    return _runtime_spec_for_direct_grasp(payload, num_envs, seed)


def _load_runtime_spec(path: str, num_envs: int, seed: int | None, paths_yaml: str | None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime spec must be a JSON object: {path}")
    payload = _runtime_spec_for_direct_grasp(payload, num_envs, seed)
    if paths_yaml is not None:
        payload["paths_yaml"] = str(Path(paths_yaml).expanduser().resolve())
    return payload


def _materialize_runtime_spec(spec: dict[str, Any]) -> str:
    spec_dir = Path(tempfile.gettempdir()) / "tool_generalist_graspgen_lift_grasps"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"rl_runtime_spec_{MODE_NAME}_envs_{spec['num_envs']}.json"
    validate_runtime_spec(spec, spec_path)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return str(spec_path)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    return quat / max(float(np.linalg.norm(quat)), 1e-8)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q.tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_matrix_from_pos_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quat_wxyz_to_matrix(quat)
    matrix[:3, 3] = np.asarray(pos, dtype=np.float32)
    return matrix


def _axis_index(axis: str) -> tuple[int, float]:
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[-1]
    return {"x": 0, "y": 1, "z": 2}[name], sign


def _add_graspgen_to_path(graspgen_root: str) -> None:
    root = Path(graspgen_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GraspGen root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _jsonable_tensor(values: Any) -> Any:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return values


args_cli, hydra_args = _parse_args()
if args_cli.num_episodes <= 0:
    raise ValueError(f"--num_episodes must be positive, got {args_cli.num_episodes}")
if args_cli.num_envs <= 0:
    raise ValueError(f"--num_envs must be positive, got {args_cli.num_envs}")
if args_cli.max_candidate_attempts < 0:
    raise ValueError(f"--max_candidate_attempts must be >= 0, got {args_cli.max_candidate_attempts}")
if args_cli.lift_success_stable_seconds < 0:
    raise ValueError(f"--lift_success_stable_seconds must be >= 0, got {args_cli.lift_success_stable_seconds}")
if args_cli.lift_success_stability_tolerance < 0:
    raise ValueError(
        f"--lift_success_stability_tolerance must be >= 0, got {args_cli.lift_success_stability_tolerance}"
    )
if args_cli.worker_id < 0:
    raise ValueError(f"--worker_id must be >= 0, got {args_cli.worker_id}")
if args_cli.num_workers <= 0:
    raise ValueError(f"--num_workers must be positive, got {args_cli.num_workers}")
if args_cli.seed is not None:
    args_cli.seed = int(args_cli.seed) + int(args_cli.worker_id) * int(args_cli.worker_seed_stride)

if args_cli.runtime_spec:
    runtime_spec = _load_runtime_spec(args_cli.runtime_spec, args_cli.num_envs, args_cli.seed, args_cli.paths_yaml)
else:
    runtime_spec = _build_runtime_spec_from_config(args_cli.config, args_cli.num_envs, args_cli.seed, args_cli.paths_yaml)

if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id")
if not args_cli.task:
    raise RuntimeError("--task is required when the runtime spec does not contain task_id")

runtime_spec_path = _materialize_runtime_spec(runtime_spec)
os.environ[RUNTIME_SPEC_ENV_VAR] = runtime_spec_path
os.environ["TOOL_GENERALIST_PATHS_YAML"] = os.path.abspath(os.path.normpath(runtime_spec["paths_yaml"]))
os.environ.setdefault("TOOL_GENERALIST_GLOBAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_LOCAL_RANK", "0")
os.environ.setdefault("TOOL_GENERALIST_WORLD_SIZE", "1")

args_cli.enable_cameras = bool(args_cli.record_video or getattr(args_cli, "enable_cameras", False))
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import (
    apply_delta_pose,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
import IsaacLab_nonPrehensile.tasks
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    OBJECT_ASSET_CFGS,
    get_object_index_for_env,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


def _make_record_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/GraspRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.08, 0.0, 0.60),
            rot=(-0.3337, 0.6234, 0.6234, -0.3337),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=0.80,
            horizontal_aperture=28.0,
            clipping_range=(0.02, 20.0),
        ),
        width=int(args_cli.video_width),
        height=int(args_cli.video_height),
    )


def _start_ffmpeg_writer(path: Path) -> subprocess.Popen:
    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG_PATH}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        f"{int(args_cli.video_width)}x{int(args_cli.video_height)}",
        "-r",
        str(int(args_cli.video_fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _close_ffmpeg_writer(writer: subprocess.Popen) -> None:
    if writer.stdin is not None:
        writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def _init_video_state(output_dir: Path, num_envs: int, episode_idx: int | None = None) -> dict[str, Any] | None:
    if not args_cli.record_video:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if episode_idx is None:
        run_name = f"graspgen_lift_grasps_{timestamp}"
    else:
        run_name = f"episode_{int(episode_idx):04d}_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    paths = []
    for env_id in range(int(num_envs)):
        path = run_dir / f"env_{env_id:03d}.mp4"
        writers.append(_start_ffmpeg_writer(path))
        paths.append(path)
    print(f"[INFO] recording {num_envs} per-env videos to: {run_dir}", flush=True)
    return {
        "run_dir": run_dir,
        "writers": writers,
        "paths": paths,
        "frames": 0,
        "active": [True for _ in range(int(num_envs))],
        "stop_reasons": [None for _ in range(int(num_envs))],
    }


def _capture_video_frames(env, video_state: dict[str, Any] | None) -> None:
    if video_state is None:
        return
    env.unwrapped.sim.render()
    camera = env.unwrapped.scene["grasp_record_camera"]
    camera.update(dt=0.0, force_recompute=True)
    rgb_all = camera.data.output["rgb"]
    max_frames = int(args_cli.video_length)
    if max_frames > 0 and int(video_state["frames"]) >= max_frames:
        return
    for env_id, writer in enumerate(video_state["writers"]):
        if not bool(video_state.get("active", [True])[env_id]) or writer.stdin is None:
            continue
        frame_tensor = rgb_all[env_id, ..., :3].detach().cpu()
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = torch.clamp(frame_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
        frame = frame_tensor.contiguous().numpy()
        writer.stdin.write(frame.tobytes())
    video_state["frames"] = int(video_state["frames"]) + 1


def _stop_video_for_envs(
    video_state: dict[str, Any] | None,
    ended_mask: torch.Tensor | None,
    *,
    reason: str,
) -> None:
    if video_state is None or ended_mask is None:
        return
    ended = ended_mask.detach().cpu().numpy().astype(bool).tolist()
    active = video_state.get("active")
    if active is None:
        return
    for env_id, should_stop in enumerate(ended):
        if not should_stop or not bool(active[env_id]):
            continue
        writer = video_state["writers"][env_id]
        _close_ffmpeg_writer(writer)
        active[env_id] = False
        video_state["stop_reasons"][env_id] = reason
        print(f"[INFO] stopped video env={env_id:03d} reason={reason}: {video_state['paths'][env_id]}", flush=True)


def _close_video_state(video_state: dict[str, Any] | None) -> None:
    if video_state is None:
        return
    errors = []
    for env_id, writer in enumerate(video_state["writers"]):
        try:
            if bool(video_state.get("active", [True])[env_id]):
                _close_ffmpeg_writer(writer)
                video_state["active"][env_id] = False
                if video_state.get("stop_reasons", [None])[env_id] is None:
                    video_state["stop_reasons"][env_id] = "script_end"
        except Exception as exc:
            errors.append(exc)
    for path in video_state["paths"]:
        print(f"[INFO] video: {path}", flush=True)
    if errors:
        raise errors[0]


def _draw_grasp_guide(
    stage,
    prim_prefix: str,
    transform: np.ndarray,
    color: tuple[float, float, float],
    *,
    half_width: float,
    depth: float,
    thickness: float,
    opacity: float = 1.0,
) -> None:
    from pxr import Gf, UsdGeom

    transform = np.asarray(transform, dtype=np.float32)
    origin = transform[:3, 3]
    x_axis = transform[:3, 0]
    y_axis = transform[:3, 1]
    z_axis = transform[:3, 2]
    z0 = -0.25 * depth
    z1 = 0.75 * depth
    zc = 0.5 * (z0 + z1)

    for side_index, side in enumerate((-1.0, 1.0)):
        center = origin + side * half_width * x_axis + zc * z_axis
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 0] = x_axis * thickness
        matrix[:3, 1] = y_axis * thickness
        matrix[:3, 2] = z_axis * (z1 - z0)
        matrix[:3, 3] = center

        cube = UsdGeom.Cube.Define(stage, f"{prim_prefix}/finger_{side_index}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cube.CreateDisplayOpacityAttr([float(opacity)])
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.flatten().tolist()))

    connector_center = origin + z0 * z_axis
    connector_matrix = np.eye(4, dtype=np.float64)
    connector_matrix[:3, 0] = x_axis * (2.0 * half_width + thickness)
    connector_matrix[:3, 1] = y_axis * thickness
    connector_matrix[:3, 2] = z_axis * thickness
    connector_matrix[:3, 3] = connector_center
    connector = UsdGeom.Cube.Define(stage, f"{prim_prefix}/connector")
    connector.CreateSizeAttr(1.0)
    connector.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    connector.CreateDisplayOpacityAttr([float(opacity)])
    xformable = UsdGeom.Xformable(connector.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(Gf.Matrix4d(*connector_matrix.T.flatten().tolist()))


def _visualize_grasp_guides(base_env, records: list[dict[str, Any]]) -> None:
    if args_cli.no_visualize_grasps:
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    half_width = 0.5 * float(args_cli.grasp_line_width)
    depth = float(args_cli.grasp_line_depth)
    thickness = float(args_cli.grasp_line_thickness)
    candidate_thickness = float(args_cli.candidate_grasp_line_thickness)

    if args_cli.visualize_candidate_grasps:
        for record in records:
            env_id = int(record["env_id"])
            candidates = record.get("candidate_grasps") or []
            for candidate in candidates:
                transform = np.asarray(candidate["grasp_matrix_w"], dtype=np.float32)
                safe = candidate.get("table_collision_safe")
                hand_safe = candidate.get("table_hand_collision_safe")
                if safe is True:
                    color = (0.10, 0.48, 1.00)
                    opacity = 0.65
                elif hand_safe is True:
                    color = (1.00, 0.82, 0.10)
                    opacity = 0.62
                elif safe is False:
                    color = (1.00, 0.30, 0.05)
                    opacity = 0.55
                else:
                    color = (0.55, 0.65, 0.75)
                    opacity = 0.45
                rank = int(candidate.get("rank", 0))
                _draw_grasp_guide(
                    stage,
                    f"/Visuals/GraspGen/candidate_grasps/env_{env_id:03d}/rank_{rank:03d}",
                    transform,
                    color,
                    half_width=half_width,
                    depth=depth * 0.82,
                    thickness=candidate_thickness,
                    opacity=opacity,
                )

    for record in records:
        if not bool(record.get("execute_grasp", record.get("status") == "ok")):
            continue
        env_id = int(record["env_id"])
        transform = np.asarray(record["grasp_matrix_w"], dtype=np.float32)
        _draw_grasp_guide(
            stage,
            f"/Visuals/GraspGen/selected_grasp/env_{env_id:03d}",
            transform,
            (0.1, 1.0, 0.1),
            half_width=half_width,
            depth=depth,
            thickness=thickness,
            opacity=1.0,
        )


def _visualize_object_cloud_points(base_env, pointcloud_w: np.ndarray) -> None:
    if args_cli.no_visualize_object_cloud:
        return
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    max_points = max(0, int(args_cli.object_cloud_vis_points))
    if max_points == 0:
        return
    size = float(args_cli.object_cloud_point_size)
    stride = max(1, int(math.ceil(pointcloud_w.shape[1] / max_points)))
    points = pointcloud_w[:, ::stride, :][:, :max_points, :]

    for env_id in range(points.shape[0]):
        for point_id, point in enumerate(points[env_id]):
            if not np.isfinite(point).all():
                continue
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] *= size
            matrix[:3, 3] = point.astype(np.float64)
            prim_path = f"/Visuals/GraspGen/object_cloud/env_{env_id:03d}/pt_{point_id:04d}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.45, 1.0)])
            xformable = UsdGeom.Xformable(cube.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.flatten().tolist()))


def _resample_graspgen_point_cloud(
    pointcloud_w: np.ndarray,
    env_id: int,
) -> tuple[np.ndarray, int]:
    target_num_points = int(args_cli.graspgen_point_cloud_points)
    if target_num_points <= 0:
        raise ValueError(f"--graspgen_point_cloud_points must be > 0, got {target_num_points}")

    pointcloud_w = np.asarray(pointcloud_w, dtype=np.float32)
    finite_mask = np.isfinite(pointcloud_w).all(axis=1)
    finite_points = pointcloud_w[finite_mask]
    finite_count = int(finite_points.shape[0])
    if finite_count == 0:
        return np.full((target_num_points, 3), np.nan, dtype=np.float32), 0
    if finite_count == target_num_points:
        return finite_points.astype(np.float32, copy=False), finite_count

    seed = int(args_cli.seed) if args_cli.seed is not None else 0
    rng = np.random.default_rng(seed + 1009 * int(env_id) + 17 * target_num_points)
    indices = rng.choice(
        finite_count,
        size=target_num_points,
        replace=finite_count < target_num_points,
    )
    return finite_points[indices].astype(np.float32, copy=False), finite_count


def _sample_graspgen_mesh_points(obj_path: str, target_num_points: int, rng: np.random.Generator) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(obj_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        return np.asarray(mesh.sample(target_num_points), dtype=np.float32)

    vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise RuntimeError(f"cannot sample points from mesh without faces/vertices: {obj_path}")
    indices = rng.choice(
        vertices.shape[0],
        size=target_num_points,
        replace=vertices.shape[0] < target_num_points,
    )
    return vertices[indices].astype(np.float32, copy=False)


def _prepare_graspgen_point_clouds(base_env, fallback_pointcloud_w: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target_num_points = int(args_cli.graspgen_point_cloud_points)
    if target_num_points <= 0:
        raise ValueError(f"--graspgen_point_cloud_points must be > 0, got {target_num_points}")
    seed = int(args_cli.seed) if args_cli.seed is not None else 0
    try:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_object_asset_cfg_for_env,
        )

        obj = base_env.scene["object"]
        env_ids = list(range(base_env.num_envs))
        scales = mdp.get_rigid_body_scale(base_env, SceneEntityCfg("object"), env_ids).detach().cpu().numpy()
        positions_w = obj.data.root_pos_w[:, :3].detach().cpu().numpy()
        quats_w = obj.data.root_quat_w.detach().cpu().numpy()
        prepared = []
        stats = []
        for env_id in env_ids:
            rng = np.random.default_rng(seed + 1009 * int(env_id) + 17 * target_num_points)
            obj_path = get_object_asset_cfg_for_env(env_id).obj_path
            local_points = _sample_graspgen_mesh_points(obj_path, target_num_points, rng)
            local_points = local_points * scales[env_id].reshape(1, 3).astype(np.float32)
            rot_w = _quat_wxyz_to_matrix(quats_w[env_id])
            pc_w = local_points @ rot_w.T + positions_w[env_id].reshape(1, 3)
            prepared.append(pc_w.astype(np.float32, copy=False))
            stats.append(
                {
                    "raw_points": int(fallback_pointcloud_w.shape[1]),
                    "finite_points": int(np.isfinite(pc_w).all(axis=1).sum()),
                    "sent_points": int(pc_w.shape[0]),
                    "source": "mesh_sample",
                }
            )
        return np.stack(prepared, axis=0).astype(np.float32, copy=False), stats
    except Exception as exc:
        print(
            f"[WARNING] failed to sample {target_num_points} GraspGen points from mesh; "
            f"falling back to resampling observation cloud: {exc}",
            flush=True,
        )

    prepared = []
    stats = []
    for env_id, pc_w in enumerate(fallback_pointcloud_w):
        pc_resampled, finite_count = _resample_graspgen_point_cloud(pc_w, env_id)
        prepared.append(pc_resampled)
        stats.append(
            {
                "raw_points": int(pc_w.shape[0]),
                "finite_points": int(finite_count),
                "sent_points": int(pc_resampled.shape[0]),
                "source": "observation_resample",
            }
        )
    return np.stack(prepared, axis=0).astype(np.float32, copy=False), stats


def _clamp_vector_norm(values: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(values, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / torch.clamp(norm, min=1e-8), max=1.0)
    return values * scale


def _resolve_fixed_base_jacobian_index(robot, body_id: int) -> int:
    shared_metatype = getattr(getattr(robot, "root_physx_view", None), "shared_metatype", None)
    fixed_base = bool(getattr(shared_metatype, "fixed_base", True))
    return int(body_id) - 1 if fixed_base else int(body_id)


def _action_scale_tensor(device: torch.device) -> torch.Tensor:
    scale = runtime_spec.get("action_params", {}).get("scale", DEFAULT_ACTION_SCALE)
    if isinstance(scale, (list, tuple)):
        values = [float(v) for v in scale]
        if len(values) == 1:
            values = values * 7
        if len(values) != 7:
            raise RuntimeError(f"Expected 7 arm action scales, got {values!r}")
        return torch.tensor(values, dtype=torch.float32, device=device).view(1, 7)
    return torch.full((1, 7), float(scale), dtype=torch.float32, device=device)


def _tcp_offset_tensor(device: torch.device | str, batch_size: int) -> torch.Tensor:
    offset = torch.zeros((int(batch_size), 3), dtype=torch.float32, device=device)
    offset[:, 2] = float(args_cli.panda_hand_to_tcp_z)
    return offset


def _grasp_to_hand_rot_matrix_np() -> np.ndarray:
    if args_cli.grasp_to_hand_rotation == "identity":
        return np.eye(3, dtype=np.float32)
    # GraspGen franka_panda: finger-line is +X, approach/depth is +Z.
    # Isaac panda_hand: finger-line/opening is +Y, approach/depth is +Z.
    return np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _grasp_to_hand_quat_tensor(device: torch.device | str, batch_size: int) -> torch.Tensor:
    if args_cli.grasp_to_hand_rotation == "identity":
        quat = (1.0, 0.0, 0.0, 0.0)
    else:
        half_sqrt = math.sqrt(0.5)
        quat = (half_sqrt, 0.0, 0.0, -half_sqrt)
    return torch.tensor(quat, dtype=torch.float32, device=device).view(1, 4).repeat(int(batch_size), 1)


def _grasp_pose_to_hand_target(pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hand_quat_w = quat_mul(quat_w, _grasp_to_hand_quat_tensor(pos_w.device, pos_w.shape[0]))
    if args_cli.grasp_pose_frame == "base":
        return pos_w, hand_quat_w
    offset_w = quat_apply(hand_quat_w, _tcp_offset_tensor(pos_w.device, pos_w.shape[0]))
    return pos_w - offset_w, hand_quat_w


def _current_hand_pose_w(base_env, env_id: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    robot = base_env.scene["robot"]
    hand_cfg = getattr(base_env, "_graspgen_fallback_hand_cfg", None)
    if hand_cfg is None:
        hand_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
        hand_cfg.resolve(base_env.scene)
        setattr(base_env, "_graspgen_fallback_hand_cfg", hand_cfg)
    state = robot.data.body_state_w[:, hand_cfg.body_ids[0]]
    if env_id is not None:
        state = state[int(env_id) : int(env_id) + 1]
    return state[:, :3], state[:, 3:7]


def _current_grasp_pose_for_noop(base_env, env_id: int) -> tuple[np.ndarray, np.ndarray]:
    hand_pos_w, hand_quat_w = _current_hand_pose_w(base_env, env_id)
    raw_quat_w = quat_mul(hand_quat_w, quat_inv(_grasp_to_hand_quat_tensor(hand_quat_w.device, 1)))
    if args_cli.grasp_pose_frame == "tcp":
        hand_pos_w = hand_pos_w + quat_apply(hand_quat_w, _tcp_offset_tensor(hand_pos_w.device, 1))
    pos = hand_pos_w[0].detach().cpu().numpy().astype(np.float32)
    quat = raw_quat_w[0].detach().cpu().numpy().astype(np.float32)
    return pos, quat


def _box_corner_points(center: tuple[float, float, float], extent: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    hx, hy, hz = (0.5 * float(v) for v in extent)
    return np.asarray(
        [
            [cx + sx * hx, cy + sy * hy, cz + sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )


def _fallback_panda_gripper_proxy_points_h() -> tuple[np.ndarray, np.ndarray]:
    width = 0.10537486
    depth = 0.10527314
    palm = _box_corner_points((0.0, 0.0, 0.005), (0.120, 0.075, 0.050))
    left_finger = _box_corner_points((0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    right_finger = _box_corner_points((-0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    all_points = np.concatenate([palm, left_finger, right_finger], axis=0).astype(np.float32)
    return all_points, palm.astype(np.float32)


def _downsample_proxy_points(points: np.ndarray) -> np.ndarray:
    max_points = max(8, int(args_cli.table_collision_proxy_points))
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _get_gripper_collision_proxy_points_h(base_env) -> dict[str, np.ndarray] | None:
    if args_cli.no_table_collision_filter:
        return None
    cached = getattr(base_env, "_graspgen_table_collision_proxy_points_h", None)
    if cached is not None:
        return cached
    try:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
            _get_official_panda_gripper_bucket_clouds,
        )

        bucket_clouds = _get_official_panda_gripper_bucket_clouds(base_env)
        all_points_h = bucket_clouds[-1].float().detach().cpu().numpy()
        hand_count = max(1, min(all_points_h.shape[0], int(getattr(base_env.cfg, "num_points", 512)) // 2))
        hand_points_h = all_points_h[:hand_count]
        source = "official_panda_gripper_mesh"
    except Exception as exc:
        print(f"[WARNING] failed to load official Panda gripper proxy mesh; using box proxy: {exc}", flush=True)
        all_points_h, hand_points_h = _fallback_panda_gripper_proxy_points_h()
        source = "fallback_box_proxy"
    proxy = {
        "all": _downsample_proxy_points(np.asarray(all_points_h, dtype=np.float32)),
        "hand": _downsample_proxy_points(np.asarray(hand_points_h, dtype=np.float32)),
    }
    setattr(base_env, "_graspgen_table_collision_proxy_points_h", proxy)
    setattr(base_env, "_graspgen_table_collision_proxy_source", source)
    print(
        f"[INFO] table collision proxy source={source} "
        f"all_points={proxy['all'].shape[0]} hand_points={proxy['hand'].shape[0]}",
        flush=True,
    )
    return proxy


def _support_surface_info_w(base_env, env_id: int) -> tuple[float, np.ndarray | None]:
    origin = base_env.scene.env_origins[int(env_id)].detach().cpu().numpy().astype(np.float32)
    if bool(getattr(base_env.cfg, "table_enabled", False)):
        table_pose = np.asarray(getattr(base_env.cfg, "table_pose_xyz", (0.0, 0.0, -0.02)), dtype=np.float32)
        table_size = np.asarray(getattr(base_env.cfg, "table_size_xyz", (1.0, 1.0, 0.04)), dtype=np.float32)
        top_z = float(origin[2] + table_pose[2] + 0.5 * table_size[2])
        half_xy = 0.5 * table_size[:2] + float(args_cli.table_collision_xy_margin)
        center_xy = origin[:2] + table_pose[:2]
        bounds_xy = np.stack((center_xy - half_xy, center_xy + half_xy), axis=0)
        return top_z, bounds_xy.astype(np.float32)
    return 0.0, None


def _candidate_hand_matrix_w(grasp_matrix_w: np.ndarray) -> np.ndarray:
    matrix = np.asarray(grasp_matrix_w, dtype=np.float32).copy()
    matrix[:3, :3] = matrix[:3, :3] @ _grasp_to_hand_rot_matrix_np()
    if args_cli.grasp_pose_frame == "tcp":
        matrix[:3, 3] -= matrix[:3, :3] @ np.asarray([0.0, 0.0, float(args_cli.panda_hand_to_tcp_z)], dtype=np.float32)
    return matrix


def _gripper_table_clearance(
    grasp_matrix_w: np.ndarray,
    proxy_points_h: np.ndarray | None,
    table_top_z_w: float,
    table_bounds_xy_w: np.ndarray | None,
) -> tuple[bool, float, int]:
    if args_cli.no_table_collision_filter or proxy_points_h is None:
        return True, float("inf"), 0
    hand_matrix_w = _candidate_hand_matrix_w(grasp_matrix_w)
    points_w = proxy_points_h @ hand_matrix_w[:3, :3].T + hand_matrix_w[:3, 3]
    finite_mask = np.isfinite(points_w).all(axis=1)
    if table_bounds_xy_w is None:
        footprint_mask = np.ones(points_w.shape[0], dtype=bool)
    else:
        low_xy, high_xy = table_bounds_xy_w
        footprint_mask = (
            (points_w[:, 0] >= low_xy[0])
            & (points_w[:, 0] <= high_xy[0])
            & (points_w[:, 1] >= low_xy[1])
            & (points_w[:, 1] <= high_xy[1])
        )
    checked_mask = finite_mask & footprint_mask
    if not np.any(checked_mask):
        return True, float("inf"), 0
    clearance = points_w[checked_mask, 2] - float(table_top_z_w)
    min_clearance = float(np.min(clearance))
    safe = min_clearance >= float(args_cli.table_collision_clearance)
    return safe, min_clearance, int(np.count_nonzero(checked_mask))


def _upward_alignment_score(grasp_matrix_w: np.ndarray) -> float:
    hand_matrix_w = _candidate_hand_matrix_w(grasp_matrix_w)
    z_axis = hand_matrix_w[:3, 2]
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-8 or not np.isfinite(norm):
        return float("-inf")
    return float(z_axis[2] / norm)


def _select_grasp_candidate(
    grasps: np.ndarray,
    confidences: np.ndarray,
    pc_mean_w: np.ndarray,
    env_id: int,
    base_env,
    proxy_points_h: dict[str, np.ndarray] | None,
) -> tuple[int | None, int | None, np.ndarray | None, dict[str, Any]]:
    order = np.argsort(-np.asarray(confidences))
    table_top_z_w, table_bounds_xy_w = _support_surface_info_w(base_env, env_id)
    full_safe_candidates: list[dict[str, Any]] = []
    hand_safe_candidates: list[dict[str, Any]] = []
    unsafe_candidates: list[dict[str, Any]] = []
    filter_enabled = not bool(args_cli.no_table_collision_filter)
    candidate_records: list[dict[str, Any]] = []
    candidate_vis_limit = max(0, int(args_cli.candidate_grasp_vis_limit))
    all_proxy_h = None if proxy_points_h is None else proxy_points_h.get("all")
    hand_proxy_h = None if proxy_points_h is None else proxy_points_h.get("hand")

    for ranked_i, candidate_i in enumerate(order):
        candidate_i = int(candidate_i)
        grasp = np.asarray(grasps[candidate_i], dtype=np.float32).copy()
        grasp[:3, 3] += pc_mean_w
        full_safe, full_clearance, full_checked_points = _gripper_table_clearance(
            grasp,
            all_proxy_h,
            table_top_z_w,
            table_bounds_xy_w,
        )
        hand_safe, hand_clearance, hand_checked_points = _gripper_table_clearance(
            grasp,
            hand_proxy_h,
            table_top_z_w,
            table_bounds_xy_w,
        )
        upward_score = _upward_alignment_score(grasp)
        entry = {
            "ranked_i": int(ranked_i),
            "candidate_i": int(candidate_i),
            "grasp": grasp,
            "full_safe": bool(full_safe),
            "full_clearance": float(full_clearance),
            "full_checked_points": int(full_checked_points),
            "hand_safe": bool(hand_safe),
            "hand_clearance": float(hand_clearance),
            "hand_checked_points": int(hand_checked_points),
            "upward_score": float(upward_score),
        }
        if full_safe:
            full_safe_candidates.append(entry)
        elif hand_safe:
            hand_safe_candidates.append(entry)
        else:
            unsafe_candidates.append(entry)
        if args_cli.visualize_candidate_grasps and (
            candidate_vis_limit == 0 or len(candidate_records) < candidate_vis_limit
        ):
            candidate_records.append(
                {
                    "rank": int(ranked_i),
                    "candidate_index": int(candidate_i),
                    "confidence": float(confidences[candidate_i]),
                    "table_collision_safe": bool(full_safe),
                    "table_hand_collision_safe": bool(hand_safe),
                    "table_clearance_m": None if not np.isfinite(full_clearance) else float(full_clearance),
                    "table_hand_clearance_m": None if not np.isfinite(hand_clearance) else float(hand_clearance),
                    "table_collision_checked_points": int(full_checked_points),
                    "table_hand_collision_checked_points": int(hand_checked_points),
                    "upward_alignment_score": float(upward_score),
                    "grasp_matrix_w": grasp.tolist(),
                }
            )

    chosen_safe_rank = min(max(int(args_cli.grasp_index), 0), max(len(full_safe_candidates) - 1, 0))
    if full_safe_candidates:
        chosen = full_safe_candidates[chosen_safe_rank]
        status = "ok"
        selection_tier = "full_safe"
    elif hand_safe_candidates:
        chosen_hand_rank = min(max(int(args_cli.grasp_index), 0), len(hand_safe_candidates) - 1)
        chosen = hand_safe_candidates[chosen_hand_rank]
        chosen_safe_rank = None
        status = "hand_safe_finger_collision"
        selection_tier = "hand_safe"
    elif unsafe_candidates:
        chosen = max(unsafe_candidates, key=lambda item: item["upward_score"])
        chosen_safe_rank = None
        status = "upward_fallback"
        selection_tier = "most_upward"
    else:
        chosen = None
        chosen_safe_rank = None
        status = "fallback_noop"
        selection_tier = "none"

    if chosen is None:
        ranked_i = candidate_i = None
        grasp = None
        full_safe = hand_safe = False
        full_clearance = hand_clearance = float("-inf")
        full_checked_points = hand_checked_points = 0
        upward_score = float("-inf")
    else:
        ranked_i = int(chosen["ranked_i"])
        candidate_i = int(chosen["candidate_i"])
        grasp = chosen["grasp"]
        full_safe = bool(chosen["full_safe"])
        hand_safe = bool(chosen["hand_safe"])
        full_clearance = float(chosen["full_clearance"])
        hand_clearance = float(chosen["hand_clearance"])
        full_checked_points = int(chosen["full_checked_points"])
        hand_checked_points = int(chosen["hand_checked_points"])
        upward_score = float(chosen["upward_score"])

    info = {
        "status": status,
        "selection_tier": selection_tier,
        "table_collision_filter_enabled": filter_enabled,
        "table_collision_safe": bool(full_safe),
        "table_hand_collision_safe": bool(hand_safe),
        "table_clearance_m": None if not np.isfinite(full_clearance) else float(full_clearance),
        "table_hand_clearance_m": None if not np.isfinite(hand_clearance) else float(hand_clearance),
        "table_collision_clearance_required_m": float(args_cli.table_collision_clearance),
        "table_collision_checked_points": int(full_checked_points),
        "table_hand_collision_checked_points": int(hand_checked_points),
        "table_top_z_w": float(table_top_z_w),
        "table_bounds_xy_w": None if table_bounds_xy_w is None else table_bounds_xy_w.astype(np.float32).tolist(),
        "table_safe_candidates": int(len(full_safe_candidates)),
        "table_full_safe_candidates": int(len(full_safe_candidates)),
        "table_hand_safe_candidates": int(len(hand_safe_candidates)),
        "table_unsafe_candidates": int(len(unsafe_candidates)),
        "chosen_safe_rank": None if chosen_safe_rank is None else int(chosen_safe_rank),
        "upward_alignment_score": None if not np.isfinite(upward_score) else float(upward_score),
    }
    if args_cli.visualize_candidate_grasps:
        info["candidate_grasps"] = candidate_records
    return ranked_i, candidate_i, grasp, info


def _rank_lift_grasp_candidates(
    grasps: np.ndarray,
    confidences: np.ndarray,
    pc_mean_w: np.ndarray,
    env_id: int,
    base_env,
    proxy_points_h: dict[str, np.ndarray] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return all full-safe and hand-safe candidates in eval-consistent order."""

    order = np.argsort(-np.asarray(confidences))
    table_top_z_w, table_bounds_xy_w = _support_surface_info_w(base_env, env_id)
    all_proxy_h = None if proxy_points_h is None else proxy_points_h.get("all")
    hand_proxy_h = None if proxy_points_h is None else proxy_points_h.get("hand")
    all_candidates: list[dict[str, Any]] = []
    lift_candidates: list[dict[str, Any]] = []
    full_safe_count = 0
    hand_safe_count = 0
    unsafe_count = 0

    for ranked_i, candidate_i in enumerate(order):
        candidate_i = int(candidate_i)
        grasp = np.asarray(grasps[candidate_i], dtype=np.float32).copy()
        grasp[:3, 3] += pc_mean_w
        full_safe, full_clearance, full_checked_points = _gripper_table_clearance(
            grasp,
            all_proxy_h,
            table_top_z_w,
            table_bounds_xy_w,
        )
        hand_safe, hand_clearance, hand_checked_points = _gripper_table_clearance(
            grasp,
            hand_proxy_h,
            table_top_z_w,
            table_bounds_xy_w,
        )
        upward_score = _upward_alignment_score(grasp)
        if full_safe:
            tier = "full_safe"
            full_safe_count += 1
        elif hand_safe:
            tier = "hand_safe"
            hand_safe_count += 1
        else:
            tier = "unsafe"
            unsafe_count += 1

        record = {
            "rank": int(ranked_i),
            "candidate_index": int(candidate_i),
            "confidence": float(confidences[candidate_i]),
            "selection_tier": tier,
            "table_collision_safe": bool(full_safe),
            "table_hand_collision_safe": bool(hand_safe),
            "table_clearance_m": None if not np.isfinite(full_clearance) else float(full_clearance),
            "table_hand_clearance_m": None if not np.isfinite(hand_clearance) else float(hand_clearance),
            "table_collision_checked_points": int(full_checked_points),
            "table_hand_collision_checked_points": int(hand_checked_points),
            "upward_alignment_score": None if not np.isfinite(upward_score) else float(upward_score),
            "grasp_matrix_w": grasp.astype(np.float32).tolist(),
            "grasp_quat_wxyz": _matrix_to_quat_wxyz(grasp[:3, :3]).astype(np.float32).tolist(),
        }
        all_candidates.append(record)
        if full_safe or hand_safe:
            lift_candidates.append(record)

    max_attempts = int(args_cli.max_candidate_attempts)
    if max_attempts > 0:
        lift_candidates = lift_candidates[:max_attempts]

    stats = {
        "num_returned": int(len(grasps)),
        "full_safe_candidates": int(full_safe_count),
        "hand_safe_candidates": int(hand_safe_count),
        "unsafe_candidates": int(unsafe_count),
        "attemptable_candidates": int(len(lift_candidates)),
        "max_candidate_attempts": int(max_attempts),
        "table_collision_filter_enabled": not bool(args_cli.no_table_collision_filter),
        "table_top_z_w": float(table_top_z_w),
        "table_bounds_xy_w": None if table_bounds_xy_w is None else table_bounds_xy_w.astype(np.float32).tolist(),
    }
    return lift_candidates, {"stats": stats, "all_candidates": all_candidates}


def _diff_ik_controller(base_env) -> DifferentialIKController:
    cache_key = (
        int(base_env.num_envs),
        str(base_env.device),
        str(args_cli.ik_method),
        float(args_cli.ik_damping),
    )
    controller = getattr(base_env, "_graspgen_diff_ik_controller", None)
    if controller is None or getattr(base_env, "_graspgen_diff_ik_controller_key", None) != cache_key:
        ik_params = {"lambda_val": float(args_cli.ik_damping)} if args_cli.ik_method == "dls" else None
        cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=args_cli.ik_method,
            ik_params=ik_params,
        )
        controller = DifferentialIKController(cfg=cfg, num_envs=int(base_env.num_envs), device=str(base_env.device))
        setattr(base_env, "_graspgen_diff_ik_controller", controller)
        setattr(base_env, "_graspgen_diff_ik_controller_key", cache_key)
    return controller


def _compute_hand_pose_b(robot, ee_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_state_w[:, ee_body_id]
    return subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )


def _compute_body_jacobian_b(robot, arm_cfg: SceneEntityCfg, jacobian_body_id: int) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_body_id, :, arm_cfg.joint_ids].clone()
    base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _compute_ik_joint_targets(
    base_env,
    arm_cfg: SceneEntityCfg,
    ee_body_id: int,
    jacobian_body_id: int,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    robot = base_env.scene["robot"]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = _compute_hand_pose_b(robot, ee_body_id)
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        target_pos_w,
        target_quat_w,
    )

    pos_error, rot_error = compute_pose_error(
        ee_pos_b,
        ee_quat_b,
        target_pos_b,
        target_quat_b,
        rot_error_type="axis_angle",
    )
    dpose = torch.cat(
        (
            _clamp_vector_norm(pos_error, args_cli.max_pos_step),
            _clamp_vector_norm(rot_error, args_cli.max_rot_step),
        ),
        dim=-1,
    )
    command_pos_b, command_quat_b = apply_delta_pose(ee_pos_b, ee_quat_b, dpose)
    controller = _diff_ik_controller(base_env)
    controller.set_command(torch.cat((command_pos_b, command_quat_b), dim=-1))

    jacobian = _compute_body_jacobian_b(robot, arm_cfg, jacobian_body_id)
    q_target = controller.compute(
        ee_pos_b,
        ee_quat_b,
        jacobian,
        robot.data.joint_pos[:, arm_cfg.joint_ids],
    )

    if hasattr(base_env.cfg, "enforce_joint_limits"):
        limits = robot.data.soft_joint_pos_limits[:, arm_cfg.joint_ids]
        q_target = torch.minimum(torch.maximum(q_target, limits[..., 0]), limits[..., 1])
    return q_target


def _step_arm_to_pose(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
    finger_open_action: float,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    *,
    steps: int,
    label: str,
) -> None:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    ee_body_id = arm_cfg.body_ids[0]
    jacobian_body_id = _resolve_fixed_base_jacobian_index(robot, ee_body_id)
    action_scale = _action_scale_tensor(base_env.device)

    for step in range(int(steps)):
        q_target = _compute_ik_joint_targets(
            base_env,
            arm_cfg,
            ee_body_id,
            jacobian_body_id,
            target_pos_w,
            target_quat_w,
        )
        q_current = robot.data.joint_pos[:, arm_cfg.joint_ids]
        arm_action = torch.clamp((q_target - q_current) / action_scale, -1.0, 1.0)
        action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)
        action[:, :7] = arm_action
        action[:, 7] = float(finger_open_action)
        _, _, terminated, truncated, _ = env.step(action)
        _record_success_state(base_env, success_tracker, debug_tracker, terminated, truncated, label)
        _stop_video_for_envs(video_state, terminated | truncated, reason=label)
        _capture_video_frames(env, video_state)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            ee_state_w = robot.data.body_state_w[:, ee_body_id]
            pos_err, rot_err = compute_pose_error(
                ee_state_w[:, :3],
                ee_state_w[:, 3:7],
                target_pos_w,
                target_quat_w,
                rot_error_type="axis_angle",
            )
            print(
                f"[MOTION] {label} step={step + 1}/{steps} "
                f"max_pos_err={float(torch.linalg.norm(pos_err, dim=-1).max().detach().cpu()):.4f} "
                f"max_rot_err={float(torch.linalg.norm(rot_err, dim=-1).max().detach().cpu()):.4f}",
                flush=True,
            )
        if bool(torch.any(terminated | truncated).detach().cpu()):
            print(f"[WARNING] termination/truncation observed during {label}; continuing without reset", flush=True)


def _pose_w_to_robot_base(
    base_env,
    env_id: int,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = base_env.scene["robot"]
    env_slice = slice(int(env_id), int(env_id) + 1)
    pos_w = pos_w.reshape(1, 3)
    quat_w = quat_w.reshape(1, 4)
    return subtract_frame_transforms(
        robot.data.root_pos_w[env_slice, :3],
        robot.data.root_quat_w[env_slice],
        pos_w,
        quat_w,
    )


def _points_w_to_robot_base(base_env, env_id: int, points_w: torch.Tensor) -> torch.Tensor:
    robot = base_env.scene["robot"]
    root_pos_w = robot.data.root_pos_w[int(env_id), :3].view(1, 3)
    root_quat_inv_w = quat_inv(robot.data.root_quat_w[int(env_id)].view(1, 4))
    return quat_apply(root_quat_inv_w.repeat(points_w.shape[0], 1), points_w - root_pos_w)


def _curobo_arm_joint_names(arm_cfg: SceneEntityCfg) -> list[str]:
    expected = [f"panda_joint{i}" for i in range(1, 8)]
    resolved = list(getattr(arm_cfg, "joint_names", []) or [])
    if set(resolved) == set(expected):
        return resolved
    return expected


def _curobo_motion_gen(base_env):
    cache_key = (
        str(base_env.device),
        str(args_cli.curobo_robot_config),
        float(args_cli.curobo_interpolation_dt),
        int(args_cli.curobo_trajopt_tsteps),
        int(args_cli.curobo_interpolation_steps),
        int(args_cli.curobo_ik_seeds),
        int(args_cli.curobo_trajopt_seeds),
        int(args_cli.curobo_graph_seeds),
        int(args_cli.curobo_grad_trajopt_iters),
        float(args_cli.curobo_collision_activation_distance),
    )
    cached = getattr(base_env, "_graspgen_curobo_motion_gen", None)
    if cached is not None and getattr(base_env, "_graspgen_curobo_motion_gen_key", None) == cache_key:
        return cached

    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType(device=torch.device(str(base_env.device)))
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        args_cli.curobo_robot_config,
        WorldConfig(),
        tensor_args,
        trajopt_tsteps=int(args_cli.curobo_trajopt_tsteps),
        interpolation_steps=int(args_cli.curobo_interpolation_steps),
        num_ik_seeds=int(args_cli.curobo_ik_seeds),
        num_trajopt_seeds=int(args_cli.curobo_trajopt_seeds),
        num_graph_seeds=int(args_cli.curobo_graph_seeds),
        grad_trajopt_iters=int(args_cli.curobo_grad_trajopt_iters),
        interpolation_dt=float(args_cli.curobo_interpolation_dt),
        evaluate_interpolated_trajectory=True,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": 4},
        collision_activation_distance=float(args_cli.curobo_collision_activation_distance),
    )
    motion_gen = MotionGen(motion_gen_cfg)
    print("[CUROBO] warming up MotionGen", flush=True)
    motion_gen.warmup()
    setattr(base_env, "_graspgen_curobo_motion_gen", motion_gen)
    setattr(base_env, "_graspgen_curobo_motion_gen_key", cache_key)
    return motion_gen


def _curobo_table_cuboid_b(base_env, env_id: int):
    if bool(args_cli.no_curobo_table_collision):
        return None
    from curobo.geom.types import Cuboid

    origin = base_env.scene.env_origins[int(env_id)].to(base_env.device)
    if bool(getattr(base_env.cfg, "table_enabled", False)):
        table_pose = torch.tensor(
            getattr(base_env.cfg, "table_pose_xyz", (0.0, 0.0, -0.02)),
            dtype=torch.float32,
            device=base_env.device,
        )
        dims = np.asarray(getattr(base_env.cfg, "table_size_xyz", (1.0, 1.0, 0.04)), dtype=np.float32)
    else:
        table_pose = torch.tensor((0.0, 0.0, -0.02), dtype=torch.float32, device=base_env.device)
        dims = np.asarray((5.0, 5.0, 0.04), dtype=np.float32)
    pad = max(0.0, float(args_cli.curobo_table_padding))
    dims = np.maximum(dims + 2.0 * pad, 1e-4).astype(np.float32)
    pos_w = (origin[:3] + table_pose).view(1, 3)
    quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=base_env.device)
    pos_b, quat_b = _pose_w_to_robot_base(base_env, env_id, pos_w, quat_w)
    pose = pos_b[0].detach().cpu().numpy().astype(np.float32).tolist()
    pose += quat_b[0].detach().cpu().numpy().astype(np.float32).tolist()
    return Cuboid(name=f"table_env_{env_id}", pose=pose, dims=dims.tolist())


def _curobo_object_cuboid_b(base_env, env_id: int):
    if bool(args_cli.no_curobo_object_collision):
        return None
    from curobo.geom.types import Cuboid

    pointcloud_flat = mdp.get_object_pointcloud(base_env, SceneEntityCfg("object"))
    points_w = pointcloud_flat.view(base_env.num_envs, -1, 3)[int(env_id)].float()
    finite_mask = torch.isfinite(points_w).all(dim=-1)
    points_w = points_w[finite_mask]
    if points_w.shape[0] < 8:
        print(f"[CUROBO][WARNING] env={env_id} object cloud too small; skipping object obstacle", flush=True)
        return None

    points_b = _points_w_to_robot_base(base_env, env_id, points_w)
    low = torch.min(points_b, dim=0).values
    high = torch.max(points_b, dim=0).values
    pad = max(0.0, float(args_cli.curobo_object_padding))
    center = 0.5 * (low + high)
    dims = torch.clamp(high - low + 2.0 * pad, min=1e-4)
    pose = center.detach().cpu().numpy().astype(np.float32).tolist() + [1.0, 0.0, 0.0, 0.0]
    return Cuboid(
        name=f"object_aabb_env_{env_id}",
        pose=pose,
        dims=dims.detach().cpu().numpy().astype(np.float32).tolist(),
    )


def _curobo_world_for_env(base_env, env_id: int, *, include_object_collision: bool):
    from curobo.geom.types import WorldConfig

    cuboids = []
    table = _curobo_table_cuboid_b(base_env, env_id)
    if table is not None:
        cuboids.append(table)
    if include_object_collision:
        obj = _curobo_object_cuboid_b(base_env, env_id)
        if obj is not None:
            cuboids.append(obj)
    return WorldConfig(cuboid=cuboids)


def _plan_curobo_pose_for_env(
    base_env,
    arm_cfg: SceneEntityCfg,
    env_id: int,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    *,
    label: str,
    include_object_collision: bool,
) -> torch.Tensor:
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    motion_gen = _curobo_motion_gen(base_env)
    world = _curobo_world_for_env(
        base_env,
        env_id,
        include_object_collision=include_object_collision and not bool(args_cli.no_curobo_object_collision),
    )
    motion_gen.world_coll_checker.clear_cache()
    motion_gen.reset(reset_seed=False)
    motion_gen.update_world(world)

    joint_names = _curobo_arm_joint_names(arm_cfg)
    robot = base_env.scene["robot"]
    q_start = robot.data.joint_pos[int(env_id) : int(env_id) + 1, arm_cfg.joint_ids].detach().clone()
    start_state = JointState.from_position(q_start.to(base_env.device), joint_names=joint_names)
    target_pos_b, target_quat_b = _pose_w_to_robot_base(
        base_env,
        env_id,
        target_pos_w[int(env_id) : int(env_id) + 1],
        target_quat_w[int(env_id) : int(env_id) + 1],
    )
    goal_pose = Pose(position=target_pos_b, quaternion=target_quat_b)
    plan_cfg = MotionGenPlanConfig(
        max_attempts=int(args_cli.curobo_max_attempts),
        timeout=float(args_cli.curobo_timeout),
    )
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    success = bool(result.success.detach().view(-1)[0].cpu().item())
    status = getattr(result, "status", None)
    print(
        f"[CUROBO] {label} env={env_id} success={success} status={status} "
        f"solve_time={float(getattr(result, 'total_time', 0.0)):.3f}s obstacles={len(world)}",
        flush=True,
    )
    if not success:
        raise RuntimeError(f"cuRobo failed to plan {label} for env {env_id}: {status}")
    trajectory = result.get_interpolated_plan().get_ordered_joint_state(joint_names)
    return trajectory.position.detach().to(base_env.device).float()


def _step_curobo_pose(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
    finger_open_action: float,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    *,
    label: str,
    include_object_collision: bool,
) -> bool:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    ee_body_id = arm_cfg.body_ids[0]
    jacobian_body_id = _resolve_fixed_base_jacobian_index(robot, ee_body_id)
    action_scale = _action_scale_tensor(base_env.device)
    plans: dict[int, torch.Tensor] = {}
    failures: dict[int, str] = {}
    for env_id in range(int(base_env.num_envs)):
        try:
            plans[env_id] = _plan_curobo_pose_for_env(
                base_env,
                arm_cfg,
                env_id,
                target_pos_w,
                target_quat_w,
                label=label,
                include_object_collision=include_object_collision,
            )
        except Exception as exc:
            failures[env_id] = str(exc)
            print(f"[CUROBO][WARNING] env={env_id} {exc}; using differential IK for this env", flush=True)

    success_count = len(plans)
    total_count = int(base_env.num_envs)
    print(
        f"[CUROBO] {label} planned {success_count}/{total_count} envs; "
        f"fallback_ik={len(failures)}",
        flush=True,
    )
    if failures and bool(args_cli.fail_on_curobo_error):
        first_env = min(failures)
        raise RuntimeError(
            f"cuRobo failed for {len(failures)}/{total_count} envs; "
            f"first env {first_env}: {failures[first_env]}"
        )
    if not plans:
        print(f"[CUROBO][WARNING] no envs planned with cuRobo; falling back to batch differential IK", flush=True)
        return False

    max_len = max(int(plan.shape[0]) for plan in plans.values())
    for step in range(max_len):
        q_target = _compute_ik_joint_targets(
            base_env,
            arm_cfg,
            ee_body_id,
            jacobian_body_id,
            target_pos_w,
            target_quat_w,
        )
        for env_id, plan in plans.items():
            idx = min(step, int(plan.shape[0]) - 1)
            q_target[env_id] = plan[idx]
        q_current = robot.data.joint_pos[:, arm_cfg.joint_ids]
        arm_action = torch.clamp((q_target - q_current) / action_scale, -1.0, 1.0)
        action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)
        action[:, :7] = arm_action
        action[:, 7] = float(finger_open_action)
        _, _, terminated, truncated, _ = env.step(action)
        _record_success_state(base_env, success_tracker, debug_tracker, terminated, truncated, label)
        _stop_video_for_envs(video_state, terminated | truncated, reason=label)
        _capture_video_frames(env, video_state)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == max_len:
            print(f"[MOTION][CUROBO] {label} step={step + 1}/{max_len}", flush=True)
        if bool(torch.any(terminated | truncated).detach().cpu()):
            print(f"[WARNING] termination/truncation observed during {label}; continuing without reset", flush=True)
    return True


def _step_curobo_pregrasp(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
    finger_open_action: float,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    *,
    label: str,
) -> bool:
    return _step_curobo_pose(
        env,
        arm_cfg,
        video_state,
        success_tracker,
        debug_tracker,
        finger_open_action,
        target_pos_w,
        target_quat_w,
        label=label,
        include_object_collision=True,
    )


def _target_object_pose_w(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    command = base_env.command_manager.get_command("target_object_pose")
    target_pos_w = command[:, :3] + base_env.scene.env_origins
    target_pos_w = target_pos_w.clone()
    target_pos_w[:, 2] += float(args_cli.curobo_goal_object_z_offset)
    target_quat_w = command[:, 3:7].clone()
    target_quat_w = target_quat_w / torch.clamp(torch.linalg.norm(target_quat_w, dim=-1, keepdim=True), min=1e-8)
    return target_pos_w, target_quat_w


def _target_hand_pose_for_object_goal(base_env, arm_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    obj = base_env.scene["object"]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_quat_w = obj.data.root_quat_w
    hand_pos_w, hand_quat_w = _current_hand_pose_w(base_env)
    object_to_hand_pos, object_to_hand_quat = subtract_frame_transforms(
        obj_pos_w,
        obj_quat_w,
        hand_pos_w,
        hand_quat_w,
    )
    target_obj_pos_w, target_obj_quat_w = _target_object_pose_w(base_env)
    target_hand_pos_w = target_obj_pos_w + quat_apply(target_obj_quat_w, object_to_hand_pos)
    target_hand_quat_w = quat_mul(target_obj_quat_w, object_to_hand_quat)
    target_hand_quat_w = target_hand_quat_w / torch.clamp(
        torch.linalg.norm(target_hand_quat_w, dim=-1, keepdim=True),
        min=1e-8,
    )
    return target_hand_pos_w, target_hand_quat_w


def _step_curobo_goal_transport(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
) -> bool:
    base_env = env.unwrapped
    target_hand_pos_w, target_hand_quat_w = _target_hand_pose_for_object_goal(base_env, arm_cfg)
    used_curobo = _step_curobo_pose(
        env,
        arm_cfg,
        video_state,
        success_tracker,
        debug_tracker,
        -1.0,
        target_hand_pos_w,
        target_hand_quat_w,
        label="transport_to_goal",
        include_object_collision=False,
    )
    if not used_curobo:
        _step_arm_to_pose(
            env,
            arm_cfg,
            video_state,
            success_tracker,
            debug_tracker,
            -1.0,
            target_hand_pos_w,
            target_hand_quat_w,
            steps=args_cli.goal_transport_steps,
            label="transport_to_goal",
        )
    return used_curobo


def _step_diff_ik_goal_transport(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
) -> None:
    base_env = env.unwrapped
    target_hand_pos_w, target_hand_quat_w = _target_hand_pose_for_object_goal(base_env, arm_cfg)
    _step_arm_to_pose(
        env,
        arm_cfg,
        video_state,
        success_tracker,
        debug_tracker,
        -1.0,
        target_hand_pos_w,
        target_hand_quat_w,
        steps=args_cli.goal_transport_steps,
        label="transport_to_goal",
    )


def _step_hold(
    env,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
    arm_open_action: float,
    steps: int,
    label: str,
) -> None:
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
    action[:, 7] = float(arm_open_action)
    for step in range(int(steps)):
        _, _, terminated, truncated, _ = env.step(action)
        _record_success_state(env.unwrapped, success_tracker, debug_tracker, terminated, truncated, label)
        _stop_video_for_envs(video_state, terminated | truncated, reason=label)
        _capture_video_frames(env, video_state)
        if step == 0 or step + 1 == steps:
            print(f"[MOTION] {label} step={step + 1}/{steps}", flush=True)


def _runtime_step_dt_seconds() -> float | None:
    env_params = runtime_spec.get("env_params", {})
    try:
        sim_dt = float(env_params["sim_dt"])
        decimation = int(env_params["decimation"])
    except (KeyError, TypeError, ValueError):
        return None
    step_dt = sim_dt * float(decimation)
    if step_dt <= 0.0:
        return None
    return step_dt


def _stable_hold_steps() -> int:
    stable_seconds = max(0.0, float(args_cli.lift_success_stable_seconds))
    base_steps = max(0, int(args_cli.hold_steps))
    if stable_seconds <= 0.0:
        return base_steps
    step_dt = _runtime_step_dt_seconds()
    if step_dt is None:
        return base_steps
    return max(base_steps, int(math.ceil(stable_seconds / step_dt)))


def _step_hold_measure_object_z(
    env,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None,
    arm_open_action: float,
    steps: int,
    label: str,
) -> dict[str, Any]:
    base_env = env.unwrapped
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)
    action[:, 7] = float(arm_open_action)
    min_object_z = base_env.scene["object"].data.root_pos_w[:, 2].detach().clone()
    for step in range(int(steps)):
        _, _, terminated, truncated, _ = env.step(action)
        current_z = base_env.scene["object"].data.root_pos_w[:, 2].detach()
        min_object_z = torch.minimum(min_object_z, current_z)
        _record_success_state(base_env, success_tracker, debug_tracker, terminated, truncated, label)
        _stop_video_for_envs(video_state, terminated | truncated, reason=label)
        _capture_video_frames(env, video_state)
        if step == 0 or step + 1 == steps:
            print(f"[MOTION] {label} step={step + 1}/{steps}", flush=True)
    final_object_z = base_env.scene["object"].data.root_pos_w[:, 2].detach().clone()
    return {
        "steps": int(steps),
        "seconds": None if _runtime_step_dt_seconds() is None else float(steps) * float(_runtime_step_dt_seconds()),
        "min_object_z_w": min_object_z,
        "final_object_z_w": final_object_z,
    }


def _runtime_max_episode_steps() -> int | None:
    step_dt = _runtime_step_dt_seconds()
    env_params = runtime_spec.get("env_params", {})
    try:
        episode_length_s = float(env_params["episode_length_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if step_dt is None:
        return None
    return max(1, int(math.ceil(episode_length_s / step_dt)))


def _step_until_episode_end(
    env,
    video_state: dict[str, Any] | None,
    success_tracker: torch.Tensor,
    debug_tracker: dict[str, Any],
) -> None:
    ended_mask = debug_tracker["ended_mask"]
    if bool(torch.all(ended_mask).detach().cpu()):
        return
    max_episode_steps = _runtime_max_episode_steps()
    if max_episode_steps is None:
        print("[WARNING] cannot infer max episode steps; not extending video to episode termination", flush=True)
        return
    current_steps = int(torch.max(debug_tracker["step_counts"]).detach().cpu().item())
    remaining = max(0, max_episode_steps - current_steps + 1)
    if remaining <= 0:
        return
    print(
        f"[MOTION] awaiting per-env episode termination remaining_steps<={remaining} "
        f"active_envs={int((~ended_mask).sum().detach().cpu().item())}",
        flush=True,
    )
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
    action[:, 7] = -1.0
    for step in range(remaining):
        if bool(torch.all(ended_mask).detach().cpu()):
            break
        _, _, terminated, truncated, _ = env.step(action)
        _record_success_state(env.unwrapped, success_tracker, debug_tracker, terminated, truncated, "await_episode_end")
        _stop_video_for_envs(video_state, terminated | truncated, reason="await_episode_end")
        _capture_video_frames(env, video_state)
        if step == 0 or (step + 1) % 50 == 0:
            print(
                f"[MOTION] await_episode_end step={step + 1}/{remaining} "
                f"active_envs={int((~ended_mask).sum().detach().cpu().item())}",
                flush=True,
            )


def _infer_grasps_with_retry(client, point_cloud: np.ndarray, env_id: int) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    attempts = [bool(args_cli.remove_outliers)]
    if attempts[0]:
        attempts.append(False)
    last_error = None
    for remove_outliers in attempts:
        try:
            grasps, confidences = client.infer(
                point_cloud,
                grasp_threshold=args_cli.grasp_threshold,
                num_grasps=args_cli.num_grasps,
                topk_num_grasps=args_cli.topk_num_grasps,
                min_grasps=args_cli.min_grasps,
                max_tries=args_cli.max_tries,
                remove_outliers=remove_outliers,
            )
            if len(grasps) == 0:
                raise RuntimeError("GraspGen returned no grasps")
            return grasps, confidences, remove_outliers, None
        except Exception as exc:
            last_error = str(exc)
            print(
                f"[WARNING] GraspGen failed env={env_id} remove_outliers={remove_outliers}: {exc}",
                flush=True,
            )
    return (
        np.empty((0, 4, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        False,
        last_error or "unknown GraspGen error",
    )


def _fallback_grasp_from_current_hand(base_env, env_id: int) -> tuple[np.ndarray, np.ndarray]:
    return _current_grasp_pose_for_noop(base_env, env_id)


def _infer_grasps(base_env) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    _add_graspgen_to_path(args_cli.graspgen_root)
    from grasp_gen.serving.zmq_client import GraspGenClient

    pointcloud_flat = mdp.get_object_pointcloud(base_env, SceneEntityCfg("object"))
    raw_pointcloud_w = pointcloud_flat.view(base_env.num_envs, -1, 3).float().detach().cpu().numpy()
    pointcloud_w, pointcloud_stats = _prepare_graspgen_point_clouds(base_env, raw_pointcloud_w)
    _visualize_object_cloud_points(base_env, pointcloud_w)
    finite_counts = [item["finite_points"] for item in pointcloud_stats]
    sent_counts = [item["sent_points"] for item in pointcloud_stats]
    print(
        "[GRASPGEN] object pointcloud "
        f"source={pointcloud_stats[0]['source']} "
        f"raw_points={raw_pointcloud_w.shape[1]} "
        f"finite_range=({min(finite_counts)}, {max(finite_counts)}) "
        f"sent_points={sent_counts[0] if len(set(sent_counts)) == 1 else sent_counts}",
        flush=True,
    )
    proxy_points_h = _get_gripper_collision_proxy_points_h(base_env)
    grasp_pos_w = []
    grasp_quat_w = []
    records = []
    axis_idx, axis_sign = _axis_index(args_cli.approach_axis)

    with GraspGenClient(
        host=args_cli.graspgen_host,
        port=args_cli.graspgen_port,
        timeout_ms=args_cli.graspgen_timeout_ms,
    ) as client:
        print(f"[GRASPGEN] connected metadata={client.server_metadata}", flush=True)
        for env_id, pc_w in enumerate(pointcloud_w):
            finite_mask = np.isfinite(pc_w).all(axis=1)
            pc_finite_w = pc_w[finite_mask]
            if pc_finite_w.shape[0] == 0:
                error = "object point cloud has no finite points"
                grasps = np.empty((0, 4, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
                used_remove_outliers = False
                pc_mean = np.zeros((1, 3), dtype=np.float32)
            else:
                pc_mean = pc_finite_w.mean(axis=0, keepdims=True)
                pc_centered = (pc_finite_w - pc_mean).astype(np.float32)
                grasps, confidences, used_remove_outliers, error = _infer_grasps_with_retry(
                    client, pc_centered, env_id
                )
            if len(grasps) == 0:
                if args_cli.fail_on_graspgen_error:
                    raise RuntimeError(f"GraspGen failed for env {env_id}: {error}")
                fallback_pos, fallback_quat = _fallback_grasp_from_current_hand(base_env, env_id)
                grasp = _pose_matrix_from_pos_quat(fallback_pos, fallback_quat)
                grasp_pos_w.append(fallback_pos)
                grasp_quat_w.append(fallback_quat)
                records.append(
                    {
                        "env_id": env_id,
                        "status": "fallback_noop",
                        "error": error,
                        "chosen_index": None,
                        "chosen_rank": None,
                        "execute_grasp": False,
                        "confidence": None,
                        "grasp_pose_frame": args_cli.grasp_pose_frame,
                        "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
                        "grasp_matrix_w": grasp.tolist(),
                        "grasp_quat_wxyz": fallback_quat.tolist(),
                        "approach_dir_w": [0.0, 0.0, 0.0],
                        "pointcloud_mean_w": pc_mean[0].astype(np.float32).tolist(),
                        "pointcloud_source": pointcloud_stats[env_id]["source"],
                        "pointcloud_raw_points": pointcloud_stats[env_id]["raw_points"],
                        "pointcloud_finite_points": pointcloud_stats[env_id]["finite_points"],
                        "pointcloud_sent_points": int(pc_finite_w.shape[0]),
                        "num_returned": 0,
                    }
                )
                print(f"[WARNING] env={env_id} using no-op fallback grasp after GraspGen failure", flush=True)
                continue
            chosen_rank, chosen_i, grasp, filter_info = _select_grasp_candidate(
                grasps,
                np.asarray(confidences),
                pc_mean[0],
                env_id,
                base_env,
                proxy_points_h,
            )
            if grasp is None or chosen_i is None:
                fallback_pos, fallback_quat = _fallback_grasp_from_current_hand(base_env, env_id)
                grasp = _pose_matrix_from_pos_quat(fallback_pos, fallback_quat)
                grasp_pos_w.append(fallback_pos)
                grasp_quat_w.append(fallback_quat)
                records.append(
                    {
                        "env_id": env_id,
                        "status": "fallback_noop",
                        "error": "no table-safe GraspGen candidate",
                        "chosen_index": None,
                        "chosen_rank": None,
                        "execute_grasp": False,
                        "confidence": None,
                        "grasp_pose_frame": args_cli.grasp_pose_frame,
                        "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
                        "grasp_matrix_w": grasp.tolist(),
                        "grasp_quat_wxyz": fallback_quat.tolist(),
                        "approach_dir_w": [0.0, 0.0, 0.0],
                        "pointcloud_mean_w": pc_mean[0].astype(np.float32).tolist(),
                        "pointcloud_source": pointcloud_stats[env_id]["source"],
                        "pointcloud_raw_points": pointcloud_stats[env_id]["raw_points"],
                        "pointcloud_finite_points": pointcloud_stats[env_id]["finite_points"],
                        "pointcloud_sent_points": int(pc_finite_w.shape[0]),
                        "num_returned": int(len(grasps)),
                        **filter_info,
                    }
                )
                print(
                    f"[WARNING] env={env_id} no selectable grasps "
                    f"full_safe={filter_info['table_full_safe_candidates']} "
                    f"hand_safe={filter_info['table_hand_safe_candidates']} "
                    f"unsafe={filter_info['table_unsafe_candidates']}; "
                    "using no-op fallback",
                    flush=True,
                )
                continue
            quat = _matrix_to_quat_wxyz(grasp[:3, :3])
            approach_dir = axis_sign * grasp[:3, axis_idx]
            grasp_pos_w.append(grasp[:3, 3])
            grasp_quat_w.append(quat)
            status = str(filter_info["status"])
            if status == "hand_safe_finger_collision":
                print(
                    f"[WARNING] env={env_id} no full gripper table-safe grasps; executing hand-safe candidate "
                    f"hand_clearance={filter_info['table_hand_clearance_m']} "
                    f"full_clearance={filter_info['table_clearance_m']}",
                    flush=True,
                )
            elif status == "upward_fallback":
                print(
                    f"[WARNING] env={env_id} no full-safe or hand-safe grasps; executing most-upward candidate "
                    f"upward_score={filter_info['upward_alignment_score']} "
                    f"full_clearance={filter_info['table_clearance_m']} "
                    f"hand_clearance={filter_info['table_hand_clearance_m']}",
                    flush=True,
                )
            records.append(
                {
                    "env_id": env_id,
                    "status": status,
                    "chosen_index": chosen_i,
                    "chosen_rank": chosen_rank,
                    "execute_grasp": True,
                    "confidence": float(confidences[chosen_i]),
                    "remove_outliers": bool(used_remove_outliers),
                    "grasp_pose_frame": args_cli.grasp_pose_frame,
                    "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
                    "grasp_matrix_w": grasp.tolist(),
                    "grasp_quat_wxyz": quat.tolist(),
                    "approach_dir_w": approach_dir.astype(np.float32).tolist(),
                    "pointcloud_mean_w": pc_mean[0].astype(np.float32).tolist(),
                    "pointcloud_source": pointcloud_stats[env_id]["source"],
                    "pointcloud_raw_points": pointcloud_stats[env_id]["raw_points"],
                    "pointcloud_finite_points": pointcloud_stats[env_id]["finite_points"],
                    "pointcloud_sent_points": int(pc_finite_w.shape[0]),
                    "num_returned": int(len(grasps)),
                    **filter_info,
                }
            )
            print(
                f"[GRASPGEN] env={env_id} grasps={len(grasps)} "
                f"full_safe={filter_info['table_full_safe_candidates']} "
                f"hand_safe={filter_info['table_hand_safe_candidates']} "
                f"unsafe={filter_info['table_unsafe_candidates']} "
                f"tier={filter_info['selection_tier']} "
                f"chosen={chosen_i} conf={float(confidences[chosen_i]):.4f} "
                f"clearance={filter_info['table_clearance_m']} "
                f"hand_clearance={filter_info['table_hand_clearance_m']} "
                f"up={filter_info['upward_alignment_score']}",
                flush=True,
            )

    pos = torch.tensor(np.asarray(grasp_pos_w), dtype=torch.float32, device=base_env.device)
    quat = torch.tensor(np.asarray(grasp_quat_w), dtype=torch.float32, device=base_env.device)
    return pos, quat, records


def _infer_lift_grasp_candidates(base_env) -> list[dict[str, Any]]:
    _add_graspgen_to_path(args_cli.graspgen_root)
    from grasp_gen.serving.zmq_client import GraspGenClient

    pointcloud_flat = mdp.get_object_pointcloud(base_env, SceneEntityCfg("object"))
    raw_pointcloud_w = pointcloud_flat.view(base_env.num_envs, -1, 3).float().detach().cpu().numpy()
    pointcloud_w, pointcloud_stats = _prepare_graspgen_point_clouds(base_env, raw_pointcloud_w)
    _visualize_object_cloud_points(base_env, pointcloud_w)
    finite_counts = [item["finite_points"] for item in pointcloud_stats]
    sent_counts = [item["sent_points"] for item in pointcloud_stats]
    print(
        "[GRASPGEN] lift eval object pointcloud "
        f"source={pointcloud_stats[0]['source']} "
        f"raw_points={raw_pointcloud_w.shape[1]} "
        f"finite_range=({min(finite_counts)}, {max(finite_counts)}) "
        f"sent_points={sent_counts[0] if len(set(sent_counts)) == 1 else sent_counts}",
        flush=True,
    )
    proxy_points_h = _get_gripper_collision_proxy_points_h(base_env)
    records: list[dict[str, Any]] = []

    with GraspGenClient(
        host=args_cli.graspgen_host,
        port=args_cli.graspgen_port,
        timeout_ms=args_cli.graspgen_timeout_ms,
    ) as client:
        print(f"[GRASPGEN] connected metadata={client.server_metadata}", flush=True)
        for env_id, pc_w in enumerate(pointcloud_w):
            finite_mask = np.isfinite(pc_w).all(axis=1)
            pc_finite_w = pc_w[finite_mask]
            if pc_finite_w.shape[0] == 0:
                grasps = np.empty((0, 4, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
                used_remove_outliers = False
                error = "object point cloud has no finite points"
                pc_mean = np.zeros((1, 3), dtype=np.float32)
            else:
                pc_mean = pc_finite_w.mean(axis=0, keepdims=True)
                pc_centered = (pc_finite_w - pc_mean).astype(np.float32)
                grasps, confidences, used_remove_outliers, error = _infer_grasps_with_retry(
                    client, pc_centered, env_id
                )

            candidates, candidate_info = _rank_lift_grasp_candidates(
                grasps,
                np.asarray(confidences),
                pc_mean[0],
                env_id,
                base_env,
                proxy_points_h,
            )
            record = {
                "env_id": int(env_id),
                "status": "ok" if len(candidates) > 0 else "no_safe_or_hand_safe_grasps",
                "error": error,
                "used_remove_outliers": bool(used_remove_outliers),
                "grasp_pose_frame": args_cli.grasp_pose_frame,
                "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
                "pointcloud_mean_w": pc_mean[0].astype(np.float32).tolist(),
                "pointcloud_source": pointcloud_stats[env_id]["source"],
                "pointcloud_raw_points": pointcloud_stats[env_id]["raw_points"],
                "pointcloud_finite_points": int(pc_finite_w.shape[0]),
                "pointcloud_sent_points": int(pc_finite_w.shape[0]),
                "candidate_stats": candidate_info["stats"],
                "all_candidates": candidate_info["all_candidates"],
                "attempt_candidates": candidates,
            }
            records.append(record)
            print(
                f"[GRASPGEN] env={env_id} returned={len(grasps)} "
                f"full_safe={candidate_info['stats']['full_safe_candidates']} "
                f"hand_safe={candidate_info['stats']['hand_safe_candidates']} "
                f"attempting={len(candidates)}",
                flush=True,
            )
    return records


def _annotate_records_with_targets(
    records: list[dict[str, Any]],
    grasp_target_pos_w: torch.Tensor,
    grasp_target_quat_w: torch.Tensor,
    pregrasp_target_pos_w: torch.Tensor,
    lift_target_pos_w: torch.Tensor,
) -> None:
    grasp_pos = grasp_target_pos_w.detach().cpu().numpy()
    grasp_quat = grasp_target_quat_w.detach().cpu().numpy()
    pregrasp_pos = pregrasp_target_pos_w.detach().cpu().numpy()
    lift_pos = lift_target_pos_w.detach().cpu().numpy()
    for env_id, record in enumerate(records):
        record["ik_body_name"] = "panda_hand"
        record["ik_target_pos_w"] = grasp_pos[env_id].astype(np.float32).tolist()
        record["ik_target_quat_wxyz"] = grasp_quat[env_id].astype(np.float32).tolist()
        record["ik_pregrasp_target_pos_w"] = pregrasp_pos[env_id].astype(np.float32).tolist()
        record["ik_lift_target_pos_w"] = lift_pos[env_id].astype(np.float32).tolist()


def _write_debug_json(base_env, output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = base_env.command_manager.get_command("target_object_pose")
    payload = {
        "runtime_spec": runtime_spec_path,
        "task": args_cli.task,
        "num_envs": int(base_env.num_envs),
        "grasp_pose_frame": args_cli.grasp_pose_frame,
        "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
        "grasp_to_hand_rot_matrix": _grasp_to_hand_rot_matrix_np().tolist(),
        "panda_hand_to_tcp_z": float(args_cli.panda_hand_to_tcp_z),
        "ik_method": args_cli.ik_method,
        "ik_damping": float(args_cli.ik_damping),
        "table_collision_filter_enabled": not bool(args_cli.no_table_collision_filter),
        "table_collision_clearance": float(args_cli.table_collision_clearance),
        "table_collision_xy_margin": float(args_cli.table_collision_xy_margin),
        "unsafe_grasp_fallback": args_cli.unsafe_grasp_fallback,
        "gripper_collision_proxy_source": getattr(base_env, "_graspgen_table_collision_proxy_source", None),
        "records": records,
        "object_root_pos_w": _jsonable_tensor(base_env.scene["object"].data.root_pos_w[:, :3]),
        "object_root_quat_wxyz": _jsonable_tensor(base_env.scene["object"].data.root_quat_w),
        "target_object_pose_env": _jsonable_tensor(command),
        "robot_root_pos_w": _jsonable_tensor(base_env.scene["robot"].data.root_pos_w[:, :3]),
        "robot_root_quat_wxyz": _jsonable_tensor(base_env.scene["robot"].data.root_quat_w),
    }
    path = output_dir / "graspgen_lift_grasps_debug.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote debug metadata: {path}", flush=True)


def _output_paths() -> tuple[Path, Path, Path | None]:
    results_dir = Path(args_cli.results_dir).expanduser().resolve()
    summary_path = (
        Path(args_cli.summary_json).expanduser().resolve()
        if args_cli.summary_json
        else results_dir / f"summary_rank_{int(args_cli.worker_id):03d}.json"
    )
    failures_path = (
        Path(args_cli.failures_jsonl).expanduser().resolve()
        if args_cli.failures_jsonl
        else results_dir / f"failures_rank_{int(args_cli.worker_id):03d}.jsonl"
    )
    all_results_path = (
        Path(args_cli.all_results_jsonl).expanduser().resolve()
        if args_cli.all_results_jsonl
        else None
    )
    return summary_path, failures_path, all_results_path


def _append_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


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
    return object_names


def _object_info_for_env(env_id: int, object_names: list[str]) -> dict[str, Any]:
    try:
        object_index = int(get_object_index_for_env(int(env_id)))
    except Exception:
        object_index = None
    object_name = None
    if object_index is not None and 0 <= object_index < len(object_names):
        object_name = object_names[object_index]
    return {"object_index": object_index, "object_name": object_name}


def _env_slice_jsonable(values: Any, env_id: int) -> Any:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        if values.ndim > 0 and values.shape[0] > int(env_id):
            return values[int(env_id)].tolist()
        return values.tolist()
    if isinstance(values, list) and len(values) > int(env_id):
        return values[int(env_id)]
    return values


def _collect_env_snapshot(base_env, env_id: int) -> dict[str, Any]:
    command = base_env.command_manager.get_command("target_object_pose")
    try:
        object_scale_xyz = _jsonable_tensor(
            mdp.get_rigid_body_scale(base_env, SceneEntityCfg("object"), [int(env_id)])[0]
        )
    except Exception:
        object_scale_xyz = None
    return {
        "object_root_pos_w": _env_slice_jsonable(base_env.scene["object"].data.root_pos_w[:, :3], env_id),
        "object_root_quat_wxyz": _env_slice_jsonable(base_env.scene["object"].data.root_quat_w, env_id),
        "object_scale_xyz": object_scale_xyz,
        "target_object_pose_env": _env_slice_jsonable(command, env_id),
        "robot_root_pos_w": _env_slice_jsonable(base_env.scene["robot"].data.root_pos_w[:, :3], env_id),
        "robot_root_quat_wxyz": _env_slice_jsonable(base_env.scene["robot"].data.root_quat_w, env_id),
    }


def _env_ids_tensor(base_env, env_ids: torch.Tensor | list[int] | None = None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(int(base_env.num_envs), dtype=torch.long, device=base_env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=base_env.device, dtype=torch.long)
    if isinstance(env_ids, int):
        return torch.tensor([int(env_ids)], dtype=torch.long, device=base_env.device)
    return torch.tensor([int(item) for item in env_ids], dtype=torch.long, device=base_env.device)


def _asset_root_state_w(asset, env_ids: torch.Tensor) -> torch.Tensor:
    data = asset.data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[env_ids].detach().clone()
    pos = data.root_pos_w[env_ids, :3]
    quat = data.root_quat_w[env_ids]
    lin_vel = getattr(data, "root_lin_vel_w", None)
    ang_vel = getattr(data, "root_ang_vel_w", None)
    if lin_vel is None:
        lin_vel = torch.zeros_like(pos)
    else:
        lin_vel = lin_vel[env_ids, :3]
    if ang_vel is None:
        ang_vel = torch.zeros_like(pos)
    else:
        ang_vel = ang_vel[env_ids, :3]
    return torch.cat([pos, quat, lin_vel, ang_vel], dim=1).detach().clone()


def _capture_lift_eval_state(base_env, env_ids: torch.Tensor | list[int] | None = None) -> dict[str, torch.Tensor]:
    env_ids_t = _env_ids_tensor(base_env, env_ids)
    robot = base_env.scene["robot"]
    obj = base_env.scene["object"]
    return {
        "env_ids": env_ids_t.detach().clone(),
        "object_root_state": _asset_root_state_w(obj, env_ids_t),
        "robot_root_state": _asset_root_state_w(robot, env_ids_t),
        "robot_joint_pos": robot.data.joint_pos[env_ids_t].detach().clone(),
        "robot_joint_vel": robot.data.joint_vel[env_ids_t].detach().clone(),
    }


def _restore_lift_eval_state(base_env, snapshot: dict[str, torch.Tensor]) -> None:
    device = base_env.device
    env_ids = snapshot["env_ids"].to(device=device, dtype=torch.long)
    robot = base_env.scene["robot"]
    obj = base_env.scene["object"]
    obj.write_root_state_to_sim(snapshot["object_root_state"].to(device), env_ids=env_ids)
    if hasattr(robot, "write_root_state_to_sim"):
        robot.write_root_state_to_sim(snapshot["robot_root_state"].to(device), env_ids=env_ids)
    robot.write_joint_state_to_sim(
        snapshot["robot_joint_pos"].to(device),
        snapshot["robot_joint_vel"].to(device),
        env_ids=env_ids,
    )
    if hasattr(robot, "set_joint_position_target"):
        robot.set_joint_position_target(snapshot["robot_joint_pos"].to(device), env_ids=env_ids)


def _attempt_lift_candidate(
    env,
    arm_cfg: SceneEntityCfg,
    candidate: dict[str, Any],
    initial_snapshot: dict[str, torch.Tensor],
    *,
    attempt_index: int,
) -> dict[str, Any]:
    base_env = env.unwrapped
    _restore_lift_eval_state(base_env, initial_snapshot)
    if args_cli.restore_settle_steps > 0:
        _step_hold(env, None, None, None, 1.0, args_cli.restore_settle_steps, "restore_open")

    grasp_matrix = np.asarray(candidate["grasp_matrix_w"], dtype=np.float32)
    grasp_pos_w = torch.tensor(grasp_matrix[:3, 3].reshape(1, 3), dtype=torch.float32, device=base_env.device)
    grasp_quat_w = torch.tensor(
        _matrix_to_quat_wxyz(grasp_matrix[:3, :3]).reshape(1, 4),
        dtype=torch.float32,
        device=base_env.device,
    )
    axis_idx, axis_sign = _axis_index(args_cli.approach_axis)
    approach_dir_np = (axis_sign * grasp_matrix[:3, axis_idx]).astype(np.float32)
    approach_dir_np = approach_dir_np / max(float(np.linalg.norm(approach_dir_np)), 1e-8)
    approach_dir_w = torch.tensor(approach_dir_np.reshape(1, 3), dtype=torch.float32, device=base_env.device)
    pregrasp_pose_pos_w = grasp_pos_w - float(args_cli.pregrasp_offset) * approach_dir_w
    lift_pose_pos_w = grasp_pos_w.clone()
    lift_pose_pos_w[:, 2] += float(args_cli.lift_distance)

    grasp_target_pos_w, grasp_target_quat_w = _grasp_pose_to_hand_target(grasp_pos_w, grasp_quat_w)
    pregrasp_target_pos_w, pregrasp_target_quat_w = _grasp_pose_to_hand_target(pregrasp_pose_pos_w, grasp_quat_w)
    lift_target_pos_w, lift_target_quat_w = _grasp_pose_to_hand_target(lift_pose_pos_w, grasp_quat_w)

    initial_object_z = float(initial_snapshot["object_root_state"][0, 2].detach().cpu().item())
    attempt_record: dict[str, Any] = {
        **candidate,
        "attempt_index": int(attempt_index),
        "approach_dir_w": approach_dir_np.tolist(),
        "ik_pregrasp_target_pos_w": pregrasp_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
        "ik_grasp_target_pos_w": grasp_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
        "ik_lift_target_pos_w": lift_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
        "initial_object_z_w": initial_object_z,
    }

    try:
        if args_cli.use_curobo_pregrasp:
            used_curobo_pregrasp = _step_curobo_pregrasp(
                env,
                arm_cfg,
                None,
                None,
                None,
                1.0,
                pregrasp_target_pos_w,
                pregrasp_target_quat_w,
                label=f"candidate_{attempt_index}_curobo_pregrasp",
            )
        else:
            used_curobo_pregrasp = False
        if not used_curobo_pregrasp:
            _step_arm_to_pose(
                env,
                arm_cfg,
                None,
                None,
                None,
                1.0,
                pregrasp_target_pos_w,
                pregrasp_target_quat_w,
                steps=args_cli.approach_steps,
                label=f"candidate_{attempt_index}_approach_pregrasp",
            )
        _step_arm_to_pose(
            env,
            arm_cfg,
            None,
            None,
            None,
            1.0,
            grasp_target_pos_w,
            grasp_target_quat_w,
            steps=args_cli.grasp_steps,
            label=f"candidate_{attempt_index}_move_to_grasp",
        )
        _step_hold(env, None, None, None, -1.0, args_cli.close_steps, f"candidate_{attempt_index}_close")
        _step_arm_to_pose(
            env,
            arm_cfg,
            None,
            None,
            None,
            -1.0,
            lift_target_pos_w,
            lift_target_quat_w,
            steps=args_cli.lift_steps,
            label=f"candidate_{attempt_index}_lift",
        )
        stability_hold = _step_hold_measure_object_z(
            env,
            None,
            None,
            None,
            -1.0,
            _stable_hold_steps(),
            f"candidate_{attempt_index}_hold_closed_stability",
        )
        attempt_record["used_curobo_pregrasp"] = bool(used_curobo_pregrasp)
        attempt_record["exception"] = None
    except Exception as exc:
        stability_hold = None
        attempt_record["used_curobo_pregrasp"] = False
        attempt_record["exception"] = f"{type(exc).__name__}: {exc}"

    final_state = _collect_env_snapshot(base_env, 0)
    if stability_hold is None:
        final_object_z = float(base_env.scene["object"].data.root_pos_w[0, 2].detach().cpu().item())
        min_hold_object_z = final_object_z
        stability_steps = 0
        stability_seconds = 0.0
    else:
        final_object_z = float(stability_hold["final_object_z_w"][0].detach().cpu().item())
        min_hold_object_z = float(stability_hold["min_object_z_w"][0].detach().cpu().item())
        stability_steps = int(stability_hold["steps"])
        stability_seconds = stability_hold["seconds"]
    lifted_delta = final_object_z - initial_object_z
    min_hold_lifted_delta = min_hold_object_z - initial_object_z
    height_threshold = float(args_cli.lift_success_height_delta)
    stability_tolerance = max(0.0, float(args_cli.lift_success_stability_tolerance))
    success = lifted_delta >= height_threshold and min_hold_lifted_delta >= height_threshold - stability_tolerance
    attempt_record["final_state"] = final_state
    attempt_record["final_object_z_w"] = final_object_z
    attempt_record["min_hold_object_z_w"] = min_hold_object_z
    attempt_record["lifted_delta_m"] = float(lifted_delta)
    attempt_record["min_hold_lifted_delta_m"] = float(min_hold_lifted_delta)
    attempt_record["lift_success_height_delta_m"] = float(args_cli.lift_success_height_delta)
    attempt_record["lift_success_stable_seconds"] = float(args_cli.lift_success_stable_seconds)
    attempt_record["lift_success_stability_tolerance_m"] = float(args_cli.lift_success_stability_tolerance)
    attempt_record["stability_hold_steps"] = int(stability_steps)
    attempt_record["stability_hold_seconds"] = None if stability_seconds is None else float(stability_seconds)
    attempt_record["success"] = bool(success)
    return attempt_record


def _attempt_lift_candidate_batch(
    env,
    arm_cfg: SceneEntityCfg,
    candidates_by_env: dict[int, dict[str, Any]],
    initial_snapshot: dict[str, torch.Tensor],
    *,
    attempt_index: int,
) -> dict[int, dict[str, Any]]:
    base_env = env.unwrapped
    num_envs = int(base_env.num_envs)
    _restore_lift_eval_state(base_env, initial_snapshot)
    if args_cli.restore_settle_steps > 0:
        _step_hold(env, None, None, None, 1.0, args_cli.restore_settle_steps, f"restore_open_{attempt_index}")

    current_hand_pos_w, current_hand_quat_w = _current_hand_pose_w(base_env)
    pregrasp_target_pos_w = current_hand_pos_w.clone()
    pregrasp_target_quat_w = current_hand_quat_w.clone()
    grasp_target_pos_w = current_hand_pos_w.clone()
    grasp_target_quat_w = current_hand_quat_w.clone()
    lift_target_pos_w = current_hand_pos_w.clone()
    lift_target_quat_w = current_hand_quat_w.clone()

    env_ids_snapshot = [int(v) for v in initial_snapshot["env_ids"].detach().cpu().tolist()]
    snapshot_row_by_env = {env_id: row for row, env_id in enumerate(env_ids_snapshot)}
    attempt_records: dict[int, dict[str, Any]] = {}
    axis_idx, axis_sign = _axis_index(args_cli.approach_axis)

    for env_id, candidate in sorted(candidates_by_env.items()):
        grasp_matrix = np.asarray(candidate["grasp_matrix_w"], dtype=np.float32)
        grasp_pos_w = torch.tensor(grasp_matrix[:3, 3].reshape(1, 3), dtype=torch.float32, device=base_env.device)
        grasp_quat_w = torch.tensor(
            _matrix_to_quat_wxyz(grasp_matrix[:3, :3]).reshape(1, 4),
            dtype=torch.float32,
            device=base_env.device,
        )
        approach_dir_np = (axis_sign * grasp_matrix[:3, axis_idx]).astype(np.float32)
        approach_dir_np = approach_dir_np / max(float(np.linalg.norm(approach_dir_np)), 1e-8)
        approach_dir_w = torch.tensor(approach_dir_np.reshape(1, 3), dtype=torch.float32, device=base_env.device)
        pregrasp_pose_pos_w = grasp_pos_w - float(args_cli.pregrasp_offset) * approach_dir_w
        lift_pose_pos_w = grasp_pos_w.clone()
        lift_pose_pos_w[:, 2] += float(args_cli.lift_distance)
        env_grasp_target_pos_w, env_grasp_target_quat_w = _grasp_pose_to_hand_target(grasp_pos_w, grasp_quat_w)
        env_pregrasp_target_pos_w, env_pregrasp_target_quat_w = _grasp_pose_to_hand_target(
            pregrasp_pose_pos_w,
            grasp_quat_w,
        )
        env_lift_target_pos_w, env_lift_target_quat_w = _grasp_pose_to_hand_target(lift_pose_pos_w, grasp_quat_w)

        pregrasp_target_pos_w[env_id] = env_pregrasp_target_pos_w[0]
        pregrasp_target_quat_w[env_id] = env_pregrasp_target_quat_w[0]
        grasp_target_pos_w[env_id] = env_grasp_target_pos_w[0]
        grasp_target_quat_w[env_id] = env_grasp_target_quat_w[0]
        lift_target_pos_w[env_id] = env_lift_target_pos_w[0]
        lift_target_quat_w[env_id] = env_lift_target_quat_w[0]

        snapshot_row = snapshot_row_by_env[int(env_id)]
        initial_object_z = float(initial_snapshot["object_root_state"][snapshot_row, 2].detach().cpu().item())
        attempt_records[int(env_id)] = {
            **candidate,
            "env_id": int(env_id),
            "attempt_index": int(attempt_index),
            "approach_dir_w": approach_dir_np.tolist(),
            "ik_pregrasp_target_pos_w": env_pregrasp_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "ik_grasp_target_pos_w": env_grasp_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "ik_lift_target_pos_w": env_lift_target_pos_w[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "initial_object_z_w": initial_object_z,
        }

    try:
        _step_arm_to_pose(
            env,
            arm_cfg,
            None,
            None,
            None,
            1.0,
            pregrasp_target_pos_w,
            pregrasp_target_quat_w,
            steps=args_cli.approach_steps,
            label=f"candidate_{attempt_index}_approach_pregrasp_batch",
        )
        _step_arm_to_pose(
            env,
            arm_cfg,
            None,
            None,
            None,
            1.0,
            grasp_target_pos_w,
            grasp_target_quat_w,
            steps=args_cli.grasp_steps,
            label=f"candidate_{attempt_index}_move_to_grasp_batch",
        )
        _step_hold(env, None, None, None, -1.0, args_cli.close_steps, f"candidate_{attempt_index}_close_batch")
        _step_arm_to_pose(
            env,
            arm_cfg,
            None,
            None,
            None,
            -1.0,
            lift_target_pos_w,
            lift_target_quat_w,
            steps=args_cli.lift_steps,
            label=f"candidate_{attempt_index}_lift_batch",
        )
        stability_hold = _step_hold_measure_object_z(
            env,
            None,
            None,
            None,
            -1.0,
            _stable_hold_steps(),
            f"candidate_{attempt_index}_hold_closed_stability_batch",
        )
        exception = None
    except Exception as exc:
        stability_hold = None
        exception = f"{type(exc).__name__}: {exc}"

    for env_id, attempt_record in attempt_records.items():
        final_state = _collect_env_snapshot(base_env, env_id)
        if stability_hold is None:
            final_object_z = float(base_env.scene["object"].data.root_pos_w[env_id, 2].detach().cpu().item())
            min_hold_object_z = final_object_z
            stability_steps = 0
            stability_seconds = 0.0
        else:
            final_object_z = float(stability_hold["final_object_z_w"][env_id].detach().cpu().item())
            min_hold_object_z = float(stability_hold["min_object_z_w"][env_id].detach().cpu().item())
            stability_steps = int(stability_hold["steps"])
            stability_seconds = stability_hold["seconds"]
        lifted_delta = final_object_z - float(attempt_record["initial_object_z_w"])
        min_hold_lifted_delta = min_hold_object_z - float(attempt_record["initial_object_z_w"])
        height_threshold = float(args_cli.lift_success_height_delta)
        stability_tolerance = max(0.0, float(args_cli.lift_success_stability_tolerance))
        success = lifted_delta >= height_threshold and min_hold_lifted_delta >= height_threshold - stability_tolerance
        attempt_record["exception"] = exception
        attempt_record["final_state"] = final_state
        attempt_record["final_object_z_w"] = final_object_z
        attempt_record["min_hold_object_z_w"] = min_hold_object_z
        attempt_record["lifted_delta_m"] = float(lifted_delta)
        attempt_record["min_hold_lifted_delta_m"] = float(min_hold_lifted_delta)
        attempt_record["lift_success_height_delta_m"] = float(args_cli.lift_success_height_delta)
        attempt_record["lift_success_stable_seconds"] = float(args_cli.lift_success_stable_seconds)
        attempt_record["lift_success_stability_tolerance_m"] = float(args_cli.lift_success_stability_tolerance)
        attempt_record["stability_hold_steps"] = int(stability_steps)
        attempt_record["stability_hold_seconds"] = None if stability_seconds is None else float(stability_seconds)
        attempt_record["success"] = bool(success)
    return attempt_records


def _current_success_mask(base_env) -> list[bool]:
    if not hasattr(base_env, "episode_success_buf"):
        raise AttributeError("Environment does not have episode_success_buf.")
    values = base_env.episode_success_buf.detach().cpu().numpy().astype(bool)
    return [bool(item) for item in values.tolist()]


def _termination_term_mask(base_env, name: str) -> torch.Tensor | None:
    try:
        value = base_env.termination_manager.get_term(name)
    except Exception:
        return None
    if value is None:
        return None
    return value.detach().to(base_env.device).bool()


def _object_goal_error_tensors(base_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = base_env.scene["object"]
    command = base_env.command_manager.get_command("target_object_pose")
    target_pos_env = command[:, :3]
    target_quat = command[:, 3:7]
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_env = object_pos_w - base_env.scene.env_origins
    object_quat = obj.data.root_quat_w
    position_error = torch.linalg.norm(target_pos_env - object_pos_env, dim=1)
    quat_dot = torch.sum(object_quat * target_quat, dim=1)
    quat_dot = torch.clamp(torch.abs(quat_dot), max=1.0)
    rotation_error = 2.0 * torch.acos(quat_dot)
    return object_pos_w[:, 2], position_error, rotation_error


def _append_termination_debug_events(
    base_env,
    debug_tracker: dict[str, Any] | None,
    terminated: torch.Tensor | None,
    truncated: torch.Tensor | None,
    label: str,
) -> None:
    if debug_tracker is None or terminated is None or truncated is None:
        return
    episode_ended = (terminated | truncated).detach().to(base_env.device).bool()
    ended_mask = debug_tracker.get("ended_mask")
    if ended_mask is not None:
        episode_ended = episode_ended & ~ended_mask
    debug_tracker["last_step_new_ended_mask"] = episode_ended.clone()
    if not bool(torch.any(episode_ended).detach().cpu()):
        return

    reached = _termination_term_mask(base_env, "reached")
    object_dropped = _termination_term_mask(base_env, "object_dropped")
    time_out = _termination_term_mask(base_env, "time_out")
    object_z, pos_error, rot_error = _object_goal_error_tensors(base_env)
    before_reset = getattr(base_env, "_episode_success_before_reset", None)
    if before_reset is not None:
        before_reset = before_reset.detach().to(base_env.device).bool()

    events_by_env: list[list[dict[str, Any]]] = debug_tracker["termination_events"]
    counts: torch.Tensor = debug_tracker["step_counts"]
    for env_id in torch.where(episode_ended)[0].detach().cpu().tolist():
        env_i = int(env_id)
        event = {
            "event_index": len(events_by_env[env_i]),
            "script_step_count": int(counts[env_i].detach().cpu().item()),
            "motion_label": str(label),
            "terminated": bool(terminated[env_i].detach().cpu().item()),
            "truncated": bool(truncated[env_i].detach().cpu().item()),
            "reached": None if reached is None else bool(reached[env_i].detach().cpu().item()),
            "object_dropped": None
            if object_dropped is None
            else bool(object_dropped[env_i].detach().cpu().item()),
            "time_out": None if time_out is None else bool(time_out[env_i].detach().cpu().item()),
            "success_before_reset": None
            if before_reset is None
            else bool(before_reset[env_i].detach().cpu().item()),
            "object_root_z_w": float(object_z[env_i].detach().cpu().item()),
            "goal_position_error_m": float(pos_error[env_i].detach().cpu().item()),
            "goal_rotation_error_rad": float(rot_error[env_i].detach().cpu().item()),
        }
        events_by_env[env_i].append(event)
        print(
            "[TERMINATION] "
            f"label={label} env={env_i} terminated={event['terminated']} truncated={event['truncated']} "
            f"reached={event['reached']} dropped={event['object_dropped']} timeout={event['time_out']} "
            f"success_before_reset={event['success_before_reset']} "
            f"z={event['object_root_z_w']:.4f} "
            f"pos_err={event['goal_position_error_m']:.4f} "
            f"rot_err={event['goal_rotation_error_rad']:.4f}",
            flush=True,
        )
    if ended_mask is not None:
        ended_mask[episode_ended] = True


def _record_success_state(
    base_env,
    success_tracker: torch.Tensor | None,
    debug_tracker: dict[str, Any] | None = None,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
    label: str = "",
) -> None:
    if debug_tracker is not None:
        debug_tracker["last_step_new_ended_mask"] = None
        debug_tracker["step_counts"] += 1
        _append_termination_debug_events(base_env, debug_tracker, terminated, truncated, label)
    if success_tracker is None:
        return
    if hasattr(base_env, "episode_success_buf"):
        current_success = base_env.episode_success_buf.detach().to(success_tracker.device).bool()
        ended_mask = None if debug_tracker is None else debug_tracker.get("ended_mask")
        if ended_mask is not None:
            current_success = current_success & ~ended_mask
        success_tracker |= current_success
    if terminated is None or truncated is None:
        return
    if debug_tracker is not None and debug_tracker.get("last_step_new_ended_mask") is not None:
        episode_ended = debug_tracker["last_step_new_ended_mask"].detach().to(success_tracker.device).bool()
    else:
        episode_ended = (terminated | truncated).detach().to(success_tracker.device).bool()
    if not bool(torch.any(episode_ended).detach().cpu()):
        return
    before_reset = getattr(base_env, "_episode_success_before_reset", None)
    if before_reset is not None:
        before_reset = before_reset.detach().to(success_tracker.device).bool()
        success_tracker[episode_ended] |= before_reset[episode_ended]


def _success_tracker_mask(success_tracker: torch.Tensor) -> list[bool]:
    values = success_tracker.detach().cpu().numpy().astype(bool)
    return [bool(item) for item in values.tolist()]


def _run_scripted_episode(
    env,
    arm_cfg: SceneEntityCfg,
    video_output_dir: Path | None = None,
    episode_idx: int | None = None,
) -> tuple[list[dict[str, Any]], list[bool]]:
    base_env = env.unwrapped
    success_tracker = torch.zeros(int(base_env.num_envs), dtype=torch.bool, device=base_env.device)
    debug_tracker: dict[str, Any] = {
        "step_counts": torch.zeros(int(base_env.num_envs), dtype=torch.int32, device=base_env.device),
        "ended_mask": torch.zeros(int(base_env.num_envs), dtype=torch.bool, device=base_env.device),
        "last_step_new_ended_mask": None,
        "termination_events": [[] for _ in range(int(base_env.num_envs))],
    }
    video_state = None
    records: list[dict[str, Any]] = []
    try:
        if args_cli.record_video:
            video_state = _init_video_state(
                video_output_dir or Path(args_cli.results_dir),
                base_env.num_envs,
                episode_idx,
            )
            _capture_video_frames(env, video_state)

        _step_hold(env, video_state, success_tracker, debug_tracker, 1.0, args_cli.settle_steps, "settle_open")
        _step_hold(env, video_state, success_tracker, debug_tracker, 1.0, args_cli.open_steps, "open")

        grasp_pose_pos_w, grasp_pose_quat_w, records = _infer_grasps(base_env)
        initial_snapshots = [_collect_env_snapshot(base_env, env_id) for env_id in range(int(base_env.num_envs))]

        approach_dirs = [np.asarray(record["approach_dir_w"], dtype=np.float32) for record in records]
        approach_dir_w = torch.tensor(np.asarray(approach_dirs), dtype=torch.float32, device=base_env.device)
        approach_dir_w = approach_dir_w / torch.clamp(torch.linalg.norm(approach_dir_w, dim=-1, keepdim=True), min=1e-8)
        execute_mask = torch.tensor(
            [bool(record.get("execute_grasp", record.get("status") == "ok")) for record in records],
            dtype=torch.bool,
            device=base_env.device,
        )
        pregrasp_candidate_pos_w = grasp_pose_pos_w - float(args_cli.pregrasp_offset) * approach_dir_w
        pregrasp_pose_pos_w = torch.where(execute_mask.unsqueeze(-1), pregrasp_candidate_pos_w, grasp_pose_pos_w)
        lift_pose_pos_w = grasp_pose_pos_w.clone()
        lift_pose_pos_w[execute_mask, 2] += float(args_cli.lift_distance)

        grasp_target_pos_w, grasp_target_quat_w = _grasp_pose_to_hand_target(grasp_pose_pos_w, grasp_pose_quat_w)
        pregrasp_target_pos_w, pregrasp_target_quat_w = _grasp_pose_to_hand_target(
            pregrasp_pose_pos_w,
            grasp_pose_quat_w,
        )
        lift_target_pos_w, lift_target_quat_w = _grasp_pose_to_hand_target(lift_pose_pos_w, grasp_pose_quat_w)
        _annotate_records_with_targets(
            records,
            grasp_target_pos_w,
            grasp_target_quat_w,
            pregrasp_target_pos_w,
            lift_target_pos_w,
        )

        for env_id, record in enumerate(records):
            record["initial_state"] = initial_snapshots[env_id]

        if args_cli.dry_run_graspgen:
            for record in records:
                record["motion_skipped"] = True
            return records, [False for _ in records]

        _visualize_grasp_guides(base_env, records)
        if args_cli.record_video and video_state is None:
            video_state = _init_video_state(
                video_output_dir or Path(args_cli.results_dir),
                base_env.num_envs,
                episode_idx,
            )
            _capture_video_frames(env, video_state)
        if video_state is not None:
            for env_id, record in enumerate(records):
                record["video_path"] = str(video_state["paths"][env_id])

        used_curobo_pregrasp = False
        if args_cli.use_curobo_pregrasp:
            used_curobo_pregrasp = _step_curobo_pregrasp(
                env,
                arm_cfg,
                video_state,
                success_tracker,
                debug_tracker,
                1.0,
                pregrasp_target_pos_w,
                pregrasp_target_quat_w,
                label="approach_pregrasp",
            )
        if not used_curobo_pregrasp:
            _step_arm_to_pose(
                env,
                arm_cfg,
                video_state,
                success_tracker,
                debug_tracker,
                1.0,
                pregrasp_target_pos_w,
                pregrasp_target_quat_w,
                steps=args_cli.approach_steps,
                label="approach_pregrasp",
            )
        _step_arm_to_pose(
            env,
            arm_cfg,
            video_state,
            success_tracker,
            debug_tracker,
            1.0,
            grasp_target_pos_w,
            grasp_target_quat_w,
            steps=args_cli.grasp_steps,
            label="move_to_grasp",
        )
        _step_hold(env, video_state, success_tracker, debug_tracker, -1.0, args_cli.close_steps, "close")
        used_curobo_goal_transport = False
        used_goal_transport = args_cli.post_grasp_motion == "goal_transport"
        used_pre_transport_lift = False
        if used_goal_transport:
            if args_cli.pre_transport_lift:
                _step_arm_to_pose(
                    env,
                    arm_cfg,
                    video_state,
                    success_tracker,
                    debug_tracker,
                    -1.0,
                    lift_target_pos_w,
                    lift_target_quat_w,
                    steps=args_cli.lift_steps,
                    label="pre_transport_lift",
                )
                used_pre_transport_lift = True
            if args_cli.use_curobo_goal_transport:
                used_curobo_goal_transport = _step_curobo_goal_transport(
                    env,
                    arm_cfg,
                    video_state,
                    success_tracker,
                    debug_tracker,
                )
            else:
                _step_diff_ik_goal_transport(env, arm_cfg, video_state, success_tracker, debug_tracker)
        else:
            _step_arm_to_pose(
                env,
                arm_cfg,
                video_state,
                success_tracker,
                debug_tracker,
                -1.0,
                lift_target_pos_w,
                lift_target_quat_w,
                steps=args_cli.lift_steps,
                label="lift",
            )
        _step_hold(env, video_state, success_tracker, debug_tracker, -1.0, args_cli.hold_steps, "hold_closed")
        if args_cli.record_video:
            _step_until_episode_end(env, video_state, success_tracker, debug_tracker)

        _record_success_state(base_env, success_tracker, debug_tracker)
        success_mask = _success_tracker_mask(success_tracker)
        for env_id, record in enumerate(records):
            record["success_detected_during_scripted_episode"] = bool(success_mask[env_id])
            record["termination_events"] = debug_tracker["termination_events"][env_id]
            if video_state is not None:
                record["video_stop_reason"] = video_state["stop_reasons"][env_id]
            record["post_grasp_motion"] = args_cli.post_grasp_motion
            record["used_goal_transport"] = bool(used_goal_transport)
            record["used_pre_transport_lift"] = bool(used_pre_transport_lift)
            record["used_curobo_pregrasp"] = bool(used_curobo_pregrasp)
            record["used_curobo_goal_transport"] = bool(used_curobo_goal_transport)
            record["final_state"] = _collect_env_snapshot(base_env, env_id)
        return records, success_mask
    finally:
        _close_video_state(video_state)


def _run_lift_grasp_sample(env, arm_cfg: SceneEntityCfg, episode_idx: int) -> tuple[dict[str, Any], bool]:
    base_env = env.unwrapped
    _step_hold(env, None, None, None, 1.0, args_cli.settle_steps, "settle_open")
    _step_hold(env, None, None, None, 1.0, args_cli.open_steps, "open")
    initial_snapshot = _capture_lift_eval_state(base_env, 0)
    initial_state = _collect_env_snapshot(base_env, 0)
    candidate_records = _infer_lift_grasp_candidates(base_env)
    if len(candidate_records) != 1:
        raise RuntimeError(f"Lift grasp eval requires exactly one env record, got {len(candidate_records)}")
    candidate_record = candidate_records[0]
    attempts: list[dict[str, Any]] = []
    success = False
    successful_attempt: dict[str, Any] | None = None

    for attempt_index, candidate in enumerate(candidate_record["attempt_candidates"]):
        attempt = _attempt_lift_candidate(
            env,
            arm_cfg,
            candidate,
            initial_snapshot,
            attempt_index=attempt_index,
        )
        attempts.append(attempt)
        print(
            f"[LIFT_EVAL] episode={episode_idx} candidate={attempt_index + 1}/"
            f"{len(candidate_record['attempt_candidates'])} rank={candidate.get('rank')} "
            f"tier={candidate.get('selection_tier')} success={attempt['success']} "
            f"lift_delta={attempt['lifted_delta_m']:.4f} "
            f"min_hold_delta={attempt['min_hold_lifted_delta_m']:.4f}",
            flush=True,
        )
        if bool(attempt["success"]):
            success = True
            successful_attempt = attempt
            break

    failed_attempts = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    row = {
        "episode_index": int(episode_idx),
        "env_id": 0,
        "success": bool(success),
        "initial_state": initial_state,
        "candidate_generation": candidate_record,
        "attempts": attempts,
        "failed_attempts": failed_attempts,
        "successful_attempt": successful_attempt,
        "attempted_candidates": int(len(attempts)),
        "attemptable_candidates": int(len(candidate_record["attempt_candidates"])),
        "lift_success_height_delta_m": float(args_cli.lift_success_height_delta),
        "lift_success_stable_seconds": float(args_cli.lift_success_stable_seconds),
        "lift_success_stability_tolerance_m": float(args_cli.lift_success_stability_tolerance),
    }
    return row, success


def _run_lift_grasp_samples(env, arm_cfg: SceneEntityCfg, episode_idx: int) -> tuple[list[dict[str, Any]], list[bool]]:
    base_env = env.unwrapped
    num_envs = int(base_env.num_envs)
    _step_hold(env, None, None, None, 1.0, args_cli.settle_steps, "settle_open")
    _step_hold(env, None, None, None, 1.0, args_cli.open_steps, "open")
    initial_snapshot = _capture_lift_eval_state(base_env)
    initial_states = [_collect_env_snapshot(base_env, env_id) for env_id in range(num_envs)]
    candidate_records = _infer_lift_grasp_candidates(base_env)
    if len(candidate_records) != num_envs:
        raise RuntimeError(f"Expected {num_envs} candidate records, got {len(candidate_records)}")

    attempts_by_env: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    successful_attempts: list[dict[str, Any] | None] = [None for _ in range(num_envs)]
    success_mask = [False for _ in range(num_envs)]
    max_rounds = max((len(record["attempt_candidates"]) for record in candidate_records), default=0)

    for attempt_index in range(max_rounds):
        candidates_by_env: dict[int, dict[str, Any]] = {}
        for env_id, candidate_record in enumerate(candidate_records):
            if success_mask[env_id]:
                continue
            candidates = candidate_record["attempt_candidates"]
            if attempt_index < len(candidates):
                candidates_by_env[env_id] = candidates[attempt_index]
        if not candidates_by_env:
            continue

        round_attempts = _attempt_lift_candidate_batch(
            env,
            arm_cfg,
            candidates_by_env,
            initial_snapshot,
            attempt_index=attempt_index,
        )
        for env_id, attempt in sorted(round_attempts.items()):
            attempts_by_env[env_id].append(attempt)
            print(
                f"[LIFT_EVAL] episode={episode_idx} env={env_id} "
                f"candidate={attempt_index + 1}/{len(candidate_records[env_id]['attempt_candidates'])} "
                f"rank={attempt.get('rank')} tier={attempt.get('selection_tier')} "
                f"success={attempt['success']} lift_delta={attempt['lifted_delta_m']:.4f} "
                f"min_hold_delta={attempt['min_hold_lifted_delta_m']:.4f}",
                flush=True,
            )
            if bool(attempt["success"]):
                success_mask[env_id] = True
                successful_attempts[env_id] = attempt
        if all(success_mask[env_id] or len(candidate_records[env_id]["attempt_candidates"]) <= attempt_index + 1 for env_id in range(num_envs)):
            if all(success_mask[env_id] or not candidate_records[env_id]["attempt_candidates"] for env_id in range(num_envs)):
                break

    rows: list[dict[str, Any]] = []
    for env_id in range(num_envs):
        attempts = attempts_by_env[env_id]
        failed_attempts = [attempt for attempt in attempts if not bool(attempt.get("success"))]
        rows.append(
            {
                "episode_index": int(episode_idx),
                "env_id": int(env_id),
                "success": bool(success_mask[env_id]),
                "initial_state": initial_states[env_id],
                "candidate_generation": candidate_records[env_id],
                "attempts": attempts,
                "failed_attempts": failed_attempts,
                "successful_attempt": successful_attempts[env_id],
                "attempted_candidates": int(len(attempts)),
                "attemptable_candidates": int(len(candidate_records[env_id]["attempt_candidates"])),
                "lift_success_height_delta_m": float(args_cli.lift_success_height_delta),
                "lift_success_stable_seconds": float(args_cli.lift_success_stable_seconds),
                "lift_success_stability_tolerance_m": float(args_cli.lift_success_stability_tolerance),
            }
        )
    return rows, success_mask


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.disable_obs_noise = True
    env_cfg.viewer.eye = (1.05, 0.25, 0.48)
    env_cfg.viewer.lookat = (0.58, 0.0, 0.05)
    env_cfg.visualize_current_object_pose = bool(args_cli.visualize_object_markers)
    env_cfg.visualize_object_pointcloud = False
    env_cfg.visualize_tool_pointcloud = False
    env_cfg.visualize_head_area_center = False
    env_cfg.visualize_eef_position = False
    target_command_cfg = getattr(getattr(env_cfg, "commands", None), "target_object_pose", None)
    if target_command_cfg is not None and hasattr(target_command_cfg, "debug_vis"):
        target_command_cfg.debug_vis = bool(args_cli.visualize_object_markers)
    if args_cli.record_video:
        env_cfg.scene.grasp_record_camera = _make_record_camera_cfg()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    summary_path, failures_path, all_results_path = _output_paths()
    video_output_dir = (
        Path(args_cli.video_dir).expanduser().resolve()
        if args_cli.video_dir
        else summary_path.parent / "videos"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    if failures_path.exists():
        failures_path.unlink()
    if all_results_path is not None:
        all_results_path.parent.mkdir(parents=True, exist_ok=True)
        if all_results_path.exists():
            all_results_path.unlink()

    print(f"[INFO] runtime spec: {runtime_spec_path}", flush=True)
    print(
        f"[INFO] eval task={args_cli.task} num_envs={args_cli.num_envs} "
        f"episodes={args_cli.num_episodes} worker={args_cli.worker_id}/{args_cli.num_workers}",
        flush=True,
    )
    start_time = time.monotonic()
    total_attempts = 0
    total_successes = 0
    failures_written = 0
    failed_object_names: set[str] = set()
    failed_grasp_pose_rows: list[dict[str, Any]] = []
    object_names: list[str] = []
    try:
        object_names = _object_names_from_loaded_data()
        for episode_idx in range(int(args_cli.num_episodes)):
            env.reset()
            base_env = env.unwrapped
            arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
            arm_cfg.resolve(base_env.scene)
            finger_cfg.resolve(base_env.scene)
            if len(arm_cfg.joint_ids) != 7:
                raise RuntimeError(f"Expected 7 Panda arm joints, got {arm_cfg.joint_names!r}")
            if len(finger_cfg.joint_ids) != 2:
                raise RuntimeError(f"Expected 2 Panda finger joints, got {finger_cfg.joint_names!r}")

            sample_records, success_mask = _run_lift_grasp_samples(env, arm_cfg, episode_idx)
            episode_rows = []
            failure_rows = []
            for env_id, sample_record in enumerate(sample_records):
                success = bool(success_mask[env_id])
                object_info = _object_info_for_env(env_id, object_names)
                sample_index = int(episode_idx) * int(args_cli.num_envs) + int(env_id)
                row = {
                    "worker_id": int(args_cli.worker_id),
                    "num_workers": int(args_cli.num_workers),
                    "episode_index": int(episode_idx),
                    "sample_index": int(sample_index),
                    "env_id": int(env_id),
                    "seed": None if args_cli.seed is None else int(args_cli.seed),
                    "success": bool(success),
                    "task": args_cli.task,
                    **object_info,
                    "record": sample_record,
                }
                episode_rows.append(row)
                if not success:
                    failure_rows.append(row)
                    failed_name = object_info.get("object_name")
                    if failed_name is not None:
                        failed_object_names.add(str(failed_name))
                    failed_pose_entry = {
                        "worker_id": int(args_cli.worker_id),
                        "episode_index": int(episode_idx),
                        "sample_index": int(sample_index),
                        "env_id": int(env_id),
                        **object_info,
                        "initial_object_pose": sample_record["initial_state"],
                        "candidate_stats": sample_record["candidate_generation"]["candidate_stats"],
                        "failed_grasp_poses": [
                            {
                                "attempt_index": int(attempt["attempt_index"]),
                                "rank": int(attempt["rank"]),
                                "candidate_index": int(attempt["candidate_index"]),
                                "selection_tier": attempt["selection_tier"],
                                "confidence": float(attempt["confidence"]),
                                "grasp_matrix_w": attempt["grasp_matrix_w"],
                                "grasp_quat_wxyz": attempt["grasp_quat_wxyz"],
                                "lifted_delta_m": float(attempt["lifted_delta_m"]),
                                "min_hold_lifted_delta_m": float(attempt["min_hold_lifted_delta_m"]),
                                "final_object_z_w": float(attempt["final_object_z_w"]),
                                "min_hold_object_z_w": float(attempt["min_hold_object_z_w"]),
                                "stability_hold_seconds": attempt["stability_hold_seconds"],
                            }
                            for attempt in sample_record["failed_attempts"]
                        ],
                    }
                    failed_grasp_pose_rows.append(failed_pose_entry)

            _append_jsonl(all_results_path, episode_rows)
            _append_jsonl(failures_path, failure_rows)
            total_attempts += len(episode_rows)
            episode_successes = sum(1 for item in episode_rows if bool(item["success"]))
            total_successes += episode_successes
            failures_written += len(failure_rows)
            success_rate = float(total_successes) / float(total_attempts) if total_attempts else 0.0
            print(
                f"[EVAL] episode={episode_idx + 1}/{args_cli.num_episodes} "
                f"episode_successes={episode_successes}/{len(episode_rows)} "
                f"total_success_rate={success_rate * 100.0:.2f}% "
                f"failures_written={failures_written}",
                flush=True,
            )
    finally:
        env.close()
        simulation_app.close()
    elapsed_s = time.monotonic() - start_time
    payload = {
        "runtime_spec": runtime_spec_path,
        "task": args_cli.task,
        "worker_id": int(args_cli.worker_id),
        "num_workers": int(args_cli.num_workers),
        "seed": None if args_cli.seed is None else int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "num_episodes": int(args_cli.num_episodes),
        "attempts": int(total_attempts),
        "successes": int(total_successes),
        "failures": int(total_attempts - total_successes),
        "success_rate": float(total_successes) / float(total_attempts) if total_attempts else 0.0,
        "elapsed_s": float(elapsed_s),
        "failures_jsonl": str(failures_path),
        "all_results_jsonl": None if all_results_path is None else str(all_results_path),
        "record_video": bool(args_cli.record_video),
        "video_dir": str(video_output_dir) if args_cli.record_video else None,
        "post_grasp_motion": args_cli.post_grasp_motion,
        "pre_transport_lift": bool(args_cli.pre_transport_lift),
        "use_curobo_goal_transport": bool(args_cli.use_curobo_goal_transport),
        "graspgen_host": args_cli.graspgen_host,
        "graspgen_port": int(args_cli.graspgen_port),
        "num_grasps": int(args_cli.num_grasps),
        "topk_num_grasps": int(args_cli.topk_num_grasps),
        "table_collision_filter_enabled": not bool(args_cli.no_table_collision_filter),
        "table_collision_clearance": float(args_cli.table_collision_clearance),
        "lift_success_height_delta": float(args_cli.lift_success_height_delta),
        "lift_success_stable_seconds": float(args_cli.lift_success_stable_seconds),
        "lift_success_stability_tolerance": float(args_cli.lift_success_stability_tolerance),
        "stability_hold_steps": int(_stable_hold_steps()),
        "max_candidate_attempts": int(args_cli.max_candidate_attempts),
        "failed_object_names": sorted(failed_object_names),
        "failed_grasp_pose_rows": failed_grasp_pose_rows,
        "object_count_loaded": len(object_names),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("\n========== GraspGen Lift Grasp Eval ==========")
    print(f"Attempts: {total_attempts}")
    print(f"Successes: {total_successes}")
    print(f"Failures: {total_attempts - total_successes}")
    print(f"Success Rate: {payload['success_rate'] * 100.0:.2f}%")
    print(f"Failed Object Names: {payload['failed_object_names']}")
    print(f"Failed Grasp Pose Rows: {len(payload['failed_grasp_pose_rows'])}")
    print(f"Summary: {summary_path}")
    print(f"Failures: {failures_path}")
    if all_results_path is not None:
        print(f"All Results: {all_results_path}")
    print("=============================================\n")


if __name__ == "__main__":
    main()

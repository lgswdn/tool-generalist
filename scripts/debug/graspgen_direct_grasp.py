#!/usr/bin/env python3
"""Drive Panda grippers to GraspGen grasps in the unstable object task."""

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


DEFAULT_CONFIG = "configs/experiments/single_unstable_diff_post.py"
DEFAULT_GRASPGEN_ROOT = "/mnt/project/world_model/tool_generalist/GraspGen"
DEFAULT_GRASPGEN_PORT = 5556
DEFAULT_ACTION_SCALE = 0.1
FFMPEG_PATH = "/usr/bin/ffmpeg"
MODE_NAME = "graspgen_direct_grasp"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Reset the single-arm unstable task, ask GraspGen for object grasps, "
            "and servo every official Panda gripper to its selected grasp pose."
        )
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Experiment config exposing EXP_CFG.")
    source_group.add_argument("--runtime_spec", type=str, help="Existing rl_runtime_spec.json.")
    parser.add_argument("--task", type=str, default=None, help="Task id. Defaults to runtime spec task_id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime seed override.")
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
    replay_source_group = parser.add_mutually_exclusive_group()
    replay_source_group.add_argument(
        "--lift_eval_failures_jsonl",
        type=str,
        default=None,
        help=(
            "Replay failure rows from scripts/eval_graspgen_lift_grasps.py instead of calling GraspGen. "
            "The script builds a temporary object manifest from the rows, restores each logged initial "
            "object pose, and executes failed_attempts[--grasp_index]."
        ),
    )
    replay_source_group.add_argument(
        "--grasp_candidates_jsonl",
        type=str,
        default=None,
        help=(
            "Replay full_safe_candidates from scripts/generate_graspgen_candidates_from_pointclouds.py. "
            "The script restores each logged initial object pose and does not call GraspGen again."
        ),
    )
    parser.add_argument(
        "--max_lift_eval_failures",
        type=int,
        default=0,
        help="Maximum JSONL rows to replay. 0 replays every row from the selected replay source.",
    )
    parser.add_argument(
        "--single_lift_eval_grasp",
        action="store_true",
        help=(
            "In JSONL replay mode, execute only the candidate at --grasp_index. By default candidates "
            "are replayed sequentially with one video per object and candidate."
        ),
    )
    parser.add_argument(
        "--max_lift_eval_grasps_per_episode",
        type=int,
        default=0,
        help=(
            "Maximum candidates to replay per JSONL row/episode. 0 replays every available candidate."
        ),
    )

    parser.add_argument("--record_video", action="store_true", help="Record one MP4 per environment.")
    parser.add_argument("--video_dir", type=str, default="videos/graspgen_direct_grasp")
    parser.add_argument("--video_length", type=int, default=0, help="Maximum frames to record. 0 records the full scripted motion.")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument(
        "--record_camera_eye",
        type=float,
        nargs=3,
        default=(0.55, -0.72, 0.28),
        metavar=("X", "Y", "Z"),
        help="Per-environment recording camera position. Defaults to a near-horizontal side view.",
    )
    parser.add_argument(
        "--record_camera_target",
        type=float,
        nargs=3,
        default=(0.55, 0.0, 0.15),
        metavar=("X", "Y", "Z"),
        help="Per-environment world-space point viewed by the recording camera.",
    )
    parser.add_argument("--record_camera_focal_length", type=float, default=16.0)
    parser.add_argument("--record_camera_horizontal_aperture", type=float, default=28.0)
    parser.add_argument("--no_visualize_grasps", action="store_true", help="Do not render selected GraspGen grasp guides.")
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
    parser.add_argument("--no_visualize_object_cloud", action="store_true", help="Do not render the object point cloud sent to GraspGen.")
    parser.add_argument("--object_cloud_vis_points", type=int, default=128, help="Maximum object cloud points to render per env.")
    parser.add_argument("--object_cloud_point_size", type=float, default=0.006, help="Rendered cube size for each object cloud point.")
    parser.add_argument(
        "--visualize_object_markers",
        action="store_true",
        help="Show current object pose and target object pose frame markers from the Isaac task.",
    )
    parser.add_argument(
        "--record_warmup",
        action="store_true",
        help="Start recording at reset/open warmup instead of after GraspGen inference.",
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
    parser.add_argument(
        "--restore_settle_steps",
        type=int,
        default=2,
        help="Open-gripper settling steps after restoring a lift-eval replay initial state before each grasp.",
    )
    parser.add_argument("--approach_steps", type=int, default=80)
    parser.add_argument("--grasp_steps", type=int, default=50)
    parser.add_argument("--close_steps", type=int, default=35)
    parser.add_argument(
        "--post_close_hold_steps",
        type=int,
        default=10,
        help="Control steps to hold the closed grasp pose before lift or goal transport.",
    )
    parser.add_argument("--lift_steps", type=int, default=60)
    parser.add_argument("--goal_transport_steps", type=int, default=80)
    parser.add_argument("--hold_steps", type=int, default=20)
    parser.add_argument("--pregrasp_offset", type=float, default=0.10)
    parser.add_argument("--lift_distance", type=float, default=0.12)
    parser.add_argument("--max_pos_step", type=float, default=0.035)
    parser.add_argument("--max_rot_step", type=float, default=0.25)
    parser.add_argument(
        "--panda_finger_friction",
        type=float,
        default=None,
        help="Override both static and dynamic friction coefficients on the two Panda finger links.",
    )
    parser.add_argument(
        "--panda_finger_static_friction",
        type=float,
        default=None,
        help="Override Panda finger static friction; must be used with --panda_finger_dynamic_friction.",
    )
    parser.add_argument(
        "--panda_finger_dynamic_friction",
        type=float,
        default=None,
        help="Override Panda finger dynamic friction; must be used with --panda_finger_static_friction.",
    )
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
        help="After grasp close, use cuRobo to move the grasped object toward target_object_pose.",
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


def _parse_manifest_entry(item: str) -> tuple[str, float | None]:
    if "-" not in item:
        return item, None
    base, scale_str = item.rsplit("-", 1)
    try:
        return base, float(scale_str)
    except ValueError:
        return item, None


def _resolve_panda_finger_friction(args: argparse.Namespace) -> tuple[float, float] | None:
    shared = args.panda_finger_friction
    static = args.panda_finger_static_friction
    dynamic = args.panda_finger_dynamic_friction
    if shared is not None and (static is not None or dynamic is not None):
        raise ValueError(
            "--panda_finger_friction cannot be combined with the separate static/dynamic friction options"
        )
    if (static is None) != (dynamic is None):
        raise ValueError(
            "--panda_finger_static_friction and --panda_finger_dynamic_friction must be provided together"
        )
    if shared is not None:
        static = dynamic = float(shared)
    if static is None:
        return None
    static = float(static)
    dynamic = float(dynamic)
    if not math.isfinite(static) or not math.isfinite(dynamic):
        raise ValueError("Panda finger friction coefficients must be finite")
    if static < 0.0 or dynamic < 0.0:
        raise ValueError("Panda finger friction coefficients must be non-negative")
    if dynamic > static:
        raise ValueError(
            f"Panda finger dynamic friction ({dynamic}) must not exceed static friction ({static})"
        )
    return static, dynamic


def _require_vector(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must be a {length}-vector, got {values!r}")
    return [float(item) for item in values]


def _lift_eval_failed_grasps(row: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    record = row.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"lift-eval failure row {row_index} is missing record")
    failed_grasps = record.get("failed_attempts", [])
    if not isinstance(failed_grasps, list):
        raise ValueError(f"lift-eval failure row {row_index} has non-list record.failed_attempts")
    if not failed_grasps:
        candidate_generation = record.get("candidate_generation", {})
        if isinstance(candidate_generation, dict):
            all_candidates = candidate_generation.get("all_candidates", [])
            if isinstance(all_candidates, list):
                failed_grasps = all_candidates
    return [item for item in failed_grasps if isinstance(item, dict)]


def _load_lift_eval_failure_cases(path: str, max_cases: int) -> list[dict[str, Any]]:
    if max_cases < 0:
        raise ValueError(f"--max_lift_eval_failures must be >= 0, got {max_cases}")
    source_path = Path(path).expanduser().resolve()
    cases: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for row_index, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or bool(row.get("success", False)):
                continue
            record = row.get("record")
            if not isinstance(record, dict):
                continue
            initial_state = record.get("initial_state")
            if not isinstance(initial_state, dict):
                raise ValueError(f"lift-eval failure row {row_index} is missing record.initial_state")
            failed_grasps = _lift_eval_failed_grasps(row, row_index)
            if not failed_grasps:
                raise ValueError(f"lift-eval failure row {row_index} has no failed grasps to replay")
            object_name = row.get("object_name")
            if not isinstance(object_name, str) or not object_name:
                raise ValueError(f"lift-eval failure row {row_index} is missing object_name")
            cases.append(
                {
                    "source_path": str(source_path),
                    "source_row_index": int(row_index),
                    "worker_id": row.get("worker_id"),
                    "episode_index": row.get("episode_index"),
                    "sample_index": row.get("sample_index"),
                    "env_id": row.get("env_id"),
                    "object_index": row.get("object_index"),
                    "object_name": object_name,
                    "initial_object_pos_w": _require_vector(initial_state.get("object_root_pos_w"), 3, "object_root_pos_w"),
                    "initial_object_quat_wxyz": _require_vector(
                        initial_state.get("object_root_quat_wxyz"), 4, "object_root_quat_wxyz"
                    ),
                    "initial_robot_pos_w": _require_vector(initial_state.get("robot_root_pos_w"), 3, "robot_root_pos_w")
                    if initial_state.get("robot_root_pos_w") is not None
                    else None,
                    "initial_object_scale_xyz": _require_vector(
                        initial_state.get("object_scale_xyz"), 3, "object_scale_xyz"
                    )
                    if initial_state.get("object_scale_xyz") is not None
                    else None,
                    "failed_grasps": failed_grasps,
                    "raw": row,
                }
            )
            if max_cases > 0 and len(cases) >= max_cases:
                break
    if not cases:
        raise RuntimeError(f"No lift-eval failure rows found in {source_path}")
    return cases


def _load_generated_grasp_candidate_cases(path: str, max_cases: int) -> list[dict[str, Any]]:
    if max_cases < 0:
        raise ValueError(f"--max_lift_eval_failures must be >= 0, got {max_cases}")
    source_path = Path(path).expanduser().resolve()
    cases: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for row_index, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"grasp-candidate row {row_index} is not a JSON object")
            initial_state = row.get("initial_state")
            if not isinstance(initial_state, dict):
                raise ValueError(f"grasp-candidate row {row_index} is missing initial_state")
            candidate_generation = row.get("candidate_generation")
            if not isinstance(candidate_generation, dict):
                raise ValueError(f"grasp-candidate row {row_index} is missing candidate_generation")
            candidates = candidate_generation.get("full_safe_candidates")
            if not isinstance(candidates, list):
                raise ValueError(
                    f"grasp-candidate row {row_index} has non-list candidate_generation.full_safe_candidates"
                )
            candidates = [item for item in candidates if isinstance(item, dict)]
            if not candidates:
                raise ValueError(f"grasp-candidate row {row_index} has no full-safe candidates to replay")
            object_name = row.get("object_name")
            if not isinstance(object_name, str) or not object_name:
                raise ValueError(f"grasp-candidate row {row_index} is missing object_name")
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            cases.append(
                {
                    "source_path": str(source_path),
                    "source_row_index": int(row_index),
                    "worker_id": source.get("worker_id"),
                    "episode_index": source.get("episode_index"),
                    "sample_index": source.get("sample_index"),
                    "env_id": source.get("env_id"),
                    "object_index": row.get("object_index"),
                    "object_name": object_name,
                    "initial_object_pos_w": _require_vector(
                        initial_state.get("object_root_pos_w"), 3, "object_root_pos_w"
                    ),
                    "initial_object_quat_wxyz": _require_vector(
                        initial_state.get("object_root_quat_wxyz"), 4, "object_root_quat_wxyz"
                    ),
                    "initial_robot_pos_w": _require_vector(
                        initial_state.get("robot_root_pos_w"), 3, "robot_root_pos_w"
                    )
                    if initial_state.get("robot_root_pos_w") is not None
                    else None,
                    "initial_object_scale_xyz": _require_vector(
                        initial_state.get("object_scale_xyz"), 3, "object_scale_xyz"
                    )
                    if initial_state.get("object_scale_xyz") is not None
                    else None,
                    "failed_grasps": candidates,
                    "replay_source_kind": "generated_grasp_candidates",
                    "raw": row,
                }
            )
            if max_cases > 0 and len(cases) >= max_cases:
                break
    if not cases:
        raise RuntimeError(f"No generated grasp-candidate rows found in {source_path}")
    return cases


def _raw_paths_yaml_for_failure_replay(args: argparse.Namespace) -> Path:
    if args.paths_yaml is not None:
        return Path(args.paths_yaml).expanduser().resolve()
    if Path("configs/paths/panda_hand.yaml").is_file():
        return Path("configs/paths/panda_hand.yaml").resolve()
    return Path("configs/paths/default.yaml").resolve()


def _materialize_failure_replay_paths(args: argparse.Namespace, cases: list[dict[str, Any]]) -> tuple[Path, Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for JSONL replay mode") from exc

    source_paths_yaml = _raw_paths_yaml_for_failure_replay(args)
    with source_paths_yaml.open("r", encoding="utf-8") as f:
        paths_data = yaml.safe_load(f)
    if not isinstance(paths_data, dict):
        raise ValueError(f"Expected YAML mapping in {source_paths_yaml}")
    dgn = paths_data.get("dgn")
    if not isinstance(dgn, dict):
        raise ValueError(f"{source_paths_yaml} does not contain a dgn mapping")
    source_manifest_path = Path(str(dgn.get("candidates_json"))).expanduser()
    with source_manifest_path.open("r", encoding="utf-8") as f:
        source_manifest = json.load(f)
    if not isinstance(source_manifest, list) or not all(isinstance(item, str) for item in source_manifest):
        raise ValueError(f"Expected JSON list of object manifest entries: {source_manifest_path}")

    item_by_base: dict[str, str] = {}
    for item in source_manifest:
        base, _ = _parse_manifest_entry(item)
        item_by_base.setdefault(base, item)

    seen: set[str] = set()
    # Use the dictionary manifest schema so the JSONL scale is applied by the
    # object spawner before PhysX starts.  Setting xformOp:scale after env.reset
    # is too late for an already-created rigid body and can leave the rendered
    # prim, collision shapes, and cached scale out of sync.
    replay_manifest: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        object_name = str(case["object_name"])
        if object_name in seen:
            raise ValueError(
                "Failure replay requires unique object_name values because the Isaac object loader "
                f"deduplicates object assets; duplicate {object_name!r} at replay case {case_index}."
            )
        seen.add(object_name)
        try:
            source_item = item_by_base[object_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Could not find object_name={object_name!r} from failure JSONL in {source_manifest_path}"
            ) from exc

        scale_xyz = case.get("initial_object_scale_xyz")
        if scale_xyz is None:
            _, source_scale = _parse_manifest_entry(source_item)
            if source_scale is None:
                raise RuntimeError(
                    f"Replay case {case_index} for {object_name!r} has no initial object scale, "
                    f"and its source manifest entry has no usable scale: {source_item!r}"
                )
            spawn_scale = float(source_scale)
        else:
            values = [float(value) for value in scale_xyz]
            if len(values) != 3 or not all(math.isfinite(value) and value > 0.0 for value in values):
                raise ValueError(
                    f"Replay case {case_index} for {object_name!r} has invalid object scale {scale_xyz!r}"
                )
            if not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
                values[0], values[2], rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"Replay case {case_index} for {object_name!r} requires uniform object scale, "
                    f"got {scale_xyz!r}"
                )
            spawn_scale = values[0]

        case["replay_spawn_scale"] = spawn_scale
        replay_manifest.append({"object": object_name, "scale": spawn_scale})

    replay_dir = Path(tempfile.gettempdir()) / "tool_generalist_graspgen_direct_grasp" / "lift_eval_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest_path = replay_dir / f"objects_{stamp}_{len(cases)}.json"
    paths_path = replay_dir / f"paths_{stamp}_{len(cases)}.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(replay_manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    paths_data = copy.deepcopy(paths_data)
    paths_data.setdefault("dgn", {})["candidates_json"] = str(manifest_path)
    paths_data.setdefault("objects", {})["candidates_json"] = str(manifest_path)
    with paths_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(paths_data, f, sort_keys=False)

    print(
        f"[LIFT_EVAL_REPLAY] cases={len(cases)} "
        f"source={args.lift_eval_failures_jsonl or args.grasp_candidates_jsonl} "
        f"manifest={manifest_path} paths={paths_path}",
        flush=True,
    )
    return paths_path, manifest_path


def _resolve_encoder_checkpoint_for_config(cfg, config: str) -> str | None:
    try:
        resolved = _resolve_initial_encoder_checkpoint(cfg, config_source=config)
    except RuntimeError as exc:
        print(
            "[WARNING] could not resolve pretrained encoder checkpoint; "
            f"continuing without one for scripted {MODE_NAME}: {exc}",
            flush=True,
        )
        resolved = None
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


def _build_runtime_spec_from_config(
    config: str,
    num_envs: int,
    seed: int | None,
    paths_yaml: str | None,
    runtime_objects_manifest: str | None = None,
) -> dict[str, Any]:
    cfg = load_exp_cfg(config)
    _apply_panda_gripper_overrides(cfg, paths_yaml, seed)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifact_dir = Path(tempfile.gettempdir()) / "tool_generalist_graspgen_direct_grasp" / Path(config).stem
    spec = build_rl_runtime_spec(
        cfg,
        paths,
        artifact_dir,
        mode=MODE_NAME,
        encoder_checkpoint_override=_resolve_encoder_checkpoint_for_config(cfg, config),
        runtime_num_envs=num_envs,
        runtime_objects_manifest=runtime_objects_manifest,
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
    spec_dir = Path(tempfile.gettempdir()) / "tool_generalist_graspgen_direct_grasp"
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


def _object_asset_debug_rows(num_envs: int) -> dict[str, Any]:
    try:
        import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool as env_tool_module

        rows = []
        for env_id in range(int(num_envs)):
            object_index = int(env_tool_module.get_object_index_for_env(env_id))
            cfg = env_tool_module.get_object_asset_cfg_for_env(env_id)
            obj_path = getattr(cfg, "obj_path", None)
            usd_path = getattr(cfg, "usd_path", None)
            object_name = None
            if obj_path:
                object_name = Path(str(obj_path)).stem
            elif usd_path:
                object_name = Path(str(usd_path)).stem
            rows.append(
                {
                    "env_id": env_id,
                    "object_index": object_index,
                    "object_name": object_name,
                    "obj_path": str(obj_path) if obj_path else None,
                    "usd_path": str(usd_path) if usd_path else None,
                }
            )
        paths = getattr(env_tool_module, "_PATHS", {})
        dgn = paths.get("dgn", {}) if isinstance(paths, dict) else {}
        return {
            "paths_yaml": getattr(env_tool_module, "_PATHS_CFG_FILE", None),
            "dgn_candidates_json": dgn.get("candidates_json") if isinstance(dgn, dict) else None,
            "num_loaded_object_assets": len(getattr(env_tool_module, "OBJECT_ASSET_CFGS", [])),
            "object_asset_indices_by_env": list(getattr(env_tool_module, "OBJECT_ASSET_INDICES_BY_ENV", [])),
            "object_spawn_asset_indices": list(getattr(env_tool_module, "OBJECT_SPAWN_ASSET_INDICES", [])),
            "rows": rows,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _validate_lift_eval_replay_object_assets(num_envs: int) -> None:
    if not _LIFT_EVAL_FAILURE_CASES:
        return
    debug = _object_asset_debug_rows(num_envs)
    if "error" in debug:
        raise RuntimeError(f"Could not inspect loaded replay objects: {debug['error']}")
    rows = debug.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Could not inspect loaded replay objects: malformed debug rows {type(rows).__name__}")
    mismatches = []
    for env_id, case in enumerate(_LIFT_EVAL_FAILURE_CASES[: int(num_envs)]):
        actual = rows[env_id].get("object_name") if env_id < len(rows) and isinstance(rows[env_id], dict) else None
        expected = str(case["object_name"])
        if actual != expected:
            mismatches.append((env_id, expected, actual))
    if mismatches:
        preview = "; ".join(
            f"env={env_id} expected={expected} actual={actual}" for env_id, expected, actual in mismatches[:8]
        )
        raise RuntimeError(
            "Lift-eval replay object mismatch between JSONL cases and Isaac-loaded assets. "
            f"manifest={debug.get('dgn_candidates_json')} mismatches={preview}"
        )
    print(
        f"[LIFT_EVAL_REPLAY] verified loaded objects match JSONL cases: "
        f"{len(_LIFT_EVAL_FAILURE_CASES[: int(num_envs)])} envs manifest={debug.get('dgn_candidates_json')}",
        flush=True,
    )


args_cli, hydra_args = _parse_args()
_PANDA_FINGER_FRICTION = _resolve_panda_finger_friction(args_cli)

_LIFT_EVAL_FAILURE_CASES: list[dict[str, Any]] = []
_LIFT_EVAL_REPLAY_MANIFEST: Path | None = None
if args_cli.lift_eval_failures_jsonl or args_cli.grasp_candidates_jsonl:
    if int(args_cli.restore_settle_steps) < 0:
        raise ValueError(f"--restore_settle_steps must be >= 0, got {args_cli.restore_settle_steps}")
    if int(args_cli.max_lift_eval_grasps_per_episode) < 0:
        raise ValueError(
            "--max_lift_eval_grasps_per_episode must be >= 0, "
            f"got {args_cli.max_lift_eval_grasps_per_episode}"
        )
    if args_cli.grasp_candidates_jsonl:
        _LIFT_EVAL_FAILURE_CASES = _load_generated_grasp_candidate_cases(
            args_cli.grasp_candidates_jsonl,
            int(args_cli.max_lift_eval_failures),
        )
    else:
        _LIFT_EVAL_FAILURE_CASES = _load_lift_eval_failure_cases(
            args_cli.lift_eval_failures_jsonl,
            int(args_cli.max_lift_eval_failures),
        )
    replay_paths_yaml, replay_manifest = _materialize_failure_replay_paths(args_cli, _LIFT_EVAL_FAILURE_CASES)
    _LIFT_EVAL_REPLAY_MANIFEST = replay_manifest
    args_cli.paths_yaml = str(replay_paths_yaml)
    if args_cli.num_envs != len(_LIFT_EVAL_FAILURE_CASES):
        print(
            f"[LIFT_EVAL_REPLAY] overriding --num_envs {args_cli.num_envs} -> "
            f"{len(_LIFT_EVAL_FAILURE_CASES)} to match replay cases",
            flush=True,
        )
        args_cli.num_envs = len(_LIFT_EVAL_FAILURE_CASES)
    os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = "0"
    os.environ["TOOL_GENERALIST_LOCAL_RANK"] = "0"
    os.environ["TOOL_GENERALIST_WORLD_SIZE"] = "1"

if args_cli.runtime_spec:
    runtime_spec = _load_runtime_spec(args_cli.runtime_spec, args_cli.num_envs, args_cli.seed, args_cli.paths_yaml)
else:
    runtime_spec = _build_runtime_spec_from_config(
        args_cli.config,
        args_cli.num_envs,
        args_cli.seed,
        args_cli.paths_yaml,
        str(_LIFT_EVAL_REPLAY_MANIFEST) if _LIFT_EVAL_REPLAY_MANIFEST is not None else None,
    )

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
if _LIFT_EVAL_REPLAY_MANIFEST is not None:
    print(
        f"[LIFT_EVAL_REPLAY] runtime_paths={runtime_spec['paths_yaml']} "
        f"runtime_manifest={_LIFT_EVAL_REPLAY_MANIFEST}",
        flush=True,
    )

args_cli.enable_cameras = bool(args_cli.record_video or getattr(args_cli, "enable_cameras", False))
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(
    f"[STARTUP] AppLauncher ready simulation_app_running={bool(simulation_app.is_running())}",
    flush=True,
)

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
from isaaclab_tasks.utils.hydra import hydra_task_config
print("[STARTUP] IsaacLab task modules imported", flush=True)


def _panda_finger_body_cfg(base_env) -> SceneEntityCfg:
    finger_body_cfg = SceneEntityCfg(
        "robot",
        body_names=["panda_leftfinger", "panda_rightfinger"],
    )
    finger_body_cfg.resolve(base_env.scene)
    if len(finger_body_cfg.body_ids) != 2:
        raise RuntimeError(
            "Expected Panda finger bodies panda_leftfinger and panda_rightfinger, "
            f"got {finger_body_cfg.body_names!r}"
        )
    return finger_body_cfg


def _material_shape_indices_for_bodies(asset, body_ids: list[int]) -> list[int]:
    cache = getattr(asset, "_graspgen_material_shape_indices", None)
    if cache is None:
        cache = {}
        setattr(asset, "_graspgen_material_shape_indices", cache)
    cache_key = tuple(int(body_id) for body_id in body_ids)
    if cache_key in cache:
        return list(cache[cache_key])

    num_shapes_per_body: list[int] = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        num_shapes_per_body.append(int(link_view.max_shapes))
    expected_shapes = int(asset.root_physx_view.max_shapes)
    if sum(num_shapes_per_body) != expected_shapes:
        raise RuntimeError(
            "Could not map Panda articulation bodies to material shapes: "
            f"expected={expected_shapes}, parsed={sum(num_shapes_per_body)}, "
            f"per_body={num_shapes_per_body}"
        )
    shape_indices: list[int] = []
    for body_id in body_ids:
        start = sum(num_shapes_per_body[: int(body_id)])
        shape_indices.extend(range(start, start + num_shapes_per_body[int(body_id)]))
    if not shape_indices:
        raise RuntimeError(f"Panda finger bodies have no collision material shapes: body_ids={body_ids}")
    cache[cache_key] = tuple(shape_indices)
    return shape_indices


def _panda_finger_material_friction(base_env) -> np.ndarray:
    robot = base_env.scene["robot"]
    finger_body_cfg = _panda_finger_body_cfg(base_env)
    shape_indices = _material_shape_indices_for_bodies(robot, list(finger_body_cfg.body_ids))
    properties = robot.root_physx_view.get_material_properties()
    if isinstance(properties, torch.Tensor):
        values = properties.detach().cpu().numpy()
    else:
        values = np.asarray(properties)
    return np.asarray(values[:, shape_indices, :2].mean(axis=1), dtype=np.float32)


def _override_panda_finger_friction(base_env) -> None:
    if _PANDA_FINGER_FRICTION is None:
        return
    static_friction, dynamic_friction = _PANDA_FINGER_FRICTION
    robot = base_env.scene["robot"]
    finger_body_cfg = _panda_finger_body_cfg(base_env)
    shape_indices = _material_shape_indices_for_bodies(robot, list(finger_body_cfg.body_ids))
    env_ids_cpu = torch.arange(int(base_env.num_envs), dtype=torch.long, device="cpu")
    properties = robot.root_physx_view.get_material_properties().clone()
    properties[env_ids_cpu[:, None], shape_indices, 0] = float(static_friction)
    properties[env_ids_cpu[:, None], shape_indices, 1] = float(dynamic_friction)
    robot.root_physx_view.set_material_properties(properties, env_ids_cpu)
    actual = _panda_finger_material_friction(base_env)
    print(
        "[PANDA_FINGER_MATERIAL] "
        f"bodies={finger_body_cfg.body_names} shapes={shape_indices} "
        f"static_friction={static_friction:.6g} dynamic_friction={dynamic_friction:.6g} "
        f"readback_static=[{actual[:, 0].min():.6g},{actual[:, 0].max():.6g}] "
        f"readback_dynamic=[{actual[:, 1].min():.6g},{actual[:, 1].max():.6g}]",
        flush=True,
    )


def _make_record_camera_cfg() -> TiledCameraCfg:
    eye = np.asarray(args_cli.record_camera_eye, dtype=np.float64)
    target = np.asarray(args_cli.record_camera_target, dtype=np.float64)
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-8:
        raise ValueError("--record_camera_eye and --record_camera_target must be different")
    forward /= forward_norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        raise ValueError("Recording camera viewing direction cannot be parallel to world Z")
    right /= right_norm
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=1)
    camera_quat_wxyz = _matrix_to_quat_wxyz(rotation)
    focal_length = float(args_cli.record_camera_focal_length)
    horizontal_aperture = float(args_cli.record_camera_horizontal_aperture)
    if focal_length <= 0.0 or horizontal_aperture <= 0.0:
        raise ValueError("Recording camera focal length and horizontal aperture must be positive")
    horizontal_fov_deg = math.degrees(2.0 * math.atan(horizontal_aperture / (2.0 * focal_length)))
    print(
        f"[CAMERA] eye={eye.tolist()} target={target.tolist()} "
        f"horizontal_fov_deg={horizontal_fov_deg:.2f}",
        flush=True,
    )
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/GraspRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in eye),
            rot=tuple(float(value) for value in camera_quat_wxyz),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=forward_norm,
            horizontal_aperture=horizontal_aperture,
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


def _mean_material_friction(asset) -> np.ndarray:
    properties = asset.root_physx_view.get_material_properties()
    if isinstance(properties, torch.Tensor):
        values = properties.detach().cpu().numpy()
    else:
        values = np.asarray(properties)
    if values.ndim != 3 or values.shape[2] < 2:
        raise RuntimeError(f"Unexpected material property shape: {values.shape}")
    return np.asarray(values[..., :2].mean(axis=1), dtype=np.float32)


def _video_friction_values(base_env) -> list[dict[str, float]]:
    object_friction = _mean_material_friction(base_env.scene["object"])
    finger_friction = _panda_finger_material_friction(base_env)
    if object_friction.shape[0] != int(base_env.num_envs) or finger_friction.shape[0] != int(base_env.num_envs):
        raise RuntimeError(
            "Material property environment count does not match the recording environment count: "
            f"object={object_friction.shape}, finger={finger_friction.shape}, num_envs={base_env.num_envs}"
        )
    return [
        {
            "finger_static": float(finger_friction[env_id, 0]),
            "finger_dynamic": float(finger_friction[env_id, 1]),
            "object_static": float(object_friction[env_id, 0]),
            "object_dynamic": float(object_friction[env_id, 1]),
        }
        for env_id in range(int(base_env.num_envs))
    ]


def _init_video_state(
    output_dir: Path,
    base_env,
    run_label: str | None = None,
) -> dict[str, Any] | None:
    if not args_cli.record_video:
        return None
    num_envs = int(base_env.num_envs)
    friction_values = _video_friction_values(base_env)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label_suffix = f"_{run_label}" if run_label else ""
    run_dir = output_dir / f"graspgen_direct_grasp_{timestamp}{label_suffix}"
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
        "friction_values": friction_values,
        "stage": "initial",
        "stage_step": None,
        "stage_steps": None,
        "grasp_target_pos_w": None,
    }


def _video_stage_text(label: str) -> str:
    if label.startswith("replay_grasp_"):
        parts = label.split("_", 3)
        if len(parts) == 4 and parts[2].isdigit():
            return parts[3]
    return label


def _overlay_video_frame(
    frame: np.ndarray,
    friction: dict[str, float],
    stage: str,
    eef_to_grasp_distance_m: float | None,
) -> None:
    import cv2

    band_height = min(78, int(frame.shape[0]))
    if band_height <= 0:
        return
    dark = np.zeros_like(frame[:band_height])
    frame[:band_height] = cv2.addWeighted(frame[:band_height], 0.30, dark, 0.70, 0.0)
    distance_text = (
        "N/A" if eef_to_grasp_distance_m is None else f"{eef_to_grasp_distance_m:.4f} m"
    )
    lines = (
        f"Finger mu: static={friction['finger_static']:.3f}  dynamic={friction['finger_dynamic']:.3f}",
        f"Object mu: static={friction['object_static']:.3f}  dynamic={friction['object_dynamic']:.3f}",
        f"Stage: {stage}  EEF->grasp target: {distance_text}",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 19 + 24 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _capture_video_frames(
    env,
    video_state: dict[str, Any] | None,
    *,
    stage: str | None = None,
    stage_step: int | None = None,
    stage_steps: int | None = None,
) -> None:
    if video_state is None:
        return
    if stage is not None:
        video_state["stage"] = _video_stage_text(stage)
        video_state["stage_step"] = stage_step
        video_state["stage_steps"] = stage_steps
    env.unwrapped.sim.render()
    camera = env.unwrapped.scene["grasp_record_camera"]
    camera.update(dt=0.0, force_recompute=True)
    rgb_all = camera.data.output["rgb"]
    max_frames = int(args_cli.video_length)
    if max_frames > 0 and int(video_state["frames"]) >= max_frames:
        return
    grasp_target_pos_w = video_state.get("grasp_target_pos_w")
    if grasp_target_pos_w is None:
        eef_to_grasp_distances = None
    else:
        hand_pos_w, _ = _current_hand_pose_w(env.unwrapped)
        target_pos_w = torch.as_tensor(
            grasp_target_pos_w,
            dtype=hand_pos_w.dtype,
            device=hand_pos_w.device,
        )
        eef_to_grasp_distances = torch.linalg.norm(hand_pos_w - target_pos_w, dim=-1).detach().cpu().numpy()
    for env_id, writer in enumerate(video_state["writers"]):
        if writer.stdin is None:
            continue
        frame_tensor = rgb_all[env_id, ..., :3].detach().cpu()
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = torch.clamp(frame_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
        frame = frame_tensor.contiguous().numpy().copy()
        stage_text = str(video_state["stage"])
        if video_state["stage_step"] is not None and video_state["stage_steps"] is not None:
            stage_text += f" {video_state['stage_step']}/{video_state['stage_steps']}"
        distance_m = None if eef_to_grasp_distances is None else float(eef_to_grasp_distances[env_id])
        _overlay_video_frame(
            frame,
            video_state["friction_values"][env_id],
            stage_text,
            distance_m,
        )
        writer.stdin.write(frame.tobytes())
    video_state["frames"] = int(video_state["frames"]) + 1


def _close_video_state(video_state: dict[str, Any] | None) -> None:
    if video_state is None:
        return
    errors = []
    for writer in video_state["writers"]:
        try:
            _close_ffmpeg_writer(writer)
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
    finger_open_action: float,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    *,
    steps: int,
    label: str,
    finger_action_end: float | None = None,
) -> None:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    ee_body_id = arm_cfg.body_ids[0]
    jacobian_body_id = _resolve_fixed_base_jacobian_index(robot, ee_body_id)
    action_scale = _action_scale_tensor(base_env.device)
    num_steps = int(steps)

    for step in range(num_steps):
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
        if finger_action_end is None:
            finger_action = float(finger_open_action)
        elif num_steps <= 1:
            finger_action = float(finger_action_end)
        else:
            alpha = float(step) / float(num_steps - 1)
            finger_action = (1.0 - alpha) * float(finger_open_action) + alpha * float(finger_action_end)
        action[:, 7] = finger_action
        _, _, terminated, truncated, _ = env.step(action)
        _capture_video_frames(
            env,
            video_state,
            stage=label,
            stage_step=step + 1,
            stage_steps=num_steps,
        )
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == num_steps:
            ee_state_w = robot.data.body_state_w[:, ee_body_id]
            pos_err, rot_err = compute_pose_error(
                ee_state_w[:, :3],
                ee_state_w[:, 3:7],
                target_pos_w,
                target_quat_w,
                rot_error_type="axis_angle",
            )
            print(
                f"[MOTION] {label} step={step + 1}/{num_steps} "
                f"finger_action={finger_action:.3f} "
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
        _capture_video_frames(
            env,
            video_state,
            stage=label,
            stage_step=step + 1,
            stage_steps=max_len,
        )
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == max_len:
            print(f"[MOTION][CUROBO] {label} step={step + 1}/{max_len}", flush=True)
        if bool(torch.any(terminated | truncated).detach().cpu()):
            print(f"[WARNING] termination/truncation observed during {label}; continuing without reset", flush=True)
    return True


def _step_curobo_pregrasp(
    env,
    arm_cfg: SceneEntityCfg,
    video_state: dict[str, Any] | None,
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
) -> bool:
    base_env = env.unwrapped
    target_hand_pos_w, target_hand_quat_w = _target_hand_pose_for_object_goal(base_env, arm_cfg)
    used_curobo = _step_curobo_pose(
        env,
        arm_cfg,
        video_state,
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
            -1.0,
            target_hand_pos_w,
            target_hand_quat_w,
            steps=args_cli.goal_transport_steps,
            label="transport_to_goal",
        )
    return used_curobo


def _step_hold(env, video_state: dict[str, Any] | None, arm_open_action: float, steps: int, label: str) -> None:
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
    action[:, 7] = float(arm_open_action)
    for step in range(int(steps)):
        env.step(action)
        _capture_video_frames(
            env,
            video_state,
            stage=label,
            stage_step=step + 1,
            stage_steps=int(steps),
        )
        if step == 0 or step + 1 == steps:
            print(f"[MOTION] {label} step={step + 1}/{steps}", flush=True)


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


def _apply_lift_eval_initial_states(base_env) -> None:
    if not _LIFT_EVAL_FAILURE_CASES:
        return
    if len(_LIFT_EVAL_FAILURE_CASES) != int(base_env.num_envs):
        raise RuntimeError(
            f"lift-eval replay cases ({len(_LIFT_EVAL_FAILURE_CASES)}) must match num_envs ({base_env.num_envs})"
        )

    asset = base_env.scene["object"]
    env_ids = torch.arange(int(base_env.num_envs), dtype=torch.long, device=base_env.device)

    # Scale is part of the replay spawn manifest and must already be correct.
    # Clear the lazy cache so this check reads the spawned prim instead of a
    # value left by an earlier observation or reset command.
    base_env._scale_cache = {}
    actual_scales = mdp.get_rigid_body_scale(base_env, SceneEntityCfg("object"), env_ids)
    expected_scales = torch.tensor(
        [
            [float(case["replay_spawn_scale"])] * 3
            for case in _LIFT_EVAL_FAILURE_CASES
        ],
        dtype=torch.float32,
        device=base_env.device,
    )
    if not torch.allclose(actual_scales, expected_scales, rtol=0.0, atol=1e-6):
        mismatches = []
        for env_id in range(int(base_env.num_envs)):
            if not torch.allclose(actual_scales[env_id], expected_scales[env_id], rtol=0.0, atol=1e-6):
                mismatches.append(
                    f"env={env_id} expected={expected_scales[env_id].tolist()} "
                    f"actual={actual_scales[env_id].tolist()}"
                )
        raise RuntimeError(
            "Replay object scales do not match the JSONL scales applied at spawn: "
            + "; ".join(mismatches[:8])
        )
    print(
        f"[LIFT_EVAL_REPLAY] verified {len(_LIFT_EVAL_FAILURE_CASES)} spawned object scales match JSONL",
        flush=True,
    )

    root_state = torch.zeros((int(base_env.num_envs), 13), dtype=torch.float32, device=base_env.device)
    robot_root_pos_w = base_env.scene["robot"].data.root_pos_w[:, :3].detach().cpu().numpy()
    for env_id, case in enumerate(_LIFT_EVAL_FAILURE_CASES):
        object_pos_w = np.asarray(case["initial_object_pos_w"], dtype=np.float32)
        logged_robot_pos_w = case.get("initial_robot_pos_w")
        if logged_robot_pos_w is not None:
            delta = np.asarray(robot_root_pos_w[env_id], dtype=np.float32) - np.asarray(logged_robot_pos_w, dtype=np.float32)
            object_pos_w = object_pos_w + delta
            case["replay_world_translation_delta"] = delta.astype(np.float32).tolist()
        else:
            case["replay_world_translation_delta"] = [0.0, 0.0, 0.0]
        root_state[env_id, 0:3] = torch.tensor(object_pos_w, dtype=torch.float32, device=base_env.device)
        case["replay_object_pos_w"] = object_pos_w.astype(np.float32).tolist()
        root_state[env_id, 3:7] = torch.tensor(
            case["initial_object_quat_wxyz"], dtype=torch.float32, device=base_env.device
        )
    asset.write_root_state_to_sim(root_state, env_ids)
    if hasattr(base_env.scene, "write_data_to_sim"):
        base_env.scene.write_data_to_sim()
    print(
        f"[LIFT_EVAL_REPLAY] restored {len(_LIFT_EVAL_FAILURE_CASES)} logged object initial states",
        flush=True,
    )


def _lift_eval_candidate_record(
    grasp: dict[str, Any],
    fallback_rank: int,
    translation_delta: np.ndarray | None = None,
) -> dict[str, Any]:
    matrix = np.asarray(grasp.get("grasp_matrix_w"), dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"lift-eval failed grasp has invalid grasp_matrix_w shape: {matrix.shape}")
    matrix = matrix.copy()
    if translation_delta is not None:
        matrix[:3, 3] += np.asarray(translation_delta, dtype=np.float32).reshape(3)
    tier = str(grasp.get("selection_tier", ""))
    table_safe = bool(grasp.get("table_collision_safe", tier == "full_safe"))
    hand_safe = bool(grasp.get("table_hand_collision_safe", tier in {"full_safe", "hand_safe"}))
    return {
        "rank": int(grasp.get("rank", fallback_rank)),
        "candidate_index": int(grasp.get("candidate_index", fallback_rank)),
        "confidence": float(grasp.get("confidence", float("nan"))),
        "selection_tier": tier,
        "table_collision_safe": table_safe,
        "table_hand_collision_safe": hand_safe,
        "table_clearance_m": float(grasp.get("table_clearance_m", float("nan"))),
        "table_hand_clearance_m": float(grasp.get("table_hand_clearance_m", float("nan"))),
        "upward_alignment_score": _upward_alignment_score(matrix),
        "grasp_matrix_w": matrix.tolist(),
    }


def _infer_grasps_from_lift_eval_cases(
    base_env,
    replay_grasp_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if not _LIFT_EVAL_FAILURE_CASES:
        raise RuntimeError("_infer_grasps_from_lift_eval_cases called without replay cases")
    axis_idx, axis_sign = _axis_index(args_cli.approach_axis)
    grasp_pos_w: list[np.ndarray] = []
    grasp_quat_w: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for env_id, case in enumerate(_LIFT_EVAL_FAILURE_CASES):
        failed_grasps = case["failed_grasps"]
        requested_index = int(args_cli.grasp_index if replay_grasp_index is None else replay_grasp_index)
        if requested_index >= len(failed_grasps):
            fallback_pos, fallback_quat = _fallback_grasp_from_current_hand(base_env, env_id)
            grasp = _pose_matrix_from_pos_quat(fallback_pos, fallback_quat)
            grasp_pos_w.append(fallback_pos)
            grasp_quat_w.append(fallback_quat)
            records.append(
                {
                    "env_id": int(env_id),
                    "status": "lift_eval_failure_replay_no_grasp_at_index",
                    "execute_grasp": False,
                    "source_path": case["source_path"],
                    "source_row_index": case["source_row_index"],
                    "worker_id": case.get("worker_id"),
                    "episode_index": case.get("episode_index"),
                    "sample_index": case.get("sample_index"),
                    "source_env_id": case.get("env_id"),
                    "object_index": case.get("object_index"),
                    "object_name": case.get("object_name"),
                    "replay_grasp_list_index": int(requested_index),
                    "available_failed_grasps": int(len(failed_grasps)),
                    "chosen_index": None,
                    "chosen_rank": None,
                    "confidence": None,
                    "grasp_pose_frame": args_cli.grasp_pose_frame,
                    "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
                    "grasp_matrix_w": grasp.tolist(),
                    "grasp_quat_wxyz": fallback_quat.tolist(),
                    "approach_dir_w": [0.0, 0.0, 0.0],
                    "num_returned": len(failed_grasps),
                }
            )
            continue
        chosen_rank = max(requested_index, 0)
        chosen = failed_grasps[chosen_rank]
        translation_delta = np.asarray(
            case.get("replay_world_translation_delta", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        grasp = np.asarray(chosen.get("grasp_matrix_w"), dtype=np.float32)
        if grasp.shape != (4, 4):
            raise ValueError(
                f"lift-eval replay case {env_id} chosen grasp has invalid grasp_matrix_w shape: {grasp.shape}"
            )
        grasp = grasp.copy()
        grasp[:3, 3] += translation_delta
        quat = _matrix_to_quat_wxyz(grasp[:3, :3])
        approach_dir = axis_sign * grasp[:3, axis_idx]
        candidate_records = [
            _lift_eval_candidate_record(item, idx, translation_delta)
            for idx, item in enumerate(failed_grasps)
        ]
        chosen_record = _lift_eval_candidate_record(chosen, chosen_rank, translation_delta)

        grasp_pos_w.append(grasp[:3, 3])
        grasp_quat_w.append(quat)
        record = {
            "env_id": int(env_id),
            "status": "lift_eval_failure_replay",
            "execute_grasp": True,
            "source_path": case["source_path"],
            "source_row_index": case["source_row_index"],
            "worker_id": case.get("worker_id"),
            "episode_index": case.get("episode_index"),
            "sample_index": case.get("sample_index"),
            "source_env_id": case.get("env_id"),
            "object_index": case.get("object_index"),
            "object_name": case.get("object_name"),
            "replay_world_translation_delta": translation_delta.astype(np.float32).tolist(),
            "replay_object_pos_w": case.get("replay_object_pos_w"),
            "chosen_index": int(chosen_record["candidate_index"]),
            "chosen_rank": int(chosen_record["rank"]),
            "replay_grasp_list_index": int(chosen_rank),
            "available_failed_grasps": int(len(failed_grasps)),
            "confidence": chosen_record["confidence"],
            "remove_outliers": None,
            "grasp_pose_frame": args_cli.grasp_pose_frame,
            "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
            "grasp_matrix_w": grasp.tolist(),
            "grasp_quat_wxyz": quat.tolist(),
            "approach_dir_w": approach_dir.astype(np.float32).tolist(),
            "pointcloud_mean_w": None,
            "pointcloud_source": "lift_eval_failure_jsonl",
            "pointcloud_raw_points": None,
            "pointcloud_finite_points": None,
            "pointcloud_sent_points": None,
            "num_returned": len(failed_grasps),
            "selection_tier": chosen.get("selection_tier"),
            "table_safe_candidates": sum(1 for item in candidate_records if bool(item["table_collision_safe"])),
            "table_full_safe_candidates": sum(1 for item in candidate_records if bool(item["table_collision_safe"])),
            "table_hand_safe_candidates": sum(1 for item in candidate_records if bool(item["table_hand_collision_safe"])),
            "table_unsafe_candidates": sum(1 for item in candidate_records if not bool(item["table_collision_safe"])),
            "table_collision_safe": chosen_record["table_collision_safe"],
            "table_hand_collision_safe": chosen_record["table_hand_collision_safe"],
            "table_clearance_m": chosen_record["table_clearance_m"],
            "table_hand_clearance_m": chosen_record["table_hand_clearance_m"],
            "upward_alignment_score": chosen_record["upward_alignment_score"],
        }
        if args_cli.visualize_candidate_grasps:
            record["candidate_grasps"] = candidate_records
        records.append(record)
        print(
            f"[LIFT_EVAL_REPLAY] env={env_id} object={case.get('object_name')} "
            f"failed_grasps={len(failed_grasps)} replay_index={chosen_rank} "
            f"rank={chosen_record['rank']} confidence={chosen_record['confidence']:.4f}",
            flush=True,
        )

    pos = torch.tensor(np.asarray(grasp_pos_w), dtype=torch.float32, device=base_env.device)
    quat = torch.tensor(np.asarray(grasp_quat_w), dtype=torch.float32, device=base_env.device)
    return pos, quat, records


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


def _annotate_records_with_achieved_grasp_pose(
    base_env,
    records: list[dict[str, Any]],
    grasp_target_pos_w: torch.Tensor,
    grasp_target_quat_w: torch.Tensor,
) -> None:
    hand_pos_w, hand_quat_w = _current_hand_pose_w(base_env)
    pos_error, rot_error = compute_pose_error(
        hand_pos_w,
        hand_quat_w,
        grasp_target_pos_w,
        grasp_target_quat_w,
        rot_error_type="axis_angle",
    )
    pos_error_m = torch.linalg.norm(pos_error, dim=-1)
    rot_error_rad = torch.linalg.norm(rot_error, dim=-1)

    hand_pos = hand_pos_w.detach().cpu().numpy()
    hand_quat = hand_quat_w.detach().cpu().numpy()
    pos_error_values = pos_error_m.detach().cpu().numpy()
    rot_error_values = rot_error_rad.detach().cpu().numpy()
    for env_id, record in enumerate(records):
        record["achieved_grasp_hand_pos_w"] = hand_pos[env_id].astype(np.float32).tolist()
        record["achieved_grasp_hand_quat_wxyz"] = hand_quat[env_id].astype(np.float32).tolist()
        record["grasp_target_position_error_m"] = float(pos_error_values[env_id])
        record["grasp_target_rotation_error_rad"] = float(rot_error_values[env_id])
        if pos_error_values[env_id] > 0.01 or rot_error_values[env_id] > 0.10:
            print(
                f"[WARNING] grasp target not reached env={env_id} "
                f"pos_error={pos_error_values[env_id]:.4f}m "
                f"rot_error={rot_error_values[env_id]:.4f}rad",
                flush=True,
            )

    print(
        "[MOTION] achieved grasp pose "
        f"max_pos_error={float(pos_error_m.max().detach().cpu()):.4f}m "
        f"max_rot_error={float(rot_error_rad.max().detach().cpu()):.4f}rad",
        flush=True,
    )


def _write_debug_json(base_env, output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = base_env.command_manager.get_command("target_object_pose")
    try:
        object_scale_xyz = _jsonable_tensor(
            mdp.get_rigid_body_scale(base_env, SceneEntityCfg("object"), list(range(int(base_env.num_envs))))
        )
    except Exception as exc:
        object_scale_xyz = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "runtime_spec": runtime_spec_path,
        "runtime_paths_yaml": runtime_spec.get("paths_yaml"),
        "lift_eval_replay_manifest": str(_LIFT_EVAL_REPLAY_MANIFEST) if _LIFT_EVAL_REPLAY_MANIFEST is not None else None,
        "grasp_candidates_jsonl": args_cli.grasp_candidates_jsonl,
        "task": args_cli.task,
        "num_envs": int(base_env.num_envs),
        "grasp_pose_frame": args_cli.grasp_pose_frame,
        "grasp_to_hand_rotation": args_cli.grasp_to_hand_rotation,
        "grasp_to_hand_rot_matrix": _grasp_to_hand_rot_matrix_np().tolist(),
        "panda_hand_to_tcp_z": float(args_cli.panda_hand_to_tcp_z),
        "ik_method": args_cli.ik_method,
        "ik_damping": float(args_cli.ik_damping),
        "panda_finger_static_friction": (
            None if _PANDA_FINGER_FRICTION is None else float(_PANDA_FINGER_FRICTION[0])
        ),
        "panda_finger_dynamic_friction": (
            None if _PANDA_FINGER_FRICTION is None else float(_PANDA_FINGER_FRICTION[1])
        ),
        "table_collision_filter_enabled": not bool(args_cli.no_table_collision_filter),
        "table_collision_clearance": float(args_cli.table_collision_clearance),
        "table_collision_xy_margin": float(args_cli.table_collision_xy_margin),
        "unsafe_grasp_fallback": args_cli.unsafe_grasp_fallback,
        "gripper_collision_proxy_source": getattr(base_env, "_graspgen_table_collision_proxy_source", None),
        "records": records,
        "object_root_pos_w": _jsonable_tensor(base_env.scene["object"].data.root_pos_w[:, :3]),
        "object_root_quat_wxyz": _jsonable_tensor(base_env.scene["object"].data.root_quat_w),
        "object_scale_xyz": object_scale_xyz,
        "loaded_object_assets": _object_asset_debug_rows(int(base_env.num_envs)),
        "target_object_pose_env": _jsonable_tensor(command),
        "robot_root_pos_w": _jsonable_tensor(base_env.scene["robot"].data.root_pos_w[:, :3]),
        "robot_root_quat_wxyz": _jsonable_tensor(base_env.scene["robot"].data.root_quat_w),
    }
    path = output_dir / "graspgen_direct_grasp_debug.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote debug metadata: {path}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    print(
        f"[STARTUP] entered main task={args_cli.task} "
        f"simulation_app_running={bool(simulation_app.is_running())}",
        flush=True,
    )
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

    print(
        f"[STARTUP] before gym.make task={args_cli.task} num_envs={args_cli.num_envs} "
        f"record_video={bool(args_cli.record_video)}",
        flush=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print(
        f"[STARTUP] after gym.make simulation_app_running={bool(simulation_app.is_running())}",
        flush=True,
    )
    output_dir = Path(args_cli.video_dir).expanduser().resolve()
    video_state = None

    print(f"[INFO] runtime spec: {runtime_spec_path}", flush=True)
    print(f"[INFO] reset task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    try:
        env.reset()
        base_env = env.unwrapped
        print("[STARTUP] initial env.reset completed", flush=True)
        _override_panda_finger_friction(base_env)
        print("[STARTUP] Panda finger material setup completed", flush=True)
        _validate_lift_eval_replay_object_assets(int(base_env.num_envs))

        arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        arm_cfg.resolve(base_env.scene)
        finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
        finger_cfg.resolve(base_env.scene)
        if len(arm_cfg.joint_ids) != 7:
            raise RuntimeError(f"Expected 7 Panda arm joints, got {arm_cfg.joint_names!r}")
        if len(finger_cfg.joint_ids) != 2:
            raise RuntimeError(f"Expected 2 Panda finger joints, got {finger_cfg.joint_names!r}")

        all_records: list[dict[str, Any]] = []

        def _close_active_video_state() -> None:
            nonlocal video_state
            state = video_state
            video_state = None
            _close_video_state(state)

        def _run_one_grasp_batch(replay_grasp_index: int | None = None) -> bool:
            nonlocal video_state
            label_prefix = "" if replay_grasp_index is None else f"replay_grasp_{replay_grasp_index:03d}_"
            if _LIFT_EVAL_FAILURE_CASES:
                video_grasp_index = (
                    max(0, int(args_cli.grasp_index))
                    if replay_grasp_index is None
                    else int(replay_grasp_index)
                )
                video_run_label = f"replay_grasp_{video_grasp_index:03d}"
            else:
                video_run_label = None

            if video_state is not None:
                raise RuntimeError("Video state from the previous grasp batch was not closed")
            if args_cli.record_warmup:
                video_state = _init_video_state(
                    output_dir,
                    base_env,
                    run_label=video_run_label,
                )
                _capture_video_frames(env, video_state, stage="initial")

            _step_hold(env, video_state, 1.0, args_cli.settle_steps, f"{label_prefix}settle_open")
            _step_hold(env, video_state, 1.0, args_cli.open_steps, f"{label_prefix}open")
            if _LIFT_EVAL_FAILURE_CASES:
                # Lift-eval rows store initial_state after the same settle/open phase.
                # Restore it here so replay grasps keep the logged object-relative pose.
                _apply_lift_eval_initial_states(base_env)
                if int(args_cli.restore_settle_steps) > 0:
                    _step_hold(
                        env,
                        video_state,
                        1.0,
                        int(args_cli.restore_settle_steps),
                        f"{label_prefix}restore_open",
                    )
                if video_state is not None:
                    _capture_video_frames(env, video_state, stage=f"{label_prefix}restore_initial_state")

            if _LIFT_EVAL_FAILURE_CASES:
                grasp_pose_pos_w, grasp_pose_quat_w, records = _infer_grasps_from_lift_eval_cases(
                    base_env,
                    replay_grasp_index,
                )
            else:
                grasp_pose_pos_w, grasp_pose_quat_w, records = _infer_grasps(base_env)
            for record in records:
                if replay_grasp_index is not None:
                    record["replay_sequence_index"] = int(replay_grasp_index)
                all_records.append(record)

            approach_dirs = []
            for record in records:
                approach_dirs.append(np.asarray(record["approach_dir_w"], dtype=np.float32))
            approach_dir_w = torch.tensor(np.asarray(approach_dirs), dtype=torch.float32, device=base_env.device)
            approach_dir_w = approach_dir_w / torch.clamp(
                torch.linalg.norm(approach_dir_w, dim=-1, keepdim=True),
                min=1e-8,
            )
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
            if video_state is not None:
                video_state["grasp_target_pos_w"] = grasp_target_pos_w.detach().cpu().numpy()
            _write_debug_json(base_env, output_dir, all_records if _LIFT_EVAL_FAILURE_CASES else records)
            if args_cli.dry_run_graspgen:
                _close_active_video_state()
                return False
            _visualize_grasp_guides(base_env, records)
            if video_state is None:
                video_state = _init_video_state(
                    output_dir,
                    base_env,
                    run_label=video_run_label,
                )
                if video_state is not None:
                    video_state["grasp_target_pos_w"] = grasp_target_pos_w.detach().cpu().numpy()
                _capture_video_frames(env, video_state, stage="grasp_ready")

            used_curobo_pregrasp = False
            if args_cli.use_curobo_pregrasp:
                used_curobo_pregrasp = _step_curobo_pregrasp(
                    env,
                    arm_cfg,
                    video_state,
                    1.0,
                    pregrasp_target_pos_w,
                    pregrasp_target_quat_w,
                    label=f"{label_prefix}approach_pregrasp",
                )
            if not used_curobo_pregrasp:
                _step_arm_to_pose(
                    env,
                    arm_cfg,
                    video_state,
                    1.0,
                    pregrasp_target_pos_w,
                    pregrasp_target_quat_w,
                    steps=args_cli.approach_steps,
                    label=f"{label_prefix}approach_pregrasp",
                )
            _step_arm_to_pose(
                env,
                arm_cfg,
                video_state,
                1.0,
                grasp_target_pos_w,
                grasp_target_quat_w,
                steps=args_cli.grasp_steps,
                label=f"{label_prefix}move_to_grasp",
            )
            _annotate_records_with_achieved_grasp_pose(
                base_env,
                records,
                grasp_target_pos_w,
                grasp_target_quat_w,
            )
            _write_debug_json(base_env, output_dir, all_records if _LIFT_EVAL_FAILURE_CASES else records)
            _step_arm_to_pose(
                env,
                arm_cfg,
                video_state,
                1.0,
                grasp_target_pos_w,
                grasp_target_quat_w,
                steps=args_cli.close_steps,
                label=f"{label_prefix}close",
                finger_action_end=-1.0,
            )
            _step_arm_to_pose(
                env,
                arm_cfg,
                video_state,
                -1.0,
                grasp_target_pos_w,
                grasp_target_quat_w,
                steps=args_cli.post_close_hold_steps,
                label=f"{label_prefix}post_close_hold",
            )
            if args_cli.use_curobo_goal_transport:
                _step_curobo_goal_transport(env, arm_cfg, video_state)
            else:
                _step_arm_to_pose(
                    env,
                    arm_cfg,
                    video_state,
                    -1.0,
                    lift_target_pos_w,
                    lift_target_quat_w,
                    steps=args_cli.lift_steps,
                    label=f"{label_prefix}lift",
                )
            _step_hold(env, video_state, -1.0, args_cli.hold_steps, f"{label_prefix}hold_closed")
            _close_active_video_state()
            return True

        if _LIFT_EVAL_FAILURE_CASES and not bool(args_cli.single_lift_eval_grasp):
            max_failed_grasps = max(len(case["failed_grasps"]) for case in _LIFT_EVAL_FAILURE_CASES)
            if int(args_cli.max_lift_eval_grasps_per_episode) > 0:
                max_failed_grasps = min(max_failed_grasps, int(args_cli.max_lift_eval_grasps_per_episode))
            print(
                f"[LIFT_EVAL_REPLAY] replaying all failed grasps: videos={len(_LIFT_EVAL_FAILURE_CASES)} "
                f"grasp_rounds={max_failed_grasps}",
                flush=True,
            )
            for replay_grasp_index in range(max_failed_grasps):
                if replay_grasp_index > 0:
                    env.reset()
                    base_env = env.unwrapped
                    _override_panda_finger_friction(base_env)
                print(
                    f"[LIFT_EVAL_REPLAY] grasp_round={replay_grasp_index + 1}/{max_failed_grasps}",
                    flush=True,
                )
                if not _run_one_grasp_batch(replay_grasp_index):
                    return
        else:
            if not _run_one_grasp_batch(None):
                return
    finally:
        close_error = None
        try:
            _close_video_state(video_state)
        except Exception as exc:
            close_error = exc
        env.close()
        simulation_app.close()
        if close_error is not None:
            raise close_error


if __name__ == "__main__":
    print("[STARTUP] entering Hydra task wrapper", flush=True)
    try:
        main()
    except SystemExit as exc:
        print(f"[STARTUP] SystemExit code={exc.code!r}", flush=True)
        raise
    else:
        print("[STARTUP] Hydra task wrapper returned normally", flush=True)

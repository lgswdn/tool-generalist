#!/usr/bin/env python3
"""Export OBJ scenes for visually checking GraspGen grasp quality.

Each exported OBJ contains:
- a table cuboid,
- one object mesh at a randomized initial pose,
- several GraspGen grasp guide meshes.

This is intentionally offline and does not launch Isaac. It mirrors the key
GraspGen conventions used by scripts/eval_graspgen_direct_grasp.py: object mesh
sampling, world-frame point clouds, adding the point-cloud mean back to returned
grasp translations, and the franka_panda grasp guide axes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GRASPGEN_ROOT = "/mnt/project/world_model/tool_generalist/GraspGen"
DEFAULT_GRASPGEN_PORT = 5556


@dataclass(frozen=True)
class ObjectAsset:
    name: str
    obj_path: Path
    manifest_scale: float | None = None


@dataclass
class SimpleMesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class CandidateRecord:
    rank: int
    candidate_index: int
    confidence: float
    matrix_w: np.ndarray
    full_safe: bool
    hand_safe: bool
    full_clearance: float
    hand_clearance: float
    upward_score: float
    selected: bool = False
    failed: bool = False


@dataclass
class LiftEvalFailureCase:
    source_path: str
    source_kind: str
    row_index: int
    worker_id: int | None
    episode_index: int | None
    sample_index: int | None
    env_id: int | None
    object_index: int | None
    object_name: str | None
    initial_object_pos_w: np.ndarray
    initial_object_quat_wxyz: np.ndarray
    initial_object_scale_xyz: np.ndarray | None
    failed_grasps: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass
class ConfigDefaults:
    config: str | None = None
    paths_yaml: str | None = None
    objects_json: Path | None = None
    obj_dir: Path | None = None
    table_enabled: bool | None = None
    table_size: tuple[float, float, float] | None = None
    table_pose: tuple[float, float, float] | None = None
    initial_position_range: float | None = None
    scale_range: tuple[float, float] | None = None
    scale_randomization_enabled: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OBJ debug scenes with a table, randomized object poses, "
            "and GraspGen grasp guides."
        )
    )
    parser.add_argument("--paths_yaml", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Experiment config exposing EXP_CFG. When set, object manifest/OBJ dir, "
            "table parameters, object pose sampling, and object scale randomization "
            "are read from the config unless explicitly overridden."
        ),
    )
    parser.add_argument("--objects_json", type=str, default=None, help="Object manifest JSON. Defaults to paths_yaml dgn.candidates_json.")
    parser.add_argument("--obj_dir", type=str, default=None, help="Object OBJ directory. Defaults to paths_yaml dgn.obj_dir.")
    parser.add_argument(
        "--object_obj",
        action="append",
        default=[],
        help="Direct object OBJ path. May be passed multiple times; bypasses manifest loading when set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/visualization/graspgen_grasp_viz_objs",
    )
    parser.add_argument("--num_scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    parser.add_argument(
        "--no_copy_object_assets",
        action="store_true",
        help="Do not copy source object OBJ/MTL/texture files into the output directory.",
    )
    parser.add_argument(
        "--lift_eval_json",
        type=str,
        default=None,
        help=(
            "summary_rank_*.json written by scripts/eval_graspgen_lift_grasps.py. "
            "When set, export the failed grasp cases stored in failed_grasp_pose_rows."
        ),
    )
    parser.add_argument(
        "--lift_eval_failures_jsonl",
        type=str,
        default=None,
        help=(
            "failures_rank_*.jsonl written by scripts/eval_graspgen_lift_grasps.py. "
            "When set, export each failure row."
        ),
    )
    parser.add_argument(
        "--grasp_candidates_jsonl",
        type=str,
        default=None,
        help=(
            "Candidate JSONL written by scripts/generate_graspgen_candidates_from_pointclouds.py. "
            "This mode loads the recorded canonical PLY instead of resampling the object OBJ."
        ),
    )
    parser.add_argument(
        "--grasp_candidates_filter",
        choices=("full_safe", "full_and_hand_safe", "all"),
        default="full_safe",
        help="Candidate tiers exported from --grasp_candidates_jsonl.",
    )
    parser.add_argument(
        "--max_grasp_candidate_cases",
        type=int,
        default=0,
        help="Maximum generated-candidate rows to export. 0 exports every row.",
    )
    parser.add_argument(
        "--max_lift_eval_failures",
        type=int,
        default=0,
        help="Maximum lift-eval failure cases to export. 0 exports all failures.",
    )

    parser.add_argument("--graspgen_root", type=str, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--graspgen_host", type=str, default="localhost")
    parser.add_argument("--graspgen_port", type=int, default=DEFAULT_GRASPGEN_PORT)
    parser.add_argument("--graspgen_timeout_ms", type=int, default=120_000)
    parser.add_argument("--graspgen_point_cloud_points", type=int, default=2048)
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--topk_num_grasps", type=int, default=20)
    parser.add_argument("--grasp_threshold", type=float, default=-1.0)
    parser.add_argument("--min_grasps", type=int, default=1)
    parser.add_argument("--max_tries", type=int, default=6)
    parser.add_argument("--remove_outliers", action="store_true")
    parser.add_argument(
        "--dry_run_random_grasps",
        action="store_true",
        help="Do not call GraspGen; export deterministic fake grasps for testing the OBJ pipeline.",
    )

    parser.add_argument(
        "--object_scale",
        type=float,
        default=None,
        help="Uniform scale applied to object OBJ vertices before pose randomization. Defaults to 1.0 OBJ units.",
    )
    parser.add_argument(
        "--sample_config_object_scale",
        action="store_true",
        default=None,
        help="When --config has enabled object scale randomization, sample the extra scale factor from that config range.",
    )
    parser.add_argument(
        "--no_sample_config_object_scale",
        dest="sample_config_object_scale",
        action="store_false",
        help="Disable automatic sampling from config object scale randomization.",
    )
    parser.add_argument(
        "--use_manifest_scale",
        action="store_true",
        help="Multiply --object_scale by the numeric suffix in manifest entries.",
    )
    parser.add_argument(
        "--pose_sampling_mode",
        choices=("manual", "config_initial"),
        default=None,
        help=(
            "manual uses --object_xy_range centered at zero; config_initial mirrors "
            "reset_initial_object_position x=0.5+/-range, y=+/-2*range."
        ),
    )
    parser.add_argument("--object_xy_range", type=float, nargs=2, default=None)
    parser.add_argument("--object_z_clearance", type=float, default=0.002)
    parser.add_argument(
        "--roll_pitch_range_deg",
        type=float,
        default=0.0,
        help="Symmetric random roll/pitch range in degrees. Default keeps objects upright with random yaw.",
    )

    parser.add_argument("--table_size", type=float, nargs=3, default=None)
    parser.add_argument("--table_pose", type=float, nargs=3, default=None)
    parser.add_argument("--table_collision_clearance", type=float, default=0.005)
    parser.add_argument("--table_collision_xy_margin", type=float, default=0.0)
    parser.add_argument("--no_table_collision_filter", action="store_true")
    parser.add_argument("--table_collision_proxy_points", type=int, default=512)

    parser.add_argument("--num_vis_grasps", type=int, default=20)
    parser.add_argument("--grasp_index", type=int, default=0)
    parser.add_argument("--grasp_line_width", type=float, default=0.08)
    parser.add_argument("--grasp_line_depth", type=float, default=0.10)
    parser.add_argument("--grasp_line_thickness", type=float, default=0.006)
    parser.add_argument("--candidate_grasp_line_thickness", type=float, default=0.0025)
    parser.add_argument("--include_point_cloud", action="store_true")
    parser.add_argument("--object_cloud_vis_points", type=int, default=256)
    parser.add_argument("--object_cloud_point_size", type=float, default=0.004)
    parser.add_argument(
        "--grasp_pose_frame",
        choices=("base", "tcp"),
        default="base",
        help="Frame represented by GraspGen matrices; kept consistent with eval_graspgen_direct_grasp.py.",
    )
    parser.add_argument("--grasp_to_hand_rotation", choices=("franka_panda", "identity"), default="franka_panda")
    parser.add_argument("--panda_hand_to_tcp_z", type=float, default=0.107)
    return parser.parse_args()


def _as_tuple3(values: Any, label: str) -> tuple[float, float, float]:
    if values is None:
        raise ValueError(f"{label} is missing")
    seq = list(values)
    if len(seq) != 3:
        raise ValueError(f"{label} must contain 3 values, got {values!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _as_float_pair(values: Any, label: str) -> tuple[float, float]:
    seq = list(values)
    if len(seq) != 2:
        raise ValueError(f"{label} must contain 2 values, got {values!r}")
    return (float(seq[0]), float(seq[1]))


def _load_config_defaults(config: str | None) -> ConfigDefaults:
    if not config:
        return ConfigDefaults()
    from utils.config.loader import load_exp_cfg
    from utils.config.paths import load_project_paths
    from utils.experiment.effective_paths import apply_experiment_path_overrides

    cfg = load_exp_cfg(config)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    rl = cfg.rl
    dr_object_scale = rl.domain_randomization.object.scale
    scale_enabled = bool(rl.domain_randomization.enabled and dr_object_scale.enabled)
    return ConfigDefaults(
        config=str(config),
        paths_yaml=str(paths.source_yaml),
        objects_json=paths.get("objects.candidates_json"),
        obj_dir=paths.get("objects.obj_dir"),
        table_enabled=bool(rl.table.enabled),
        table_size=_as_tuple3(rl.table.size_xyz, "RLCfg.table.size_xyz"),
        table_pose=_as_tuple3(rl.table.pose_xyz, "RLCfg.table.pose_xyz"),
        initial_position_range=float(rl.object_pose_sampling.initial_position_range),
        scale_range=_as_float_pair(dr_object_scale.range, "DomainRandomizationCfg.object.scale.range"),
        scale_randomization_enabled=scale_enabled,
    )


def _finalize_args(args: argparse.Namespace) -> ConfigDefaults:
    defaults = _load_config_defaults(args.config)
    lift_eval_mode = bool(args.lift_eval_json or args.lift_eval_failures_jsonl or args.grasp_candidates_jsonl)
    if args.paths_yaml is None:
        args.paths_yaml = defaults.paths_yaml or "configs/paths/panda_hand.yaml"
    if args.objects_json is None and defaults.objects_json is not None:
        args.objects_json = str(defaults.objects_json)
    if args.obj_dir is None and defaults.obj_dir is not None:
        args.obj_dir = str(defaults.obj_dir)
    if args.table_size is None:
        args.table_size = list(defaults.table_size or (1.0, 1.0, 0.04))
    if args.table_pose is None:
        args.table_pose = list(defaults.table_pose or (0.0, 0.0, -0.02))
    if args.object_scale is None:
        args.object_scale = 1.0
    if args.sample_config_object_scale is None:
        args.sample_config_object_scale = False if lift_eval_mode else bool(defaults.scale_randomization_enabled)
    if args.pose_sampling_mode is None:
        args.pose_sampling_mode = "config_initial" if args.config else "manual"
    if args.object_xy_range is None:
        args.object_xy_range = [-0.18, 0.18]

    args.config_table_enabled = defaults.table_enabled
    args.config_initial_position_range = defaults.initial_position_range
    args.config_object_scale_randomization_enabled = defaults.scale_randomization_enabled
    args.config_object_scale_range = None if defaults.scale_range is None else list(defaults.scale_range)
    return defaults


def _load_paths_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"paths_yaml not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --paths_yaml; pass --objects_json and --obj_dir instead.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _parse_manifest_entry(item: str) -> tuple[str, float | None]:
    if "-" not in item:
        return item, None
    base, scale_str = item.rsplit("-", 1)
    try:
        return base, float(scale_str)
    except ValueError:
        return item, None


def _load_object_assets(args: argparse.Namespace) -> list[ObjectAsset]:
    direct_paths = [Path(p).expanduser().resolve() for p in args.object_obj]
    if direct_paths:
        assets = [ObjectAsset(path.stem, path) for path in direct_paths]
    else:
        objects_json = args.objects_json
        obj_dir = args.obj_dir
        if objects_json is None or obj_dir is None:
            paths = _load_paths_yaml(Path(args.paths_yaml).expanduser())
            dgn = paths.get("dgn")
            if not isinstance(dgn, dict):
                raise ValueError(f"{args.paths_yaml} does not contain a dgn mapping")
            objects_json = objects_json or dgn.get("candidates_json")
            obj_dir = obj_dir or dgn.get("obj_dir")
        if not objects_json or not obj_dir:
            raise ValueError("Could not resolve object manifest and OBJ directory")

        manifest_path = Path(str(objects_json)).expanduser()
        object_dir = Path(str(obj_dir)).expanduser()
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError(f"Expected {manifest_path} to be a JSON list of strings")
        assets = []
        seen: set[str] = set()
        for item in data:
            base, manifest_scale = _parse_manifest_entry(item)
            if base in seen:
                continue
            seen.add(base)
            assets.append(ObjectAsset(base, object_dir / f"{base}.obj", manifest_scale))

    existing = [asset for asset in assets if asset.obj_path.is_file()]
    missing = len(assets) - len(existing)
    if missing:
        print(f"[WARNING] skipped {missing} object OBJ paths that do not exist", flush=True)
    if not existing:
        raise RuntimeError("No usable object OBJ files found")
    return existing


def _add_graspgen_to_path(graspgen_root: str) -> None:
    root = Path(graspgen_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GraspGen root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


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
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_transform_from_pos_quat_scale(pos: np.ndarray, quat_wxyz: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale_arr.shape[0] == 1:
        scale_arr = np.repeat(scale_arr, 3)
    if scale_arr.shape[0] != 3:
        raise ValueError(f"scale must be scalar or 3-vector, got {scale!r}")
    transform[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz) @ np.diag(scale_arr.astype(np.float32))
    transform[:3, 3] = np.asarray(pos, dtype=np.float32).reshape(3)
    return transform


def _parse_obj_face_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    if not raw:
        raise ValueError(f"empty OBJ face index token: {token!r}")
    index = int(raw)
    if index < 0:
        return vertex_count + index
    return index - 1


def _load_obj_mesh(path: Path) -> SimpleMesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                polygon = [_parse_obj_face_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[i], polygon[i + 1]])
    if not vertices:
        raise RuntimeError(f"OBJ has no vertices: {path}")
    if not faces:
        raise RuntimeError(f"OBJ has no faces: {path}")
    return SimpleMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _load_mesh(path: Path) -> SimpleMesh:
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise RuntimeError(f"mesh has no vertices: {path}")
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            raise RuntimeError(f"mesh has no faces: {path}")
        return SimpleMesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int64),
        )
    except ModuleNotFoundError:
        return _load_obj_mesh(path)


def _sample_mesh_points(mesh: SimpleMesh, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"point count must be positive, got {count}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        indices = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        return vertices[indices].astype(np.float32, copy=False)

    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    finite_area = np.isfinite(areas) & (areas > 1e-16)
    if not np.any(finite_area):
        indices = rng.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        return vertices[indices].astype(np.float32, copy=False)

    valid_triangles = triangles[finite_area]
    valid_areas = areas[finite_area]
    probs = valid_areas / float(np.sum(valid_areas))
    triangle_indices = rng.choice(valid_triangles.shape[0], size=count, replace=True, p=probs)
    chosen = valid_triangles[triangle_indices]
    u = rng.random(count, dtype=np.float32)
    v = rng.random(count, dtype=np.float32)
    sqrt_u = np.sqrt(u)
    points = (
        (1.0 - sqrt_u)[:, None] * chosen[:, 0]
        + (sqrt_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (sqrt_u * v)[:, None] * chosen[:, 2]
    )
    return points.astype(np.float32, copy=False)


def _random_object_transform(
    mesh: SimpleMesh,
    scale: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    table_pose = np.asarray(args.table_pose, dtype=np.float32)
    table_size = np.asarray(args.table_size, dtype=np.float32)
    table_top_z = float(table_pose[2] + 0.5 * table_size[2])
    roll_pitch = math.radians(float(args.roll_pitch_range_deg))
    roll = float(rng.uniform(-roll_pitch, roll_pitch))
    pitch = float(rng.uniform(-roll_pitch, roll_pitch))
    yaw = float(rng.uniform(-math.pi, math.pi))
    rot = _rotation_matrix_from_rpy(roll, pitch, yaw)

    local_vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
    rotated_vertices = local_vertices @ rot.T
    if args.pose_sampling_mode == "config_initial":
        initial_range = (
            float(args.config_initial_position_range)
            if args.config_initial_position_range is not None
            else 0.15
        )
        x = 0.5 + float(rng.uniform(-initial_range, initial_range))
        y = float(rng.uniform(-2.0 * initial_range, 2.0 * initial_range))
    else:
        xy_low, xy_high = [float(v) for v in args.object_xy_range]
        x = float(rng.uniform(xy_low, xy_high))
        y = float(rng.uniform(xy_low, xy_high))
    z = table_top_z + float(args.object_z_clearance) - float(np.min(rotated_vertices[:, 2]))
    pos = np.asarray([x, y, z], dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot * float(scale)
    transform[:3, 3] = pos
    pose = {
        "position_w": pos.tolist(),
        "quat_wxyz": _matrix_to_quat_wxyz(rot).tolist(),
        "roll_pitch_yaw_rad": [roll, pitch, yaw],
        "scale": float(scale),
        "pose_sampling_mode": str(args.pose_sampling_mode),
        "config_initial_position_range": args.config_initial_position_range,
    }
    return transform, pose


def _sample_object_scale(asset: ObjectAsset, args: argparse.Namespace, rng: np.random.Generator) -> tuple[float, dict[str, Any]]:
    scale = float(args.object_scale)
    components: dict[str, Any] = {
        "base_object_scale": float(args.object_scale),
        "manifest_scale": None,
        "config_random_scale_factor": None,
        "config_random_scale_range": args.config_object_scale_range,
        "config_random_scale_enabled": bool(args.config_object_scale_randomization_enabled),
        "sample_config_object_scale": bool(args.sample_config_object_scale),
    }
    if args.use_manifest_scale and asset.manifest_scale is not None:
        scale *= float(asset.manifest_scale)
        components["manifest_scale"] = float(asset.manifest_scale)

    if (
        args.sample_config_object_scale
        and args.config_object_scale_randomization_enabled
        and args.config_object_scale_range is not None
    ):
        low, high = [float(v) for v in args.config_object_scale_range]
        factor = float(rng.uniform(low, high))
        scale *= factor
        components["config_random_scale_factor"] = factor

    components["final_mesh_scale"] = float(scale)
    return scale, components


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _points_bounds(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] == 0:
        return {"count": 0, "min": None, "max": None, "extent": None}
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    return {
        "count": int(finite.shape[0]),
        "min": lower.astype(float).tolist(),
        "max": upper.astype(float).tolist(),
        "extent": (upper - lower).astype(float).tolist(),
    }


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


def _fallback_panda_gripper_proxy_points_h(args: argparse.Namespace) -> dict[str, np.ndarray]:
    width = 0.10537486
    depth = 0.10527314
    palm = _box_corner_points((0.0, 0.0, 0.005), (0.120, 0.075, 0.050))
    left_finger = _box_corner_points((0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    right_finger = _box_corner_points((-0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    all_points = np.concatenate([palm, left_finger, right_finger], axis=0).astype(np.float32)
    max_points = max(8, int(args.table_collision_proxy_points))
    if all_points.shape[0] > max_points:
        indices = np.linspace(0, all_points.shape[0] - 1, max_points, dtype=np.int64)
        all_points = all_points[indices]
    return {"all": all_points, "hand": palm.astype(np.float32)}


def _grasp_to_hand_rot_matrix_np(args: argparse.Namespace) -> np.ndarray:
    if args.grasp_to_hand_rotation == "identity":
        return np.eye(3, dtype=np.float32)
    return np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _candidate_hand_matrix_w(grasp_matrix_w: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    matrix = np.asarray(grasp_matrix_w, dtype=np.float32).copy()
    matrix[:3, :3] = matrix[:3, :3] @ _grasp_to_hand_rot_matrix_np(args)
    if args.grasp_pose_frame == "tcp":
        matrix[:3, 3] -= matrix[:3, :3] @ np.asarray([0.0, 0.0, float(args.panda_hand_to_tcp_z)], dtype=np.float32)
    return matrix


def _table_info(args: argparse.Namespace) -> tuple[float, np.ndarray]:
    table_pose = np.asarray(args.table_pose, dtype=np.float32)
    table_size = np.asarray(args.table_size, dtype=np.float32)
    top_z = float(table_pose[2] + 0.5 * table_size[2])
    half_xy = 0.5 * table_size[:2] + float(args.table_collision_xy_margin)
    center_xy = table_pose[:2]
    bounds_xy = np.stack((center_xy - half_xy, center_xy + half_xy), axis=0)
    return top_z, bounds_xy.astype(np.float32)


def _gripper_table_clearance(
    grasp_matrix_w: np.ndarray,
    proxy_points_h: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[bool, float]:
    if args.no_table_collision_filter or proxy_points_h is None:
        return True, float("inf")
    table_top_z, bounds_xy = _table_info(args)
    hand_matrix_w = _candidate_hand_matrix_w(grasp_matrix_w, args)
    points_w = proxy_points_h @ hand_matrix_w[:3, :3].T + hand_matrix_w[:3, 3]
    low_xy, high_xy = bounds_xy
    mask = (
        np.isfinite(points_w).all(axis=1)
        & (points_w[:, 0] >= low_xy[0])
        & (points_w[:, 0] <= high_xy[0])
        & (points_w[:, 1] >= low_xy[1])
        & (points_w[:, 1] <= high_xy[1])
    )
    if not np.any(mask):
        return True, float("inf")
    min_clearance = float(np.min(points_w[mask, 2] - table_top_z))
    return min_clearance >= float(args.table_collision_clearance), min_clearance


def _upward_alignment_score(grasp_matrix_w: np.ndarray, args: argparse.Namespace) -> float:
    hand_matrix_w = _candidate_hand_matrix_w(grasp_matrix_w, args)
    z_axis = hand_matrix_w[:3, 2]
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-8 or not np.isfinite(norm):
        return float("-inf")
    return float(z_axis[2] / norm)


def _infer_grasps(client, point_cloud: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    remove_outliers_options = [bool(args.remove_outliers)]
    if bool(args.remove_outliers):
        remove_outliers_options.append(False)
    last_error: str | None = None
    for remove_outliers in remove_outliers_options:
        for attempt in range(int(args.max_tries)):
            try:
                grasps, confidences = client.infer(
                    point_cloud,
                    grasp_threshold=float(args.grasp_threshold),
                    num_grasps=int(args.num_grasps),
                    topk_num_grasps=int(args.topk_num_grasps),
                    min_grasps=int(args.min_grasps),
                    remove_outliers=remove_outliers,
                )
                grasps = np.asarray(grasps, dtype=np.float32)
                confidences = np.asarray(confidences, dtype=np.float32)
                if len(grasps) == 0:
                    raise RuntimeError("GraspGen returned no grasps")
                return grasps, confidences, remove_outliers, None
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[WARNING] GraspGen failed attempt={attempt + 1}/{args.max_tries} "
                    f"remove_outliers={remove_outliers}: {exc}",
                    flush=True,
                )
                time.sleep(0.25)
    return np.empty((0, 4, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), bool(args.remove_outliers), last_error


def _fake_grasps(point_cloud_w: np.ndarray, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(point_cloud_w, axis=0).astype(np.float32)
    grasps = []
    confidences = []
    for i in range(count):
        yaw = 2.0 * math.pi * i / max(count, 1)
        rot = _rotation_matrix_from_rpy(0.0, math.radians(80.0), yaw)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rot
        matrix[:3, 3] = center + np.asarray([0.04 * math.cos(yaw), 0.04 * math.sin(yaw), 0.08], dtype=np.float32)
        grasps.append(matrix)
        confidences.append(float(1.0 - i / max(count, 1)))
    return np.stack(grasps, axis=0), np.asarray(confidences, dtype=np.float32)


def _rank_candidates(
    grasps: np.ndarray,
    confidences: np.ndarray,
    pc_mean_w: np.ndarray,
    args: argparse.Namespace,
) -> list[CandidateRecord]:
    proxy = None if args.no_table_collision_filter else _fallback_panda_gripper_proxy_points_h(args)
    records: list[CandidateRecord] = []
    order = np.argsort(-np.asarray(confidences))
    for ranked_i, candidate_i in enumerate(order):
        grasp = np.asarray(grasps[int(candidate_i)], dtype=np.float32).copy()
        if not args.dry_run_random_grasps:
            grasp[:3, 3] += pc_mean_w
        full_safe, full_clearance = _gripper_table_clearance(
            grasp,
            None if proxy is None else proxy["all"],
            args,
        )
        hand_safe, hand_clearance = _gripper_table_clearance(
            grasp,
            None if proxy is None else proxy["hand"],
            args,
        )
        records.append(
            CandidateRecord(
                rank=int(ranked_i),
                candidate_index=int(candidate_i),
                confidence=float(confidences[int(candidate_i)]),
                matrix_w=grasp,
                full_safe=bool(full_safe),
                hand_safe=bool(hand_safe),
                full_clearance=float(full_clearance),
                hand_clearance=float(hand_clearance),
                upward_score=_upward_alignment_score(grasp, args),
            )
        )

    full_safe = [item for item in records if item.full_safe]
    hand_safe = [item for item in records if item.hand_safe and not item.full_safe]
    unsafe = [item for item in records if not item.hand_safe and not item.full_safe]
    chosen: CandidateRecord | None
    if full_safe:
        chosen = full_safe[min(max(int(args.grasp_index), 0), len(full_safe) - 1)]
    elif hand_safe:
        chosen = hand_safe[min(max(int(args.grasp_index), 0), len(hand_safe) - 1)]
    elif unsafe:
        chosen = max(unsafe, key=lambda item: item.upward_score)
    else:
        chosen = None
    if chosen is not None:
        chosen.selected = True
    return records


def _make_box_vertices_faces(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    vertices = corners @ transform[:3, :3].T + transform[:3, 3]
    return vertices, faces


def _table_mesh(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.diag(np.asarray(args.table_size, dtype=np.float32))
    transform[:3, 3] = np.asarray(args.table_pose, dtype=np.float32)
    return _make_box_vertices_faces(transform)


def _grasp_guide_meshes(
    transform: np.ndarray,
    *,
    half_width: float,
    depth: float,
    thickness: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    transform = np.asarray(transform, dtype=np.float32)
    origin = transform[:3, 3]
    x_axis = transform[:3, 0]
    y_axis = transform[:3, 1]
    z_axis = transform[:3, 2]
    z0 = -0.25 * depth
    z1 = 0.75 * depth
    zc = 0.5 * (z0 + z1)
    meshes = []
    for side in (-1.0, 1.0):
        center = origin + side * half_width * x_axis + zc * z_axis
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 0] = x_axis * thickness
        matrix[:3, 1] = y_axis * thickness
        matrix[:3, 2] = z_axis * (z1 - z0)
        matrix[:3, 3] = center
        meshes.append(_make_box_vertices_faces(matrix))

    connector_center = origin + z0 * z_axis
    connector_matrix = np.eye(4, dtype=np.float32)
    connector_matrix[:3, 0] = x_axis * (2.0 * half_width + thickness)
    connector_matrix[:3, 1] = y_axis * thickness
    connector_matrix[:3, 2] = z_axis * thickness
    connector_matrix[:3, 3] = connector_center
    meshes.append(_make_box_vertices_faces(connector_matrix))
    return meshes


def _point_cloud_meshes(points_w: np.ndarray, args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    if not args.include_point_cloud:
        return []
    max_points = max(0, int(args.object_cloud_vis_points))
    if max_points == 0:
        return []
    finite = points_w[np.isfinite(points_w).all(axis=1)]
    if len(finite) == 0:
        return []
    stride = max(1, int(math.ceil(len(finite) / max_points)))
    points = finite[::stride][:max_points]
    size = float(args.object_cloud_point_size)
    meshes = []
    for point in points:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.eye(3, dtype=np.float32) * size
        matrix[:3, 3] = point
        meshes.append(_make_box_vertices_faces(matrix))
    return meshes


def _candidate_material(candidate: CandidateRecord) -> str:
    if candidate.failed:
        return "grasp_failed"
    if candidate.selected:
        return "grasp_selected"
    if candidate.full_safe:
        return "grasp_full_safe"
    if candidate.hand_safe:
        return "grasp_hand_safe"
    return "grasp_unsafe"


def _write_mtl(path: Path) -> None:
    colors = {
        "table": (0.58, 0.58, 0.58),
        "object": (0.82, 0.30, 0.25),
        "object_cloud": (0.18, 0.18, 0.18),
        "grasp_selected": (0.10, 1.00, 0.10),
        "grasp_full_safe": (0.10, 0.48, 1.00),
        "grasp_hand_safe": (1.00, 0.82, 0.10),
        "grasp_unsafe": (1.00, 0.30, 0.05),
        "grasp_failed": (1.00, 0.05, 0.05),
    }
    with path.open("w", encoding="utf-8") as f:
        for name, color in colors.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            f.write("Ka 0.050000 0.050000 0.050000\n")
            f.write("Ks 0.100000 0.100000 0.100000\n\n")


def _append_mesh_obj(
    f,
    name: str,
    material: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_offset: int,
) -> int:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    f.write(f"o {name}\n")
    f.write(f"usemtl {material}\n")
    for vertex in vertices:
        f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
    for face in faces:
        indices = face + 1 + vertex_offset
        f.write("f " + " ".join(str(int(i)) for i in indices) + "\n")
    f.write("\n")
    return vertex_offset + int(vertices.shape[0])


def _write_scene_obj(
    path: Path,
    object_mesh,
    object_transform: np.ndarray,
    point_cloud_w: np.ndarray,
    candidates: list[CandidateRecord],
    args: argparse.Namespace,
    *,
    num_vis_grasps: int | None = None,
) -> None:
    mtl_path = path.with_suffix(".mtl")
    _write_mtl(mtl_path)
    vertex_offset = 0
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n\n")
        table_vertices, table_faces = _table_mesh(args)
        vertex_offset = _append_mesh_obj(f, "table", "table", table_vertices, table_faces, vertex_offset)

        object_vertices = _transform_points(np.asarray(object_mesh.vertices, dtype=np.float32), object_transform)
        object_faces = np.asarray(object_mesh.faces, dtype=np.int64)
        vertex_offset = _append_mesh_obj(f, "object", "object", object_vertices, object_faces, vertex_offset)

        for point_id, (vertices, faces) in enumerate(_point_cloud_meshes(point_cloud_w, args)):
            vertex_offset = _append_mesh_obj(
                f,
                f"object_cloud_{point_id:04d}",
                "object_cloud",
                vertices,
                faces,
                vertex_offset,
            )

        half_width = 0.5 * float(args.grasp_line_width)
        depth = float(args.grasp_line_depth)
        grasp_limit = max(0, int(args.num_vis_grasps if num_vis_grasps is None else num_vis_grasps))
        for candidate in candidates[:grasp_limit]:
            material = _candidate_material(candidate)
            thickness = float(args.grasp_line_thickness) if candidate.selected else float(args.candidate_grasp_line_thickness)
            for part_id, (vertices, faces) in enumerate(
                _grasp_guide_meshes(
                    candidate.matrix_w,
                    half_width=half_width,
                    depth=depth if candidate.selected else depth * 0.82,
                    thickness=thickness,
                )
            ):
                vertex_offset = _append_mesh_obj(
                    f,
                    f"grasp_rank_{candidate.rank:03d}_{part_id}",
                    material,
                    vertices,
                    faces,
                    vertex_offset,
                )


def _candidate_json(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "rank": int(candidate.rank),
        "candidate_index": int(candidate.candidate_index),
        "confidence": float(candidate.confidence),
        "selected": bool(candidate.selected),
        "failed": bool(candidate.failed),
        "table_collision_safe": bool(candidate.full_safe),
        "table_hand_collision_safe": bool(candidate.hand_safe),
        "table_clearance_m": None if not np.isfinite(candidate.full_clearance) else float(candidate.full_clearance),
        "table_hand_clearance_m": None if not np.isfinite(candidate.hand_clearance) else float(candidate.hand_clearance),
        "upward_alignment_score": None if not np.isfinite(candidate.upward_score) else float(candidate.upward_score),
        "grasp_matrix_w": candidate.matrix_w.astype(float).tolist(),
    }


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {path}. Pass --overwrite to write into it.")


def _copy_file_if_exists(src: Path, dst: Path, copied: list[str], missing: list[str]) -> None:
    if not src.is_file():
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def _obj_mtllibs(obj_path: Path) -> list[str]:
    refs: list[str] = []
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0] == "mtllib":
                    refs.extend(item for item in parts[1].split() if item)
    except OSError:
        return refs
    return refs


def _mtl_texture_refs(mtl_path: Path) -> list[str]:
    texture_keys = {
        "map_Ka",
        "map_Kd",
        "map_Ks",
        "map_Ns",
        "map_d",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "norm",
        "refl",
    }
    refs: list[str] = []
    try:
        with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts and parts[0] in texture_keys and len(parts) >= 2:
                    refs.append(parts[-1])
    except OSError:
        return refs
    return refs


def _copy_object_asset_files(asset: ObjectAsset, output_dir: Path) -> dict[str, Any]:
    src_obj = asset.obj_path
    asset_dir = output_dir / "source_objects" / asset.name
    dst_obj = asset_dir / src_obj.name
    copied: list[str] = []
    missing: list[str] = []
    _copy_file_if_exists(src_obj, dst_obj, copied, missing)

    mtl_refs = _obj_mtllibs(src_obj)
    same_stem_mtl = src_obj.with_suffix(".mtl")
    if same_stem_mtl.is_file() and same_stem_mtl.name not in mtl_refs:
        mtl_refs.append(same_stem_mtl.name)

    copied_mtls: list[tuple[Path, Path]] = []
    for ref in mtl_refs:
        ref_path = Path(ref)
        src_mtl = ref_path if ref_path.is_absolute() else src_obj.parent / ref_path
        dst_mtl = asset_dir / (ref_path.name if ref_path.is_absolute() else ref_path)
        before = len(copied)
        _copy_file_if_exists(src_mtl, dst_mtl, copied, missing)
        if len(copied) > before:
            copied_mtls.append((src_mtl, dst_mtl))

    for src_mtl, dst_mtl in copied_mtls:
        for ref in _mtl_texture_refs(src_mtl):
            ref_path = Path(ref)
            src_texture = ref_path if ref_path.is_absolute() else src_mtl.parent / ref_path
            dst_texture = asset_dir / (ref_path.name if ref_path.is_absolute() else ref_path)
            _copy_file_if_exists(src_texture, dst_texture, copied, missing)

    return {
        "source_object_asset_dir": str(asset_dir),
        "source_object_obj_copy": str(dst_obj),
        "source_object_copied_files": copied,
        "source_object_missing_sidecars": missing,
    }


def _vector3_from_mapping(mapping: dict[str, Any], key: str) -> np.ndarray:
    values = mapping.get(key)
    if values is None:
        raise ValueError(f"Missing {key} in lift-eval failure row")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{key} must have 3 values, got {values!r}")
    return arr


def _quat_from_mapping(mapping: dict[str, Any], key: str) -> np.ndarray:
    values = mapping.get(key)
    if values is None:
        raise ValueError(f"Missing {key} in lift-eval failure row")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 4:
        raise ValueError(f"{key} must have 4 values, got {values!r}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _optional_scale_from_mapping(mapping: dict[str, Any]) -> np.ndarray | None:
    values = mapping.get("object_scale_xyz", mapping.get("object_scale"))
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] == 1:
        return np.repeat(arr, 3).astype(np.float32)
    if arr.shape[0] == 3:
        return arr.astype(np.float32)
    raise ValueError(f"object scale must be scalar or 3-vector, got {values!r}")


def _lift_eval_case_from_summary_row(row: dict[str, Any], source_path: Path, row_index: int) -> LiftEvalFailureCase:
    initial_pose = row.get("initial_object_pose")
    if not isinstance(initial_pose, dict):
        raise ValueError(f"summary failure row {row_index} missing initial_object_pose")
    failed_grasps = row.get("failed_grasp_poses", [])
    if not isinstance(failed_grasps, list):
        raise ValueError(f"summary failure row {row_index} has non-list failed_grasp_poses")
    return LiftEvalFailureCase(
        source_path=str(source_path),
        source_kind="summary_json",
        row_index=int(row_index),
        worker_id=row.get("worker_id"),
        episode_index=row.get("episode_index"),
        sample_index=row.get("sample_index"),
        env_id=row.get("env_id"),
        object_index=row.get("object_index"),
        object_name=row.get("object_name"),
        initial_object_pos_w=_vector3_from_mapping(initial_pose, "object_root_pos_w"),
        initial_object_quat_wxyz=_quat_from_mapping(initial_pose, "object_root_quat_wxyz"),
        initial_object_scale_xyz=_optional_scale_from_mapping(initial_pose),
        failed_grasps=[item for item in failed_grasps if isinstance(item, dict)],
        raw=row,
    )


def _lift_eval_case_from_failure_jsonl_row(row: dict[str, Any], source_path: Path, row_index: int) -> LiftEvalFailureCase:
    record = row.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"failure JSONL row {row_index} missing record")
    initial_pose = record.get("initial_state")
    if not isinstance(initial_pose, dict):
        raise ValueError(f"failure JSONL row {row_index} missing record.initial_state")
    failed_grasps = record.get("failed_attempts", [])
    if not isinstance(failed_grasps, list):
        raise ValueError(f"failure JSONL row {row_index} has non-list record.failed_attempts")
    if not failed_grasps:
        candidate_generation = record.get("candidate_generation", {})
        if isinstance(candidate_generation, dict):
            all_candidates = candidate_generation.get("all_candidates", [])
            if isinstance(all_candidates, list):
                failed_grasps = all_candidates
    return LiftEvalFailureCase(
        source_path=str(source_path),
        source_kind="failures_jsonl",
        row_index=int(row_index),
        worker_id=row.get("worker_id"),
        episode_index=row.get("episode_index"),
        sample_index=row.get("sample_index"),
        env_id=row.get("env_id"),
        object_index=row.get("object_index"),
        object_name=row.get("object_name"),
        initial_object_pos_w=_vector3_from_mapping(initial_pose, "object_root_pos_w"),
        initial_object_quat_wxyz=_quat_from_mapping(initial_pose, "object_root_quat_wxyz"),
        initial_object_scale_xyz=_optional_scale_from_mapping(initial_pose),
        failed_grasps=[item for item in failed_grasps if isinstance(item, dict)],
        raw=row,
    )


def _load_lift_eval_failure_cases(args: argparse.Namespace) -> list[LiftEvalFailureCase]:
    cases: list[LiftEvalFailureCase] = []
    if args.lift_eval_json:
        path = Path(args.lift_eval_json).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("failed_grasp_pose_rows", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ValueError(f"{path} failed_grasp_pose_rows must be a list")
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                cases.append(_lift_eval_case_from_summary_row(row, path, row_index))

    if args.lift_eval_failures_jsonl:
        path = Path(args.lift_eval_failures_jsonl).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            for row_index, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and not bool(row.get("success", False)):
                    cases.append(_lift_eval_case_from_failure_jsonl_row(row, path, row_index))

    max_cases = int(args.max_lift_eval_failures)
    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def _resolve_lift_eval_asset(case: LiftEvalFailureCase, assets: list[ObjectAsset]) -> ObjectAsset:
    if case.object_name:
        for asset in assets:
            if asset.name == case.object_name or asset.obj_path.stem == case.object_name:
                return asset
    if case.object_index is not None and 0 <= int(case.object_index) < len(assets):
        return assets[int(case.object_index)]
    raise RuntimeError(
        "Could not resolve object asset for lift-eval failure "
        f"row={case.row_index} object_name={case.object_name!r} object_index={case.object_index!r}"
    )


def _candidate_from_failed_grasp(grasp: dict[str, Any], fallback_rank: int, args: argparse.Namespace) -> CandidateRecord:
    matrix = np.asarray(grasp.get("grasp_matrix_w"), dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"failed grasp has invalid grasp_matrix_w shape: {matrix.shape}")
    tier = str(grasp.get("selection_tier", ""))
    full_safe = bool(grasp.get("table_collision_safe", tier == "full_safe"))
    hand_safe = bool(grasp.get("table_hand_collision_safe", tier in {"full_safe", "hand_safe"}))
    return CandidateRecord(
        rank=int(grasp.get("rank", fallback_rank)),
        candidate_index=int(grasp.get("candidate_index", fallback_rank)),
        confidence=float(grasp.get("confidence", float("nan"))),
        matrix_w=matrix,
        full_safe=full_safe,
        hand_safe=hand_safe,
        full_clearance=float(grasp.get("table_clearance_m", float("nan"))),
        hand_clearance=float(grasp.get("table_hand_clearance_m", float("nan"))),
        upward_score=_upward_alignment_score(matrix, args),
        selected=False,
        failed=bool(grasp.get("success", False) is False and "final_object_z_w" in grasp),
    )


def _safe_stem(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned.strip("._") or "unknown"


def _export_lift_eval_failures(
    args: argparse.Namespace,
    config_defaults: ConfigDefaults,
    assets: list[ObjectAsset],
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    cases = _load_lift_eval_failure_cases(args)
    manifest_rows: list[dict[str, Any]] = []
    asset_cache: dict[Path, SimpleMesh] = {}
    for case_idx, case in enumerate(cases):
        asset = _resolve_lift_eval_asset(case, assets)
        mesh = asset_cache.get(asset.obj_path)
        if mesh is None:
            mesh = _load_mesh(asset.obj_path)
            asset_cache[asset.obj_path] = mesh

        object_scale_xyz = (
            case.initial_object_scale_xyz
            if case.initial_object_scale_xyz is not None
            else np.asarray([float(args.object_scale)] * 3, dtype=np.float32)
        )
        object_transform = _pose_transform_from_pos_quat_scale(
            case.initial_object_pos_w,
            case.initial_object_quat_wxyz,
            object_scale_xyz,
        )
        scene_args = argparse.Namespace(**vars(args))
        scene_table_pose = [float(v) for v in args.table_pose]
        scene_table_pose[0] = float(case.initial_object_pos_w[0])
        scene_table_pose[1] = float(case.initial_object_pos_w[1])
        scene_args.table_pose = scene_table_pose
        if args.include_point_cloud:
            local_points = _sample_mesh_points(mesh, int(args.object_cloud_vis_points), rng)
            point_cloud_w = _transform_points(local_points, object_transform).astype(np.float32)
        else:
            point_cloud_w = np.empty((0, 3), dtype=np.float32)

        candidates = [
            _candidate_from_failed_grasp(grasp, grasp_idx, args)
            for grasp_idx, grasp in enumerate(case.failed_grasps)
        ]
        stem = (
            f"lift_failure_{case_idx:04d}_"
            f"sample_{_safe_stem(case.sample_index if case.sample_index is not None else case.row_index)}_"
            f"env_{_safe_stem(case.env_id if case.env_id is not None else 'x')}_"
            f"{_safe_stem(asset.name)}"
        )
        obj_path = output_dir / f"{stem}.obj"
        json_path = output_dir / f"{stem}.json"
        source_asset_copy = (
            None if args.no_copy_object_assets else _copy_object_asset_files(asset, output_dir)
        )

        _write_scene_obj(
            obj_path,
            mesh,
            object_transform,
            point_cloud_w,
            candidates,
            scene_args,
            num_vis_grasps=len(candidates),
        )
        metadata = {
            "scene_index": int(case_idx),
            "source_kind": case.source_kind,
            "source_path": case.source_path,
            "source_row_index": int(case.row_index),
            "worker_id": case.worker_id,
            "episode_index": case.episode_index,
            "sample_index": case.sample_index,
            "env_id": case.env_id,
            "object_index": case.object_index,
            "object_name": case.object_name or asset.name,
            "object_obj_path": str(asset.obj_path),
            "source_object_asset_copy": source_asset_copy,
            "object_pose": {
                "position_w": case.initial_object_pos_w.astype(float).tolist(),
                "quat_wxyz": case.initial_object_quat_wxyz.astype(float).tolist(),
                "scale_xyz": object_scale_xyz.astype(float).tolist(),
                "scale_source": "eval_json.object_scale_xyz" if case.initial_object_scale_xyz is not None else "--object_scale fallback",
                "pose_source": "eval_graspgen_lift_grasps.initial_state",
            },
            "table_pose": [float(v) for v in scene_args.table_pose],
            "table_pose_note": "XY centered at failed object position for per-env world-coordinate visualization.",
            "table_size": [float(v) for v in args.table_size],
            "config_source": config_defaults.config,
            "config_paths_yaml": config_defaults.paths_yaml,
            "num_failed_grasps": int(len(candidates)),
            "num_exported": int(len(candidates)),
            "candidates": [_candidate_json(candidate) for candidate in candidates],
            "raw_failure_case": case.raw,
            "obj_path": str(obj_path),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        manifest_rows.append(metadata)
        print(
            f"[EXPORT_FAILURE] {obj_path} object={asset.name} failed_grasps={len(candidates)}",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "lift_eval_failures",
                "args": vars(args),
                "config_defaults": {
                    "config": config_defaults.config,
                    "paths_yaml": config_defaults.paths_yaml,
                    "objects_json": None if config_defaults.objects_json is None else str(config_defaults.objects_json),
                    "obj_dir": None if config_defaults.obj_dir is None else str(config_defaults.obj_dir),
                    "table_enabled": config_defaults.table_enabled,
                    "table_size": config_defaults.table_size,
                    "table_pose": config_defaults.table_pose,
                    "initial_position_range": config_defaults.initial_position_range,
                    "scale_range": config_defaults.scale_range,
                    "scale_randomization_enabled": config_defaults.scale_randomization_enabled,
                },
                "num_failure_cases_loaded": len(cases),
                "num_scenes": len(manifest_rows),
                "scenes": manifest_rows,
            },
            f,
            indent=2,
        )
    print(f"[DONE] wrote {len(manifest_rows)} lift-eval failure scenes to {output_dir}", flush=True)


def _load_generated_candidate_rows(path: Path, max_cases: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row_index, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{row_index + 1}")
            row["_candidate_jsonl_row_index"] = int(row_index)
            rows.append(row)
            if max_cases > 0 and len(rows) >= max_cases:
                break
    if not rows:
        raise ValueError(f"No candidate rows found in {path}")
    return rows


def _generated_candidate_asset(row: dict[str, Any], assets: list[ObjectAsset]) -> ObjectAsset:
    object_name = str(row.get("object_name") or "")
    for asset in assets:
        if asset.name == object_name or asset.obj_path.stem == object_name:
            return asset
    direct_path = row.get("object_obj_path")
    if direct_path:
        path = Path(str(direct_path)).expanduser().resolve()
        if path.is_file():
            return ObjectAsset(object_name or path.stem, path)
    raise RuntimeError(
        f"Could not resolve object asset for candidate row={row.get('_candidate_jsonl_row_index')} "
        f"object_name={object_name!r}"
    )


def _load_ply_points(path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to load candidate point-cloud PLY files") from exc
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    points = np.asarray(getattr(loaded, "vertices", None), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"PLY does not contain an Nx3 vertex cloud: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"PLY contains non-finite points: {path}")
    return points


def _optional_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _candidate_from_generated(entry: dict[str, Any], fallback_rank: int, args: argparse.Namespace) -> CandidateRecord:
    matrix = np.asarray(entry.get("grasp_matrix_w"), dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Generated candidate has invalid grasp_matrix_w shape/value: {matrix!r}")
    return CandidateRecord(
        rank=int(entry.get("rank", fallback_rank)),
        candidate_index=int(entry.get("candidate_index", fallback_rank)),
        confidence=float(entry.get("confidence", float("nan"))),
        matrix_w=matrix,
        full_safe=bool(entry.get("table_collision_safe", False)),
        hand_safe=bool(entry.get("table_hand_collision_safe", False)),
        full_clearance=_optional_float(entry.get("table_clearance_m")),
        hand_clearance=_optional_float(entry.get("table_hand_clearance_m")),
        upward_score=_upward_alignment_score(matrix, args),
        selected=bool(entry.get("selected", False)),
        failed=False,
    )


def _generated_candidate_entries(row: dict[str, Any], filter_mode: str) -> list[dict[str, Any]]:
    generation = row.get("candidate_generation")
    if not isinstance(generation, dict):
        raise ValueError(f"Candidate row {row.get('_candidate_jsonl_row_index')} has no candidate_generation")
    if filter_mode == "full_safe":
        entries = generation.get("full_safe_candidates", [])
    elif filter_mode == "full_and_hand_safe":
        entries = list(generation.get("full_safe_candidates", [])) + list(
            generation.get("hand_safe_candidates", [])
        )
    else:
        entries = generation.get("all_candidates", [])
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError(
            f"Candidate row {row.get('_candidate_jsonl_row_index')} has invalid entries for filter={filter_mode}"
        )
    return sorted(entries, key=lambda item: int(item.get("rank", 0)))


def _export_generated_candidates(
    args: argparse.Namespace,
    config_defaults: ConfigDefaults,
    assets: list[ObjectAsset],
    output_dir: Path,
) -> None:
    source_path = Path(args.grasp_candidates_jsonl).expanduser().resolve()
    rows = _load_generated_candidate_rows(source_path, int(args.max_grasp_candidate_cases))
    manifest_rows: list[dict[str, Any]] = []
    asset_cache: dict[Path, SimpleMesh] = {}

    for scene_index, row in enumerate(rows):
        asset = _generated_candidate_asset(row, assets)
        object_mesh = asset_cache.get(asset.obj_path)
        if object_mesh is None:
            object_mesh = _load_mesh(asset.obj_path)
            asset_cache[asset.obj_path] = object_mesh

        initial_state = row.get("initial_state")
        if not isinstance(initial_state, dict):
            raise ValueError(f"Candidate row {scene_index} has no initial_state")
        object_pos = _vector3_from_mapping(initial_state, "object_root_pos_w")
        object_quat = _quat_from_mapping(initial_state, "object_root_quat_wxyz")
        object_scale = _optional_scale_from_mapping(initial_state)
        if object_scale is None:
            raise ValueError(f"Candidate row {scene_index} has no object_scale_xyz")
        object_transform = _pose_transform_from_pos_quat_scale(object_pos, object_quat, object_scale)

        pointcloud = row.get("pointcloud")
        if not isinstance(pointcloud, dict) or not pointcloud.get("canonical_path"):
            raise ValueError(f"Candidate row {scene_index} has no pointcloud.canonical_path")
        pointcloud_path = Path(str(pointcloud["canonical_path"])).expanduser().resolve()
        if not pointcloud_path.is_file():
            raise FileNotFoundError(pointcloud_path)
        points_local = _load_ply_points(pointcloud_path)
        point_cloud_w = _transform_points(points_local, object_transform).astype(np.float32)

        table = row.get("table")
        if not isinstance(table, dict):
            raise ValueError(f"Candidate row {scene_index} has no table metadata")
        scene_args = argparse.Namespace(**vars(args))
        scene_args.table_pose = [float(value) for value in table.get("pose_w", args.table_pose)]
        scene_args.table_size = [float(value) for value in table.get("size", args.table_size)]

        entries = _generated_candidate_entries(row, str(args.grasp_candidates_filter))
        candidates = [
            _candidate_from_generated(entry, rank, scene_args)
            for rank, entry in enumerate(entries)
        ]
        if candidates and not any(candidate.selected for candidate in candidates):
            candidates[0].selected = True
        elif sum(1 for candidate in candidates if candidate.selected) > 1:
            first_selected = next(candidate for candidate in candidates if candidate.selected)
            for candidate in candidates:
                candidate.selected = candidate is first_selected

        stem = f"grasp_candidates_{scene_index:04d}_{_safe_stem(asset.name)}"
        obj_path = output_dir / f"{stem}.obj"
        json_path = output_dir / f"{stem}.json"
        source_asset_copy = (
            None if args.no_copy_object_assets else _copy_object_asset_files(asset, output_dir)
        )
        _write_scene_obj(
            obj_path,
            object_mesh,
            object_transform,
            point_cloud_w,
            candidates,
            scene_args,
        )

        num_exported = min(len(candidates), max(0, int(args.num_vis_grasps)))
        metadata = {
            "scene_index": int(scene_index),
            "source_kind": "generated_grasp_candidates_jsonl",
            "source_path": str(source_path),
            "source_row_index": int(row.get("_candidate_jsonl_row_index", scene_index)),
            "object_name": asset.name,
            "object_obj_path": str(asset.obj_path),
            "source_object_asset_copy": source_asset_copy,
            "object_pose": {
                "position_w": object_pos.astype(float).tolist(),
                "quat_wxyz": object_quat.astype(float).tolist(),
                "scale_xyz": object_scale.astype(float).tolist(),
            },
            "table_pose": scene_args.table_pose,
            "table_size": scene_args.table_size,
            "pointcloud_path": str(pointcloud_path),
            "pointcloud_points": int(len(points_local)),
            "pointcloud_world_bounds": _points_bounds(point_cloud_w),
            "candidate_filter": str(args.grasp_candidates_filter),
            "num_candidates_loaded": int(len(candidates)),
            "num_exported": int(num_exported),
            "candidates": [_candidate_json(candidate) for candidate in candidates],
            "candidate_generation_stats": row.get("candidate_generation", {}).get("stats"),
            "raw_candidate_row": {key: value for key, value in row.items() if not key.startswith("_")},
            "obj_path": str(obj_path),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        manifest_rows.append(metadata)
        print(
            f"[EXPORT_CANDIDATES] {obj_path} object={asset.name} "
            f"filter={args.grasp_candidates_filter} candidates={len(candidates)} exported={num_exported}",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "generated_grasp_candidates",
                "source_path": str(source_path),
                "candidate_filter": str(args.grasp_candidates_filter),
                "config_defaults": {
                    "config": config_defaults.config,
                    "paths_yaml": config_defaults.paths_yaml,
                    "objects_json": None if config_defaults.objects_json is None else str(config_defaults.objects_json),
                    "obj_dir": None if config_defaults.obj_dir is None else str(config_defaults.obj_dir),
                },
                "num_scenes": len(manifest_rows),
                "scenes": manifest_rows,
            },
            f,
            indent=2,
        )
    print(f"[DONE] wrote {len(manifest_rows)} generated-candidate scenes to {output_dir}", flush=True)


def main() -> None:
    args = _parse_args()
    input_modes = [
        bool(args.lift_eval_json),
        bool(args.lift_eval_failures_jsonl),
        bool(args.grasp_candidates_jsonl),
    ]
    if sum(input_modes) > 1:
        raise ValueError(
            "--lift_eval_json, --lift_eval_failures_jsonl, and --grasp_candidates_jsonl "
            "are mutually exclusive"
        )
    config_defaults = _finalize_args(args)
    if args.num_scenes <= 0:
        raise ValueError(f"--num_scenes must be positive, got {args.num_scenes}")
    if args.num_vis_grasps < 0:
        raise ValueError(f"--num_vis_grasps must be >= 0, got {args.num_vis_grasps}")
    if args.max_lift_eval_failures < 0:
        raise ValueError(f"--max_lift_eval_failures must be >= 0, got {args.max_lift_eval_failures}")
    if args.max_grasp_candidate_cases < 0:
        raise ValueError(
            f"--max_grasp_candidate_cases must be >= 0, got {args.max_grasp_candidate_cases}"
        )
    if args.graspgen_point_cloud_points <= 0:
        raise ValueError("--graspgen_point_cloud_points must be positive")

    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, args.overwrite)
    assets = _load_object_assets(args)
    rng = np.random.default_rng(int(args.seed))
    if args.grasp_candidates_jsonl:
        _export_generated_candidates(args, config_defaults, assets, output_dir)
        return
    if args.lift_eval_json or args.lift_eval_failures_jsonl:
        _export_lift_eval_failures(args, config_defaults, assets, output_dir, rng)
        return

    client = None
    if not args.dry_run_random_grasps:
        _add_graspgen_to_path(args.graspgen_root)
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host=args.graspgen_host,
            port=int(args.graspgen_port),
            timeout_ms=int(args.graspgen_timeout_ms),
        )

    manifest_rows: list[dict[str, Any]] = []
    for scene_idx in range(int(args.num_scenes)):
        asset = assets[int(rng.integers(0, len(assets)))]
        mesh = _load_mesh(asset.obj_path)
        scale, scale_components = _sample_object_scale(asset, args, rng)
        object_transform, object_pose = _random_object_transform(mesh, scale, args, rng)
        local_points = _sample_mesh_points(mesh, int(args.graspgen_point_cloud_points), rng)
        point_cloud_w = _transform_points(local_points, object_transform).astype(np.float32)
        finite_mask = np.isfinite(point_cloud_w).all(axis=1)
        pc_finite_w = point_cloud_w[finite_mask]
        pc_mean_w = (
            np.nanmean(pc_finite_w, axis=0).astype(np.float32)
            if pc_finite_w.shape[0] > 0
            else np.zeros((3,), dtype=np.float32)
        )

        if args.dry_run_random_grasps:
            pc_centered = None
            grasps, confidences = _fake_grasps(point_cloud_w, rng, min(max(int(args.topk_num_grasps), 1), 32))
            used_remove_outliers = False
            error = None
        else:
            assert client is not None
            pc_centered = np.empty((0, 3), dtype=np.float32)
            if pc_finite_w.shape[0] == 0:
                grasps = np.empty((0, 4, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
                used_remove_outliers = False
                error = "object point cloud has no finite points"
            else:
                pc_centered = (pc_finite_w - pc_mean_w.reshape(1, 3)).astype(np.float32)
                grasps, confidences, used_remove_outliers, error = _infer_grasps(client, pc_centered, args)

        candidates = _rank_candidates(grasps, confidences, pc_mean_w, args) if len(grasps) else []
        safe_count = sum(1 for candidate in candidates if candidate.full_safe)
        hand_safe_count = sum(1 for candidate in candidates if candidate.hand_safe)
        selected = next((candidate for candidate in candidates if candidate.selected), None)
        stem = f"scene_{scene_idx:04d}_{asset.name}"
        obj_path = output_dir / f"{stem}.obj"
        json_path = output_dir / f"{stem}.json"
        source_asset_copy = (
            None if args.no_copy_object_assets else _copy_object_asset_files(asset, output_dir)
        )

        _write_scene_obj(obj_path, mesh, object_transform, point_cloud_w, candidates, args)
        metadata = {
            "scene_index": int(scene_idx),
            "object_name": asset.name,
            "object_obj_path": str(asset.obj_path),
            "source_object_asset_copy": source_asset_copy,
            "object_pose": object_pose,
            "object_scale_components": scale_components,
            "table_pose": [float(v) for v in args.table_pose],
            "table_size": [float(v) for v in args.table_size],
            "config_source": config_defaults.config,
            "config_paths_yaml": config_defaults.paths_yaml,
            "config_table_enabled": config_defaults.table_enabled,
            "point_cloud_points": int(point_cloud_w.shape[0]),
            "point_cloud_finite_points": int(pc_finite_w.shape[0]),
            "point_cloud_mean_w": pc_mean_w.tolist(),
            "point_cloud_world_bounds": _points_bounds(point_cloud_w),
            "graspgen_input_cloud_bounds": _points_bounds(
                point_cloud_w if args.dry_run_random_grasps else pc_centered
            ),
            "graspgen_input_frame": "world" if args.dry_run_random_grasps else "centered_at_point_cloud_mean",
            "graspgen_error": error,
            "used_remove_outliers": bool(used_remove_outliers),
            "num_returned": int(len(grasps)),
            "num_exported": int(min(len(candidates), max(0, int(args.num_vis_grasps)))),
            "full_safe_candidates": int(safe_count),
            "hand_safe_candidates": int(hand_safe_count),
            "selected_rank": None if selected is None else int(selected.rank),
            "selected_candidate_index": None if selected is None else int(selected.candidate_index),
            "selected_confidence": None if selected is None else float(selected.confidence),
            "candidates": [_candidate_json(candidate) for candidate in candidates],
            "obj_path": str(obj_path),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        manifest_rows.append(metadata)
        print(
            f"[EXPORT] {obj_path} object={asset.name} grasps={len(grasps)} "
            f"full_safe={safe_count} hand_safe={hand_safe_count} selected_rank={metadata['selected_rank']}",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "config_defaults": {
                    "config": config_defaults.config,
                    "paths_yaml": config_defaults.paths_yaml,
                    "objects_json": None if config_defaults.objects_json is None else str(config_defaults.objects_json),
                    "obj_dir": None if config_defaults.obj_dir is None else str(config_defaults.obj_dir),
                    "table_enabled": config_defaults.table_enabled,
                    "table_size": config_defaults.table_size,
                    "table_pose": config_defaults.table_pose,
                    "initial_position_range": config_defaults.initial_position_range,
                    "scale_range": config_defaults.scale_range,
                    "scale_randomization_enabled": config_defaults.scale_randomization_enabled,
                },
                "num_scenes": len(manifest_rows),
                "scenes": manifest_rows,
            },
            f,
            indent=2,
        )
    print(f"[DONE] wrote {len(manifest_rows)} scenes to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

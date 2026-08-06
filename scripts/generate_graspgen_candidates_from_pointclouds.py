#!/usr/bin/env python3
"""Generate and table-filter GraspGen candidates from canonical point-cloud PLY files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GRASPGEN_ROOT = Path("/mnt/project/world_model/tool_generalist/GraspGen")
DEFAULT_OBJ_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore object poses from an existing lift-eval JSONL, transform precomputed canonical "
            "point clouds into those poses, call GraspGen, and retain strict table-safe candidates."
        )
    )
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--pointcloud-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--pointcloud-suffix", default="_first_hit_fps_2048.ply")
    parser.add_argument("--pointcloud-metadata", type=Path, default=None)
    parser.add_argument("--obj-dir", type=Path, default=DEFAULT_OBJ_DIR)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--graspgen-root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--graspgen-host", default="localhost")
    parser.add_argument("--graspgen-port", type=int, default=5556)
    parser.add_argument("--graspgen-timeout-ms", type=int, default=120_000)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk-num-grasps", type=int, default=64)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--min-grasps", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument(
        "--allow-non-franka-server",
        action="store_true",
        help="Allow server metadata whose gripper_name is not franka_panda.",
    )
    parser.add_argument(
        "--dry-run-random-grasps",
        action="store_true",
        help="Generate deterministic fake candidates without connecting to GraspGen.",
    )

    parser.add_argument("--table-collision-clearance", type=float, default=0.005)
    parser.add_argument("--table-collision-xy-margin", type=float, default=0.0)
    parser.add_argument("--table-size", type=float, nargs=3, default=(1.0, 1.0, 0.04))
    parser.add_argument("--table-pose", type=float, nargs=3, default=(0.0, 0.0, -0.02))
    parser.add_argument("--no-table-collision-filter", action="store_true")
    parser.add_argument("--grasp-pose-frame", choices=("base", "tcp"), default="base")
    parser.add_argument("--panda-hand-to-tcp-z", type=float, default=0.107)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector of shape ({size},), got {value!r}")
    return array


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.astype(np.float64) / max(float(np.linalg.norm(quat)), 1.0e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = [(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = [(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
    quat_array = np.asarray(quat, dtype=np.float32)
    return quat_array / max(float(np.linalg.norm(quat_array)), 1.0e-8)


def _load_ply_points(path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to load point-cloud PLY files") from exc
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    points = np.asarray(getattr(loaded, "vertices", None), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"PLY does not contain an Nx3 vertex cloud: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"PLY contains non-finite points: {path}")
    return points


def _load_source_rows(path: Path, max_objects: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{row_index + 1}")
            row["_source_row_index"] = int(row_index)
            rows.append(row)
            if max_objects > 0 and len(rows) >= max_objects:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _initial_state(row: dict[str, Any]) -> dict[str, Any]:
    record = row.get("record")
    if not isinstance(record, dict) or not isinstance(record.get("initial_state"), dict):
        raise ValueError(f"Source row {row.get('_source_row_index')} has no record.initial_state")
    return record["initial_state"]


def _source_table(row: dict[str, Any], args: argparse.Namespace, robot_root_pos: np.ndarray) -> dict[str, Any]:
    record = row.get("record", {})
    generation = record.get("candidate_generation", {}) if isinstance(record, dict) else {}
    stats = generation.get("candidate_stats", {}) if isinstance(generation, dict) else {}
    bounds = stats.get("table_bounds_xy_w") if isinstance(stats, dict) else None
    top_z = stats.get("table_top_z_w") if isinstance(stats, dict) else None
    if bounds is not None and top_z is not None:
        bounds_xy = np.asarray(bounds, dtype=np.float32)
        if bounds_xy.shape == (2, 2) and np.isfinite(bounds_xy).all() and np.isfinite(float(top_z)):
            size_xy = bounds_xy[1] - bounds_xy[0]
            center_xy = bounds_xy.mean(axis=0)
            size_z = float(args.table_size[2])
            return {
                "bounds_xy_w": bounds_xy,
                "top_z_w": float(top_z),
                "pose_w": [float(center_xy[0]), float(center_xy[1]), float(top_z) - 0.5 * size_z],
                "size": [float(size_xy[0]), float(size_xy[1]), size_z],
                "source": "source_jsonl.candidate_stats",
            }

    pose = np.asarray(args.table_pose, dtype=np.float32).copy()
    pose[:2] += robot_root_pos[:2]
    size = np.asarray(args.table_size, dtype=np.float32)
    half_xy = 0.5 * size[:2] + float(args.table_collision_xy_margin)
    bounds_xy = np.stack((pose[:2] - half_xy, pose[:2] + half_xy), axis=0)
    return {
        "bounds_xy_w": bounds_xy,
        "top_z_w": float(pose[2] + 0.5 * size[2]),
        "pose_w": pose.astype(float).tolist(),
        "size": size.astype(float).tolist(),
        "source": "robot_root_plus_cli_table",
    }


def _box_corner_points(center: tuple[float, float, float], extent: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    hx, hy, hz = (0.5 * float(value) for value in extent)
    return np.asarray(
        [[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float32,
    )


def _gripper_proxy_points() -> dict[str, np.ndarray]:
    width = 0.10537486
    depth = 0.10527314
    palm = _box_corner_points((0.0, 0.0, 0.005), (0.120, 0.075, 0.050))
    left = _box_corner_points((0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    right = _box_corner_points((-0.5 * width, 0.0, 0.5 * depth), (0.020, 0.025, depth))
    return {"all": np.concatenate((palm, left, right), axis=0), "hand": palm}


def _gripper_base_matrix(grasp_matrix: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    matrix = np.asarray(grasp_matrix, dtype=np.float32).copy()
    if args.grasp_pose_frame == "tcp":
        matrix[:3, 3] -= matrix[:3, :3] @ np.asarray([0.0, 0.0, float(args.panda_hand_to_tcp_z)], dtype=np.float32)
    return matrix


def _table_clearance(grasp_matrix: np.ndarray, proxy_points: np.ndarray, table: dict[str, Any], args: argparse.Namespace) -> tuple[bool, float]:
    if args.no_table_collision_filter:
        return True, float("inf")
    matrix = _gripper_base_matrix(grasp_matrix, args)
    points_w = proxy_points @ matrix[:3, :3].T + matrix[:3, 3]
    bounds = np.asarray(table["bounds_xy_w"], dtype=np.float32)
    low = bounds[0]
    high = bounds[1]
    mask = (
        np.isfinite(points_w).all(axis=1)
        & (points_w[:, 0] >= low[0])
        & (points_w[:, 0] <= high[0])
        & (points_w[:, 1] >= low[1])
        & (points_w[:, 1] <= high[1])
    )
    if not np.any(mask):
        return True, float("inf")
    clearance = float(np.min(points_w[mask, 2] - float(table["top_z_w"])))
    return clearance >= float(args.table_collision_clearance), clearance


def _validate_grasp_matrix(matrix: np.ndarray) -> tuple[bool, str | None]:
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False, "matrix must be finite 4x4"
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-4):
        return False, "invalid homogeneous last row"
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        return False, "rotation is not orthonormal"
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2.0e-3:
        return False, "rotation determinant is not +1"
    return True, None


def _fake_grasps(count: int) -> tuple[np.ndarray, np.ndarray]:
    grasps = []
    confidences = []
    for index in range(count):
        yaw = 2.0 * math.pi * index / max(count, 1)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.asarray(
            [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )
        matrix[0, 0] *= -1.0
        matrix[1, 0] *= -1.0
        matrix[:3, 3] = [0.04 * math.cos(yaw), 0.04 * math.sin(yaw), 0.11]
        grasps.append(matrix)
        confidences.append(1.0 - index / max(count, 1))
    return np.stack(grasps), np.asarray(confidences, dtype=np.float32)


def _infer(client, pointcloud_centered: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    remove_options = [bool(args.remove_outliers)] + ([False] if args.remove_outliers else [])
    error = None
    for remove_outliers in remove_options:
        for attempt in range(int(args.max_tries)):
            try:
                grasps, confidences = client.infer(
                    pointcloud_centered,
                    grasp_threshold=float(args.grasp_threshold),
                    num_grasps=int(args.num_grasps),
                    topk_num_grasps=int(args.topk_num_grasps),
                    min_grasps=int(args.min_grasps),
                    max_tries=int(args.max_tries),
                    remove_outliers=remove_outliers,
                )
                grasps = np.asarray(grasps, dtype=np.float32)
                confidences = np.asarray(confidences, dtype=np.float32)
                if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) == 0:
                    raise RuntimeError(f"GraspGen returned invalid grasp shape {grasps.shape}")
                if confidences.shape != (len(grasps),):
                    raise RuntimeError(f"GraspGen returned invalid confidence shape {confidences.shape}")
                return grasps, confidences, bool(remove_outliers), None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[WARNING] inference attempt={attempt + 1}/{args.max_tries} error={error}", flush=True)
                time.sleep(0.25)
    return np.empty((0, 4, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), bool(args.remove_outliers), error


def _candidate_records(grasps: np.ndarray, confidences: np.ndarray, mean_w: np.ndarray, table: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    proxy = _gripper_proxy_points()
    all_candidates: list[dict[str, Any]] = []
    rejected_invalid = 0
    order = np.argsort(-confidences)
    for rank, candidate_index in enumerate(order):
        matrix = np.asarray(grasps[int(candidate_index)], dtype=np.float32).copy()
        matrix[:3, 3] += mean_w
        valid, invalid_reason = _validate_grasp_matrix(matrix)
        if not valid:
            rejected_invalid += 1
            full_safe = hand_safe = False
            full_clearance = hand_clearance = float("-inf")
        else:
            full_safe, full_clearance = _table_clearance(matrix, proxy["all"], table, args)
            hand_safe, hand_clearance = _table_clearance(matrix, proxy["hand"], table, args)
        if full_safe:
            tier = "full_safe"
        elif hand_safe:
            tier = "hand_safe"
        else:
            tier = "unsafe"
        approach = matrix[:3, 2]
        entry = {
            "rank": int(rank),
            "candidate_index": int(candidate_index),
            "confidence": float(confidences[int(candidate_index)]),
            "valid_se3": bool(valid),
            "invalid_reason": invalid_reason,
            "selection_tier": tier,
            "table_collision_safe": bool(full_safe),
            "table_hand_collision_safe": bool(hand_safe),
            "table_clearance_m": None if not np.isfinite(full_clearance) else float(full_clearance),
            "table_hand_clearance_m": None if not np.isfinite(hand_clearance) else float(hand_clearance),
            "grasp_matrix_w": matrix.astype(float).tolist(),
            "grasp_quat_wxyz": _matrix_to_quat_wxyz(matrix[:3, :3]).astype(float).tolist(),
            "approach_dir_w": approach.astype(float).tolist(),
        }
        all_candidates.append(entry)

    full_safe_candidates = [item for item in all_candidates if item["valid_se3"] and item["table_collision_safe"]]
    hand_safe_candidates = [
        item for item in all_candidates
        if item["valid_se3"] and item["table_hand_collision_safe"] and not item["table_collision_safe"]
    ]
    unsafe_candidates = [item for item in all_candidates if item not in full_safe_candidates and item not in hand_safe_candidates]
    selected = full_safe_candidates[0] if full_safe_candidates else None
    if selected is not None:
        selected["selected"] = True
    return {
        "all_candidates": all_candidates,
        "full_safe_candidates": full_safe_candidates,
        "hand_safe_candidates": hand_safe_candidates,
        "unsafe_candidates": unsafe_candidates,
        "selected_candidate": selected,
        "stats": {
            "num_returned": int(len(grasps)),
            "full_safe_candidates": int(len(full_safe_candidates)),
            "hand_safe_candidates": int(len(hand_safe_candidates)),
            "unsafe_candidates": int(len(unsafe_candidates)),
            "invalid_se3_candidates": int(rejected_invalid),
            "strict_legal_definition": "valid_se3_and_full_gripper_table_safe",
            "table_collision_clearance_m": float(args.table_collision_clearance),
        },
    }


def _load_sampling_metadata(path: Path | None, pointcloud_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = path or pointcloud_dir.parent / "samples.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Pointcloud metadata must be a JSON list: {metadata_path}")
    return {str(row["object_name"]): row for row in rows if isinstance(row, dict) and row.get("object_name")}


def main() -> int:
    args = _parse_args()
    source_path = args.source_jsonl.expanduser().resolve()
    pointcloud_dir = args.pointcloud_dir.expanduser().resolve()
    output_path = args.output_jsonl.expanduser().resolve()
    obj_dir = args.obj_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not pointcloud_dir.is_dir():
        raise FileNotFoundError(pointcloud_dir)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    if args.max_objects < 0:
        raise ValueError("--max-objects must be non-negative")

    rows = _load_source_rows(source_path, int(args.max_objects))
    sampling_metadata = _load_sampling_metadata(args.pointcloud_metadata, pointcloud_dir)
    client = None
    server_metadata: dict[str, Any] | None = None
    if not args.dry_run_random_grasps:
        graspgen_root = args.graspgen_root.expanduser().resolve()
        if not graspgen_root.is_dir():
            raise FileNotFoundError(graspgen_root)
        sys.path.insert(0, str(graspgen_root))
        from grasp_gen.serving.zmq_client import GraspGenClient

        client = GraspGenClient(
            host=str(args.graspgen_host),
            port=int(args.graspgen_port),
            timeout_ms=int(args.graspgen_timeout_ms),
        )
        server_metadata = dict(client.server_metadata or {})
        gripper_name = str(server_metadata.get("gripper_name", ""))
        if gripper_name != "franka_panda" and not args.allow_non_franka_server:
            raise RuntimeError(f"Expected franka_panda GraspGen server, got {gripper_name!r}")

    output_rows: list[dict[str, Any]] = []
    try:
        for output_index, source_row in enumerate(rows):
            object_name = str(source_row.get("object_name") or "")
            if not object_name:
                raise ValueError(f"Source row {source_row['_source_row_index']} has no object_name")
            state = _initial_state(source_row)
            object_pos = _require_vector(state.get("object_root_pos_w"), 3, "object_root_pos_w")
            object_quat = _require_vector(state.get("object_root_quat_wxyz"), 4, "object_root_quat_wxyz")
            object_scale = _require_vector(state.get("object_scale_xyz"), 3, "object_scale_xyz")
            robot_root_pos = _require_vector(state.get("robot_root_pos_w"), 3, "robot_root_pos_w")
            pointcloud_path = pointcloud_dir / f"{object_name}{args.pointcloud_suffix}"
            if not pointcloud_path.is_file():
                raise FileNotFoundError(f"Missing pointcloud for {object_name}: {pointcloud_path}")
            object_obj_path = obj_dir / f"{object_name}.obj"
            if not object_obj_path.is_file():
                raise FileNotFoundError(f"Missing object OBJ for {object_name}: {object_obj_path}")

            points_local = _load_ply_points(pointcloud_path)
            rotation = _quat_wxyz_to_matrix(object_quat)
            points_w = (points_local * object_scale.reshape(1, 3)) @ rotation.T + object_pos.reshape(1, 3)
            mean_w = points_w.mean(axis=0).astype(np.float32)
            points_centered = (points_w - mean_w.reshape(1, 3)).astype(np.float32)
            table = _source_table(source_row, args, robot_root_pos)
            if args.dry_run_random_grasps:
                grasps, confidences = _fake_grasps(min(max(int(args.topk_num_grasps), 1), 16))
                used_remove_outliers = False
                inference_error = None
            else:
                assert client is not None
                grasps, confidences, used_remove_outliers, inference_error = _infer(client, points_centered, args)
            generation = _candidate_records(grasps, confidences, mean_w, table, args)

            output_row = {
                "schema_version": 1,
                "source": {
                    "jsonl_path": str(source_path),
                    "row_index": int(source_row["_source_row_index"]),
                    "worker_id": source_row.get("worker_id"),
                    "episode_index": source_row.get("episode_index"),
                    "sample_index": source_row.get("sample_index"),
                    "env_id": source_row.get("env_id"),
                },
                "object_name": object_name,
                "object_index": source_row.get("object_index"),
                "object_obj_path": str(object_obj_path),
                "initial_state": state,
                "table": {
                    **table,
                    "bounds_xy_w": np.asarray(table["bounds_xy_w"]).astype(float).tolist(),
                },
                "pointcloud": {
                    "canonical_path": str(pointcloud_path),
                    "canonical_sha256": _sha256(pointcloud_path),
                    "canonical_points": int(len(points_local)),
                    "canonical_bounds": [points_local.min(axis=0).astype(float).tolist(), points_local.max(axis=0).astype(float).tolist()],
                    "world_mean_w": mean_w.astype(float).tolist(),
                    "world_bounds": [points_w.min(axis=0).astype(float).tolist(), points_w.max(axis=0).astype(float).tolist()],
                    "sampling_metadata": sampling_metadata.get(object_name),
                },
                "graspgen": {
                    "host": str(args.graspgen_host),
                    "port": int(args.graspgen_port),
                    "server_metadata": server_metadata,
                    "input_frame": "centered_at_pointcloud_world_mean",
                    "input_points": int(len(points_centered)),
                    "num_grasps": int(args.num_grasps),
                    "topk_num_grasps": int(args.topk_num_grasps),
                    "grasp_threshold": float(args.grasp_threshold),
                    "used_remove_outliers": bool(used_remove_outliers),
                    "error": inference_error,
                    "dry_run_random_grasps": bool(args.dry_run_random_grasps),
                },
                "candidate_generation": generation,
            }
            output_rows.append(output_row)
            stats = generation["stats"]
            print(
                f"[CANDIDATES] {output_index + 1}/{len(rows)} object={object_name} "
                f"returned={stats['num_returned']} full_safe={stats['full_safe_candidates']} "
                f"hand_safe={stats['hand_safe_candidates']} unsafe={stats['unsafe_candidates']}",
                flush=True,
            )
    finally:
        if client is not None:
            client.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(f"[DONE] wrote rows={len(output_rows)} output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

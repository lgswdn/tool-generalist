#!/usr/bin/env python3
"""Compare GraspGen feasibility with recorded GraspSim episode outcomes.

For every recorded object in an ``episodes.jsonl`` file this script restores
its initial scale and world pose, sends a point cloud in that pose to GraspGen,
and keeps candidates for which the GraspGen Panda *hand* collision mesh
intersects neither the object nor its supporting ground/table plane.

Multiple ``--backend`` arguments create persistent workers backed by a shared
queue.  Faster servers therefore receive more work automatically.  The main
thread is the only progress reporter and JSONL writer; optional per-task OBJ
scenes are written by the worker that already owns the transformed geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EPISODES = Path(
    "/mnt/project/world_model/tool_generalist/grasp_result_dgn_full_yes/"
    "conclusions/episodes.jsonl"
)
DEFAULT_DGN_MESH_DIR = Path(
    "/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized"
)
DEFAULT_POINTCLOUD_DIR = Path(
    "/mnt/project/world_model/tool_generalist/assets/DGN/first_hit_fps_pointclouds/npy"
)
DEFAULT_GRASPGEN_ROOT = Path("/mnt/project/world_model/tool_generalist/GraspGen")
DEFAULT_OUTPUT = Path("scripts/outputs/graspgen_episode_consistency.jsonl")


@dataclass(frozen=True)
class Backend:
    name: str
    host: str
    port: int
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeTask:
    index: int
    source_line: int
    record: dict[str, Any]
    object_name: str
    scale: float
    position: np.ndarray
    orientation: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore every recorded GraspSim episode object pose, generate GraspGen "
            "candidates, reject Panda-hand/object or Panda-hand/ground intersections, "
            "and compare candidate availability with the recorded success flag."
        )
    )
    parser.add_argument("--episodes-jsonl", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_DGN_MESH_DIR)
    parser.add_argument("--graspgen-root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        metavar="[NAME=]HOST:PORT",
        help=(
            "GraspGen endpoint; repeat for multiple backends. Tasks are dynamically "
            "assigned to the next free backend. Default: localhost:5556."
        ),
    )
    parser.add_argument("--graspgen-timeout-ms", type=int, default=120_000)
    parser.add_argument("--allow-non-franka-server", action="store_true")

    parser.add_argument("--pointcloud-points", type=int, default=2048)
    parser.add_argument(
        "--pointcloud-dir",
        type=Path,
        default=DEFAULT_POINTCLOUD_DIR,
        help="Directory containing fixed canonical first-hit FPS NPY point clouds.",
    )
    parser.add_argument(
        "--pointcloud-suffix",
        default=None,
        help=(
            "Filename suffix after object ID. Default: "
            "_first_hit_fps_<pointcloud-points>.npy"
        ),
    )
    parser.add_argument(
        "--object-cache-size",
        type=int,
        default=64,
        help="Per-backend LRU size for canonical meshes and point clouds.",
    )

    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk-num-grasps", type=int, default=64)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--min-grasps", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--remove-outliers", action="store_true")

    parser.add_argument(
        "--ground-size",
        type=float,
        default=4.0,
        help="Side length of the square collision ground/table proxy in metres.",
    )
    parser.add_argument("--ground-thickness", type=float, default=0.10)
    parser.add_argument(
        "--ground-top-offset",
        type=float,
        default=0.0,
        help="Offset added to the support height inferred from the transformed object bottom.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0,
        help="FCL collision security margin in metres; zero means strict mesh intersection.",
    )

    parser.add_argument(
        "--export-viz-objs",
        action="store_true",
        help=(
            "Write one OBJ+MTL scene per task containing the posed object, support "
            "plane, and collision-free grasp guides."
        ),
    )
    parser.add_argument(
        "--viz-output-dir",
        type=Path,
        default=None,
        help="Visualization directory. Default: <output-jsonl-stem>_viz beside the JSONL.",
    )
    parser.add_argument(
        "--viz-max-grasps",
        type=int,
        default=0,
        help="Maximum safe grasps rendered per OBJ; 0 renders every safe grasp.",
    )
    parser.add_argument("--viz-grasp-width", type=float, default=0.08)
    parser.add_argument("--viz-grasp-depth", type=float, default=0.10)
    parser.add_argument("--viz-grasp-thickness", type=float, default=0.0025)
    parser.add_argument(
        "--viz-include-panda-hand",
        action="store_true",
        help=(
            "Also place the official GraspGen Panda hand mesh at every visualized "
            "collision-free grasp pose."
        ),
    )
    parser.add_argument(
        "--viz-panda-hand-opacity",
        type=float,
        default=0.30,
        help="Panda hand MTL opacity in [0, 1]. Default: 0.30.",
    )

    parser.add_argument(
        "--task-mode",
        choices=("object", "pose", "episode"),
        default="object",
        help=(
            "object: one initial pose per object and any-success outcome (default); "
            "pose: one task per exact object/scale/full-pose tuple; "
            "episode: every JSONL row."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run-random-grasps",
        action="store_true",
        help="Do not connect to servers; generate deterministic fake candidates for testing.",
    )
    return parser.parse_args()


def _finite_vector(value: Any, size: int, label: str, line_number: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(
            f"{label} at episodes line {line_number} must be a finite ({size},) vector; "
            f"got {value!r}"
        )
    return result


def _load_tasks(path: Path, task_mode: str, max_tasks: int) -> list[EpisodeTask]:
    raw_tasks: list[EpisodeTask] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            object_name = str(row.get("object") or "")
            if not object_name:
                raise ValueError(f"Missing object at {path}:{line_number}")
            scale = float(row.get("scale"))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Invalid scale at {path}:{line_number}: {row.get('scale')!r}")
            position = _finite_vector(row.get("position"), 3, "position", line_number)
            orientation = _finite_vector(row.get("orientation"), 4, "orientation", line_number)
            norm = float(np.linalg.norm(orientation))
            if norm <= 1.0e-12:
                raise ValueError(f"Zero orientation quaternion at {path}:{line_number}")
            orientation = orientation / norm
            raw_tasks.append(
                EpisodeTask(
                    index=len(raw_tasks),
                    source_line=line_number,
                    record=row,
                    object_name=object_name,
                    scale=scale,
                    position=position,
                    orientation=orientation,
                )
            )
    if not raw_tasks:
        raise ValueError(f"No episode tasks found in {path}")

    selected: list[EpisodeTask]
    if task_mode == "episode":
        selected = raw_tasks
    elif task_mode == "pose":
        selected = []
        seen: set[str] = set()
        for task in raw_tasks:
            key_payload = [
                task.object_name,
                task.scale,
                task.position.tolist(),
                task.orientation.tolist(),
            ]
            key = json.dumps(key_payload, separators=(",", ":"))
            if key not in seen:
                selected.append(task)
                seen.add(key)
    elif task_mode == "object":
        grouped: dict[str, list[EpisodeTask]] = {}
        for task in raw_tasks:
            grouped.setdefault(task.object_name, []).append(task)
        selected = []
        for object_tasks in grouped.values():
            initial = next(
                (
                    task
                    for task in object_tasks
                    if task.record.get("failure_retry_index") is None
                ),
                object_tasks[0],
            )
            record = dict(initial.record)
            record["_object_episode_aggregate"] = {
                "attempts": len(object_tasks),
                "success": any(bool(task.record.get("success")) for task in object_tasks),
                "source_lines": [task.source_line for task in object_tasks],
                "record_ids": [task.record.get("record_id") for task in object_tasks],
                "task_ids": [task.record.get("task_id") for task in object_tasks],
            }
            selected.append(
                EpisodeTask(
                    index=len(selected),
                    source_line=initial.source_line,
                    record=record,
                    object_name=initial.object_name,
                    scale=initial.scale,
                    position=initial.position,
                    orientation=initial.orientation,
                )
            )
    else:
        raise ValueError(f"Unsupported task mode {task_mode!r}")

    if max_tasks > 0:
        selected = selected[:max_tasks]
    return [
        EpisodeTask(
            index=index,
            source_line=task.source_line,
            record=task.record,
            object_name=task.object_name,
            scale=task.scale,
            position=task.position,
            orientation=task.orientation,
        )
        for index, task in enumerate(selected)
    ]


def _recorded_success(task: EpisodeTask) -> bool:
    aggregate = task.record.get("_object_episode_aggregate")
    if isinstance(aggregate, dict):
        return bool(aggregate.get("success"))
    return bool(task.record.get("success"))


def _source_record(task: EpisodeTask) -> dict[str, Any]:
    source = {
        "episodes_line": task.source_line,
        "record_id": task.record.get("record_id"),
        "task_id": task.record.get("task_id"),
        "failure_retry_index": task.record.get("failure_retry_index"),
    }
    aggregate = task.record.get("_object_episode_aggregate")
    if isinstance(aggregate, dict):
        source["object_episode_aggregate"] = aggregate
    return source


def _parse_backend(value: str, index: int) -> Backend:
    name = f"backend_{index}"
    endpoint = value.strip()
    if "=" in endpoint:
        name, endpoint = endpoint.split("=", 1)
        name = name.strip()
    if ":" not in endpoint:
        raise ValueError(f"Backend must be [NAME=]HOST:PORT, got {value!r}")
    host, port_text = endpoint.rsplit(":", 1)
    host = host.strip()
    if not name or not host:
        raise ValueError(f"Invalid backend {value!r}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid backend port in {value!r}")
    return Backend(name=name, host=host, port=port)


def _quat_wxyz_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _object_transform(task: EpisodeTask) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_matrix(task.orientation) * task.scale
    transform[:3, 3] = task.position
    return transform


def _load_pointcloud(path: Path, expected_points: int) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (expected_points, 3):
        raise ValueError(
            f"Point-cloud NPY must have shape ({expected_points}, 3), "
            f"got {points.shape}: {path}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"Point-cloud NPY contains non-finite values: {path}")
    return points


def _fake_grasps(count: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(max(count, 1), 16)
    grasps = []
    scores = []
    for index in range(count):
        yaw = 2.0 * math.pi * index / count
        rotation = np.asarray(
            [
                [-math.cos(yaw), -math.sin(yaw), 0.0],
                [-math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [0.04 * math.cos(yaw), 0.04 * math.sin(yaw), 0.11]
        grasps.append(matrix)
        scores.append(1.0 - index / count)
    return np.stack(grasps), np.asarray(scores, dtype=np.float64)


def _infer(client: Any, points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool]:
    remove_options = [bool(args.remove_outliers)]
    if args.remove_outliers:
        remove_options.append(False)
    errors: list[str] = []
    for remove_outliers in remove_options:
        for _ in range(int(args.max_tries)):
            try:
                grasps, scores = client.infer(
                    points,
                    grasp_threshold=float(args.grasp_threshold),
                    num_grasps=int(args.num_grasps),
                    topk_num_grasps=int(args.topk_num_grasps),
                    min_grasps=int(args.min_grasps),
                    max_tries=int(args.max_tries),
                    remove_outliers=remove_outliers,
                )
                grasps = np.asarray(grasps, dtype=np.float64)
                scores = np.asarray(scores, dtype=np.float64)
                if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or not len(grasps):
                    raise RuntimeError(f"invalid grasp array shape {grasps.shape}")
                if scores.shape != (len(grasps),):
                    raise RuntimeError(f"invalid score shape {scores.shape}")
                if not np.isfinite(grasps).all() or not np.isfinite(scores).all():
                    raise RuntimeError("GraspGen returned non-finite values")
                return grasps, scores, remove_outliers
            except Exception as exc:  # server failures are recorded per episode
                errors.append(f"{type(exc).__name__}: {exc}")
                time.sleep(0.25)
    raise RuntimeError("; ".join(errors[-3:]) or "GraspGen inference failed")


def _valid_se3(matrix: np.ndarray) -> tuple[bool, str | None]:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False, "not a finite 4x4 matrix"
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-4):
        return False, "invalid homogeneous last row"
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        return False, "rotation is not orthonormal"
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2.0e-3:
        return False, "rotation determinant is not +1"
    return True, None


def _collision_with_margin(
    manager_a: Any,
    manager_b: Any,
    margin: float,
) -> tuple[bool, float | None]:
    colliding = bool(manager_a.in_collision_other(manager_b))
    if colliding:
        return True, 0.0
    if margin <= 0.0:
        return False, None
    distance = float(manager_a.min_distance_other(manager_b))
    return colliding or distance < margin, distance


def _box_vertices_faces(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return corners @ transform[:3, :3].T + transform[:3, 3], faces


def _grasp_guide_meshes(
    transform: np.ndarray,
    *,
    width: float,
    depth: float,
    thickness: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match scripts/visualization/export_graspgen_grasp_viz_objs.py's guide."""
    transform = np.asarray(transform, dtype=np.float64)
    origin = transform[:3, 3]
    x_axis, y_axis, z_axis = (transform[:3, axis] for axis in range(3))
    half_width = 0.5 * width
    z0, z1 = -0.25 * depth, 0.75 * depth
    z_center = 0.5 * (z0 + z1)
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for side in (-1.0, 1.0):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 0] = x_axis * thickness
        matrix[:3, 1] = y_axis * thickness
        matrix[:3, 2] = z_axis * (z1 - z0)
        matrix[:3, 3] = origin + side * half_width * x_axis + z_center * z_axis
        meshes.append(_box_vertices_faces(matrix))
    connector = np.eye(4, dtype=np.float64)
    connector[:3, 0] = x_axis * (width + thickness)
    connector[:3, 1] = y_axis * thickness
    connector[:3, 2] = z_axis * thickness
    connector[:3, 3] = origin + z0 * z_axis
    meshes.append(_box_vertices_faces(connector))
    return meshes


def _append_obj_mesh(
    stream: Any,
    name: str,
    material: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_offset: int,
) -> int:
    stream.write(f"o {name}\nusemtl {material}\n")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    for vertex in vertices:
        stream.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
    for face in faces:
        indices = face + vertex_offset + 1
        stream.write("f " + " ".join(str(int(index)) for index in indices) + "\n")
    stream.write("\n")
    return vertex_offset + len(vertices)


def _write_viz_materials(path: Path, panda_hand_opacity: float) -> None:
    colors = {
        "ground": (0.58, 0.58, 0.58),
        "object": (0.82, 0.30, 0.25),
        "grasp_best": (0.10, 1.00, 0.10),
        "grasp_safe": (0.10, 0.48, 1.00),
        "panda_hand": (0.25, 0.85, 0.95),
    }
    with path.open("w", encoding="utf-8") as stream:
        for name, color in colors.items():
            stream.write(f"newmtl {name}\n")
            stream.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            stream.write("Ka 0.050000 0.050000 0.050000\n")
            stream.write("Ks 0.100000 0.100000 0.100000\n")
            if name == "panda_hand":
                # Write both common Wavefront transparency spellings. `d` is
                # opacity; `Tr` is transparency and therefore 1 - opacity.
                stream.write(f"d {panda_hand_opacity:.6f}\n")
                stream.write(f"Tr {1.0 - panda_hand_opacity:.6f}\n")
                stream.write("illum 2\n")
            stream.write("\n")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "object"


def _write_visualization_obj(
    path: Path,
    object_mesh: Any,
    object_transform: np.ndarray,
    ground_mesh: Any,
    ground_transform: np.ndarray,
    panda_hand_mesh: Any,
    safe_candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    mtl_path = path.with_suffix(".mtl")
    _write_viz_materials(mtl_path, float(args.viz_panda_hand_opacity))
    object_vertices = object_mesh.vertices @ object_transform[:3, :3].T + object_transform[:3, 3]
    ground_vertices = ground_mesh.vertices @ ground_transform[:3, :3].T + ground_transform[:3, 3]
    candidates = safe_candidates
    if int(args.viz_max_grasps) > 0:
        candidates = candidates[: int(args.viz_max_grasps)]
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        vertex_offset = 0
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(f"mtllib {mtl_path.name}\n\n")
            vertex_offset = _append_obj_mesh(
                stream, "ground", "ground", ground_vertices, ground_mesh.faces, vertex_offset
            )
            vertex_offset = _append_obj_mesh(
                stream, "object", "object", object_vertices, object_mesh.faces, vertex_offset
            )
            for candidate_index, candidate in enumerate(candidates):
                matrix = np.asarray(candidate["grasp_matrix_world"], dtype=np.float64)
                material = "grasp_best" if candidate_index == 0 else "grasp_safe"
                for part_index, (vertices, faces) in enumerate(
                    _grasp_guide_meshes(
                        matrix,
                        width=float(args.viz_grasp_width),
                        depth=float(args.viz_grasp_depth),
                        thickness=float(args.viz_grasp_thickness),
                    )
                ):
                    vertex_offset = _append_obj_mesh(
                        stream,
                        f"grasp_rank_{int(candidate['rank']):03d}_{part_index}",
                        material,
                        vertices,
                        faces,
                        vertex_offset,
                    )
                if args.viz_include_panda_hand:
                    hand_vertices = (
                        np.asarray(panda_hand_mesh.vertices, dtype=np.float64)
                        @ matrix[:3, :3].T
                        + matrix[:3, 3]
                    )
                    vertex_offset = _append_obj_mesh(
                        stream,
                        f"panda_hand_rank_{int(candidate['rank']):03d}",
                        "panda_hand",
                        hand_vertices,
                        panda_hand_mesh.faces,
                        vertex_offset,
                    )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EpisodeWorker:
    def __init__(self, backend: Backend, args: argparse.Namespace):
        import trimesh
        from trimesh.collision import CollisionManager

        self.backend = backend
        self.args = args
        self.trimesh = trimesh
        self.CollisionManager = CollisionManager
        self.client = None
        self.mesh_cache: OrderedDict[str, Any] = OrderedDict()
        self.point_cache: OrderedDict[str, tuple[np.ndarray, str]] = OrderedDict()
        hand_path = args.graspgen_root / "assets/panda_gripper/hand.stl"
        finger_path = args.graspgen_root / "assets/panda_gripper/finger.stl"
        self.hand_mesh = trimesh.load(str(hand_path), force="mesh", process=False)
        finger_left = trimesh.load(str(finger_path), force="mesh", process=False)
        finger_right = finger_left.copy()
        rotate_left = np.eye(4, dtype=np.float64)
        rotate_left[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        finger_left.apply_transform(rotate_left)
        finger_left.apply_translation([0.04, 0.0, 0.0584])
        finger_right.apply_translation([-0.04, 0.0, 0.0584])
        self.finger_mesh = trimesh.util.concatenate([finger_left, finger_right])
        self.hand_manager = CollisionManager()
        self.hand_manager.add_object("panda_hand", self.hand_mesh)
        self.finger_manager = CollisionManager()
        self.finger_manager.add_object("panda_fingers", self.finger_mesh)

    def connect(self) -> None:
        if self.args.dry_run_random_grasps:
            return
        from grasp_gen.serving.zmq_client import GraspGenClient

        self.client = GraspGenClient(
            host=self.backend.host,
            port=self.backend.port,
            timeout_ms=int(self.args.graspgen_timeout_ms),
            wait_for_server=False,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _mesh(self, task: EpisodeTask):
        mesh = self.mesh_cache.get(task.object_name)
        if mesh is None:
            path = self.args.mesh_dir / f"{task.object_name}.obj"
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh = self.trimesh.load(str(path), force="mesh", process=False)
            if isinstance(mesh, self.trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if not isinstance(mesh, self.trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError(f"Object OBJ is not a triangle mesh: {path}")
            self.mesh_cache[task.object_name] = mesh
            while len(self.mesh_cache) > int(self.args.object_cache_size):
                self.mesh_cache.popitem(last=False)
        else:
            self.mesh_cache.move_to_end(task.object_name)
        return mesh

    def _local_points(self, task: EpisodeTask, mesh: Any) -> tuple[np.ndarray, str]:
        cached = self.point_cache.get(task.object_name)
        if cached is not None:
            self.point_cache.move_to_end(task.object_name)
            return cached
        suffix = self.args.pointcloud_suffix or (
            f"_first_hit_fps_{int(self.args.pointcloud_points)}.npy"
        )
        path = self.args.pointcloud_dir / f"{task.object_name}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(path)
        points = _load_pointcloud(path, int(self.args.pointcloud_points))
        source = str(path)
        cached = (points, source)
        self.point_cache[task.object_name] = cached
        while len(self.point_cache) > int(self.args.object_cache_size):
            self.point_cache.popitem(last=False)
        return cached

    def run(self, task: EpisodeTask) -> dict[str, Any]:
        mesh = self._mesh(task)
        points_local, point_source = self._local_points(task, mesh)
        transform = _object_transform(task)
        points_world = self.trimesh.transform_points(points_local, transform)
        mean_world = points_world.mean(axis=0)
        points_centered = np.asarray(points_world - mean_world, dtype=np.float32)

        if self.args.dry_run_random_grasps:
            grasps, scores = _fake_grasps(int(self.args.topk_num_grasps))
            remove_outliers = False
        else:
            grasps, scores, remove_outliers = _infer(self.client, points_centered, self.args)
        grasps = grasps.copy()
        grasps[:, :3, 3] += mean_world.reshape(1, 3)

        # FCL transforms must be rigid. Apply scale to a mesh copy first rather
        # than placing a scaled rotation matrix in an FCL transform.
        collision_object_mesh = mesh.copy()
        collision_object_mesh.apply_scale(task.scale)
        object_rigid_transform = np.eye(4, dtype=np.float64)
        object_rigid_transform[:3, :3] = _quat_wxyz_matrix(task.orientation)
        object_rigid_transform[:3, 3] = task.position
        object_manager = self.CollisionManager()
        object_manager.add_object(
            "object",
            collision_object_mesh,
            transform=object_rigid_transform,
        )
        vertices_world = self.trimesh.transform_points(
            np.asarray(collision_object_mesh.vertices),
            object_rigid_transform,
        )
        support_z = float(vertices_world[:, 2].min() + self.args.ground_top_offset)
        ground_mesh = self.trimesh.creation.box(
            extents=[self.args.ground_size, self.args.ground_size, self.args.ground_thickness]
        )
        ground_transform = np.eye(4)
        ground_transform[:3, 3] = [
            float(task.position[0]),
            float(task.position[1]),
            support_z - 0.5 * float(self.args.ground_thickness),
        ]
        ground_manager = self.CollisionManager()
        ground_manager.add_object("ground", ground_mesh, transform=ground_transform)

        candidates: list[dict[str, Any]] = []
        order = np.argsort(-scores)
        for rank, source_index in enumerate(order):
            matrix = grasps[int(source_index)]
            valid, invalid_reason = _valid_se3(matrix)
            object_collision = ground_collision = True
            object_distance = ground_distance = None
            finger_object_collision = finger_ground_collision = True
            finger_object_distance = finger_ground_distance = None
            if valid:
                self.hand_manager.set_transform("panda_hand", matrix)
                self.finger_manager.set_transform("panda_fingers", matrix)
                object_collision, object_distance = _collision_with_margin(
                    object_manager, self.hand_manager, float(self.args.collision_margin)
                )
                ground_collision, ground_distance = _collision_with_margin(
                    ground_manager, self.hand_manager, float(self.args.collision_margin)
                )
                finger_object_collision, finger_object_distance = _collision_with_margin(
                    object_manager, self.finger_manager, float(self.args.collision_margin)
                )
                finger_ground_collision, finger_ground_distance = _collision_with_margin(
                    ground_manager, self.finger_manager, float(self.args.collision_margin)
                )
            collision_free = valid and not object_collision and not ground_collision
            hand_and_fingers_collision_free = (
                collision_free
                and not finger_object_collision
                and not finger_ground_collision
            )
            candidates.append(
                {
                    "rank": int(rank),
                    "source_index": int(source_index),
                    "confidence": float(scores[int(source_index)]),
                    "valid_se3": bool(valid),
                    "invalid_reason": invalid_reason,
                    "panda_hand_object_collision": bool(object_collision),
                    "panda_hand_ground_collision": bool(ground_collision),
                    "panda_hand_object_distance_m": object_distance,
                    "panda_hand_ground_distance_m": ground_distance,
                    "collision_free": bool(collision_free),
                    "panda_fingers_object_collision": bool(finger_object_collision),
                    "panda_fingers_ground_collision": bool(finger_ground_collision),
                    "panda_fingers_object_distance_m": finger_object_distance,
                    "panda_fingers_ground_distance_m": finger_ground_distance,
                    "hand_and_fingers_collision_free": bool(
                        hand_and_fingers_collision_free
                    ),
                    "grasp_matrix_world": matrix.astype(float).tolist(),
                }
            )

        safe = [candidate for candidate in candidates if candidate["collision_free"]]
        hand_and_fingers_safe = [
            candidate
            for candidate in candidates
            if candidate["hand_and_fingers_collision_free"]
        ]
        visualization_obj = None
        if self.args.export_viz_objs:
            visualization_path = self.args.viz_output_dir / (
                f"{task.index:06d}_{_safe_filename(task.object_name)}.obj"
            )
            _write_visualization_obj(
                visualization_path,
                collision_object_mesh,
                object_rigid_transform,
                ground_mesh,
                ground_transform,
                self.hand_mesh,
                safe,
                self.args,
            )
            visualization_obj = str(visualization_path)
        episode_success = _recorded_success(task)
        graspgen_feasible = bool(safe)
        graspgen_hand_and_fingers_feasible = bool(hand_and_fingers_safe)
        return {
            "schema_version": 1,
            "task_index": task.index,
            "source": _source_record(task),
            "object": task.object_name,
            "scale": task.scale,
            "position": task.position.astype(float).tolist(),
            "orientation_wxyz": task.orientation.astype(float).tolist(),
            "episode_success": episode_success,
            "graspgen_feasible": graspgen_feasible,
            "graspgen_hand_and_fingers_feasible": graspgen_hand_and_fingers_feasible,
            "consistent": episode_success == graspgen_feasible,
            "backend": {
                "name": self.backend.name,
                "host": self.backend.host,
                "port": self.backend.port,
                "metadata": self.backend.metadata,
            },
            "pointcloud": {
                "source": point_source,
                "points": int(len(points_centered)),
                "input_frame": "centered_at_world_pointcloud_mean",
                "world_mean": mean_world.astype(float).tolist(),
                "world_bounds": [
                    points_world.min(axis=0).astype(float).tolist(),
                    points_world.max(axis=0).astype(float).tolist(),
                ],
            },
            "collision": {
                "mesh": "GraspGen/assets/panda_gripper/hand.stl",
                "finger_mesh": "GraspGen/assets/panda_gripper/finger.stl",
                "support_top_z": support_z,
                "margin_m": float(self.args.collision_margin),
                "definition": "valid_se3_and_panda_hand_object_free_and_panda_hand_ground_free",
                "hand_and_fingers_definition": (
                    "valid_se3_and_panda_hand_and_both_fingers_object_free_"
                    "and_ground_free"
                ),
            },
            "graspgen": {
                "returned": len(candidates),
                "collision_free": len(safe),
                "hand_and_fingers_collision_free": len(hand_and_fingers_safe),
                "used_remove_outliers": bool(remove_outliers),
            },
            "candidates": candidates,
            "collision_free_candidate_ranks": [candidate["rank"] for candidate in safe],
            "hand_and_fingers_collision_free_candidate_ranks": [
                candidate["rank"] for candidate in hand_and_fingers_safe
            ],
            "visualization_obj": visualization_obj,
        }


def _error_result(task: EpisodeTask, backend: Backend | None, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_index": task.index,
        "source": _source_record(task),
        "object": task.object_name,
        "scale": task.scale,
        "position": task.position.astype(float).tolist(),
        "orientation_wxyz": task.orientation.astype(float).tolist(),
        "episode_success": _recorded_success(task),
        "graspgen_feasible": None,
        "graspgen_hand_and_fingers_feasible": None,
        "consistent": None,
        "backend": None
        if backend is None
        else {"name": backend.name, "host": backend.host, "port": backend.port},
        "error": f"{type(exc).__name__}: {exc}",
    }


def _probe_backends(backends: list[Backend], args: argparse.Namespace) -> list[Backend]:
    if args.dry_run_random_grasps:
        return backends
    from grasp_gen.serving.zmq_client import GraspGenClient

    available: list[Backend] = []
    failures: list[str] = []
    for backend in backends:
        client = None
        try:
            client = GraspGenClient(
                host=backend.host,
                port=backend.port,
                timeout_ms=int(args.graspgen_timeout_ms),
                wait_for_server=False,
            )
            metadata = dict(client.get_metadata() or {})
            gripper = str(metadata.get("gripper_name", ""))
            if gripper != "franka_panda" and not args.allow_non_franka_server:
                raise RuntimeError(f"expected franka_panda server, got {gripper!r}")
            available.append(Backend(backend.name, backend.host, backend.port, metadata))
        except Exception as exc:
            failures.append(f"{backend.name}={backend.host}:{backend.port}: {type(exc).__name__}: {exc}")
        finally:
            if client is not None:
                client.close()
    if failures:
        print("[WARNING] unavailable GraspGen backends:\n  " + "\n  ".join(failures), file=sys.stderr)
    if not available:
        raise RuntimeError("No usable GraspGen backends")
    return available


def _worker_loop(
    backend: Backend,
    args: argparse.Namespace,
    task_queue: queue.Queue[EpisodeTask | None],
    result_queue: queue.Queue[tuple[str, Any]],
) -> None:
    worker = None
    try:
        worker = EpisodeWorker(backend, args)
        worker.connect()
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break
            try:
                result = worker.run(task)
            except Exception as exc:
                result = _error_result(task, backend, exc)
            finally:
                task_queue.task_done()
            result_queue.put(("result", result))
    except BaseException as exc:
        result_queue.put(("worker_error", (backend, exc)))
    finally:
        if worker is not None:
            worker.close()
        result_queue.put(("worker_done", backend.name))


def _summary_update(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["processed"] += 1
    summary["objects_seen"].add(str(row.get("object", "")))
    if row.get("error"):
        summary["errors"] += 1
        return
    episode_success = bool(row["episode_success"])
    graspgen_feasible = bool(row["graspgen_feasible"])
    graspgen_stats = row.get("graspgen", {})
    summary["total_generated_grasps"] += int(graspgen_stats.get("returned", 0))
    summary["total_collision_free_grasps"] += int(
        graspgen_stats.get("collision_free", 0)
    )
    summary["total_hand_and_fingers_collision_free_grasps"] += int(
        graspgen_stats.get("hand_and_fingers_collision_free", 0)
    )
    if graspgen_feasible:
        summary["objects_with_collision_free_grasp"].add(str(row.get("object", "")))
    if bool(row.get("graspgen_hand_and_fingers_feasible")):
        summary["objects_with_hand_and_fingers_collision_free_grasp"].add(
            str(row.get("object", ""))
        )
    summary["episode_success"] += int(episode_success)
    summary["graspgen_feasible"] += int(graspgen_feasible)
    summary["consistent"] += int(episode_success == graspgen_feasible)
    summary["both_success"] += int(episode_success and graspgen_feasible)
    summary["episode_only"] += int(episode_success and not graspgen_feasible)
    summary["graspgen_only"] += int(not episode_success and graspgen_feasible)
    summary["both_failure"] += int(not episode_success and not graspgen_feasible)


def main() -> int:
    args = _parse_args()
    if args.pointcloud_points <= 0 or args.max_tries <= 0 or args.object_cache_size <= 0:
        raise ValueError(
            "--pointcloud-points, --max-tries, and --object-cache-size must be positive"
        )
    if args.max_tasks < 0:
        raise ValueError("--max-tasks must be non-negative")
    if args.ground_size <= 0 or args.ground_thickness <= 0 or args.collision_margin < 0:
        raise ValueError("Ground dimensions must be positive and collision margin non-negative")
    if args.viz_max_grasps < 0:
        raise ValueError("--viz-max-grasps must be non-negative")
    if min(args.viz_grasp_width, args.viz_grasp_depth, args.viz_grasp_thickness) <= 0:
        raise ValueError("Visualization grasp dimensions must be positive")
    if not 0.0 <= args.viz_panda_hand_opacity <= 1.0:
        raise ValueError("--viz-panda-hand-opacity must be in [0, 1]")
    if args.viz_include_panda_hand:
        args.export_viz_objs = True

    args.episodes_jsonl = args.episodes_jsonl.expanduser().resolve()
    args.output_jsonl = args.output_jsonl.expanduser().resolve()
    args.mesh_dir = args.mesh_dir.expanduser().resolve()
    args.graspgen_root = args.graspgen_root.expanduser().resolve()
    args.pointcloud_dir = args.pointcloud_dir.expanduser().resolve()
    if args.viz_output_dir is None:
        args.viz_output_dir = args.output_jsonl.with_suffix("").with_name(
            f"{args.output_jsonl.stem}_viz"
        )
    else:
        args.viz_output_dir = args.viz_output_dir.expanduser().resolve()
    if not args.episodes_jsonl.is_file():
        raise FileNotFoundError(args.episodes_jsonl)
    if not args.mesh_dir.is_dir():
        raise FileNotFoundError(args.mesh_dir)
    if not args.graspgen_root.is_dir():
        raise FileNotFoundError(args.graspgen_root)
    if not args.pointcloud_dir.is_dir():
        raise FileNotFoundError(args.pointcloud_dir)
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_jsonl}")

    sys.path.insert(0, str(args.graspgen_root))
    tasks = _load_tasks(
        args.episodes_jsonl,
        task_mode=str(args.task_mode),
        max_tasks=int(args.max_tasks),
    )
    backend_values = args.backend or ["localhost:5556"]
    parsed = [_parse_backend(value, index) for index, value in enumerate(backend_values)]
    names = [backend.name for backend in parsed]
    endpoints = [(backend.host, backend.port) for backend in parsed]
    if len(names) != len(set(names)) or len(endpoints) != len(set(endpoints)):
        raise ValueError("Backend names and HOST:PORT endpoints must be unique")
    backends = _probe_backends(parsed, args)

    if args.export_viz_objs:
        args.viz_output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[START] tasks={len(tasks)} task_mode={args.task_mode} backends={len(backends)} "
        f"episodes={args.episodes_jsonl} output={args.output_jsonl}",
        flush=True,
    )
    task_queue: queue.Queue[EpisodeTask | None] = queue.Queue()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in backends:
        task_queue.put(None)

    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(backend, args, task_queue, result_queue),
            name=f"graspgen-{backend.name}",
            daemon=True,
        )
        for backend in backends
    ]
    for thread in threads:
        thread.start()

    from tqdm import tqdm

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(f".{args.output_jsonl.name}.tmp.{os.getpid()}")
    summary: dict[str, Any] = {
        key: 0
        for key in (
            "processed",
            "errors",
            "episode_success",
            "graspgen_feasible",
            "consistent",
            "both_success",
            "episode_only",
            "graspgen_only",
            "both_failure",
            "total_generated_grasps",
            "total_collision_free_grasps",
            "total_hand_and_fingers_collision_free_grasps",
        )
    }
    summary["objects_seen"] = set()
    summary["objects_with_collision_free_grasp"] = set()
    summary["objects_with_hand_and_fingers_collision_free_grasp"] = set()
    worker_done = 0
    received: set[int] = set()
    worker_errors: list[str] = []
    try:
        with temporary.open("w", encoding="utf-8") as stream, tqdm(
            total=len(tasks),
            desc="episodes -> GraspGen",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            while len(received) < len(tasks):
                try:
                    kind, payload = result_queue.get(timeout=1.0)
                except queue.Empty:
                    if worker_done == len(backends):
                        break
                    continue
                if kind == "result":
                    row = payload
                    task_index = int(row["task_index"])
                    if task_index in received:
                        continue
                    received.add(task_index)
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    _summary_update(summary, row)
                    progress.update(1)
                    progress.set_postfix(
                        safe=summary["graspgen_feasible"],
                        agree=summary["consistent"],
                        errors=summary["errors"],
                        refresh=False,
                    )
                elif kind == "worker_error":
                    backend, exc = payload
                    worker_errors.append(
                        f"{backend.name}: {type(exc).__name__}: {exc}"
                    )
                elif kind == "worker_done":
                    worker_done += 1

            if len(received) != len(tasks):
                missing = [task for task in tasks if task.index not in received]
                reason = RuntimeError(
                    "All backend workers exited before these tasks completed; "
                    + "; ".join(worker_errors)
                )
                for task in missing:
                    row = _error_result(task, None, reason)
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    _summary_update(summary, row)
                    progress.update(1)
        os.replace(temporary, args.output_jsonl)
    finally:
        temporary.unlink(missing_ok=True)
        for thread in threads:
            thread.join(timeout=5.0)

    valid = summary["processed"] - summary["errors"]
    agreement = summary["consistent"] / valid if valid else float("nan")
    print("\n========== GraspGen / Episode Consistency ==========")
    print(f"tasks:                    {summary['processed']}")
    print(f"errors:                   {summary['errors']}")
    print(f"episode success:          {summary['episode_success']}")
    print(f"GraspGen feasible:        {summary['graspgen_feasible']}")
    print(f"both success:             {summary['both_success']}")
    print(f"episode only:             {summary['episode_only']}")
    print(f"GraspGen only:            {summary['graspgen_only']}")
    print(f"both failure:             {summary['both_failure']}")
    print(f"agreement:                {summary['consistent']}/{valid} ({agreement:.2%})")
    print("----------------------------------------------------")
    print(f"unique objects processed: {len(summary['objects_seen'])}")
    print(
        "objects with >=1 collision-free grasp: "
        f"{len(summary['objects_with_collision_free_grasp'])}"
    )
    print(f"total generated grasps:   {summary['total_generated_grasps']}")
    print(f"total collision-free:     {summary['total_collision_free_grasps']}")
    print(
        "objects with >=1 hand+fingers collision-free grasp: "
        f"{len(summary['objects_with_hand_and_fingers_collision_free_grasp'])}"
    )
    print(
        "total hand+fingers collision-free grasps:           "
        f"{summary['total_hand_and_fingers_collision_free_grasps']}"
    )
    print(f"output:                   {args.output_jsonl}")
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

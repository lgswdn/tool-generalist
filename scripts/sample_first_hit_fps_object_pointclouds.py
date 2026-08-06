#!/usr/bin/env python3
"""Export COACD exterior point clouds as PLY and NPY using first-hit rays and FPS."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MESH_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")
DEFAULT_OUTPUT_DIR = Path("scripts/outputs/first_hit_fps_object_pointcloud_samples")

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_VIEW_DIRECTIONS: np.ndarray | None = None
_WORKER_TRIMESH = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cast orthographic rays from an icosphere, retain only the first hit on the raw "
            "COACD triangle set, voxel-deduplicate the hits, and create nested FPS clouds "
            "in both PLY and NPY formats."
        )
    )
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument(
        "--objects-manifest",
        type=Path,
        default=None,
        help=(
            "JSON list of object identifiers. A trailing scale suffix such as '-0.060' is "
            "removed so output filenames and metadata match canonical OBJ stems."
        ),
    )
    parser.add_argument("--name-contains", default="bowl")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=5,
        help="Maximum selected objects; 0 means all matched objects.",
    )
    parser.add_argument(
        "--expected-num-objects",
        type=int,
        default=0,
        help="Fail before generation unless exactly this many objects are selected.",
    )
    parser.add_argument("--num-points", type=int, nargs="+", default=[512, 2048])
    parser.add_argument(
        "--view-subdivisions",
        type=int,
        default=1,
        help="Icosphere subdivisions: 1 gives 42 views, 2 gives 162 views.",
    )
    parser.add_argument("--ray-resolution", type=int, default=128)
    parser.add_argument(
        "--voxel-size-ratio",
        type=float,
        default=0.002,
        help="Voxel side length divided by the canonical mesh AABB diagonal.",
    )
    parser.add_argument("--ray-engine", choices=("auto", "embree", "triangle"), default="auto")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of persistent object-level worker processes.",
    )
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip objects with matching per-object metadata and complete output files.",
    )
    return parser.parse_args()


def _resolve_object(mesh_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        filename = path.name if path.suffix == ".obj" else f"{path.name}.obj"
        path = mesh_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Object OBJ not found: {value}")
    return path.resolve()


def _jsonl_object_names(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            name = row.get("object_name")
            if name is None and isinstance(row.get("record"), dict):
                name = row["record"].get("object_name")
            if name is not None and str(name) not in seen:
                names.append(str(name))
                seen.add(str(name))
    return names


def _strip_manifest_scale_suffix(value: str) -> str:
    """Resolve a full manifest identifier to its unscaled canonical OBJ stem."""
    value = str(value).strip()
    if value.endswith(".obj"):
        return value
    match = re.fullmatch(r"(.+)-([0-9]+(?:\.[0-9]+)?)", value)
    return match.group(1) if match is not None else value


def _manifest_object_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of objects: {path}")

    names: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(data):
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = item.get("object_name") or item.get("name") or item.get("object")
            if value is None:
                raise ValueError(f"Manifest object {path}[{index}] has no object name")
        else:
            raise ValueError(f"Unsupported manifest object at {path}[{index}]: {type(item).__name__}")
        name = _strip_manifest_scale_suffix(str(value))
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _select_objects(args: argparse.Namespace) -> list[tuple[str, Path]]:
    mesh_dir = args.mesh_dir.expanduser().resolve()
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    source_count = sum(
        source is not None for source in (args.objects, args.source_jsonl, args.objects_manifest)
    )
    if source_count > 1:
        raise ValueError("Use only one of --objects, --source-jsonl, or --objects-manifest")

    if args.objects:
        selected = [
            (Path(value).stem, _resolve_object(mesh_dir, value))
            for value in args.objects
        ]
    else:
        from_manifest = args.objects_manifest is not None
        if args.source_jsonl is not None:
            names = _jsonl_object_names(args.source_jsonl.expanduser().resolve())
        elif from_manifest:
            names = _manifest_object_names(args.objects_manifest.expanduser().resolve())
        else:
            names = sorted(path.stem for path in mesh_dir.glob("*.obj"))
        substring = str(args.name_contains).lower()
        if substring:
            names = [name for name in names if substring in name.lower()]
        if args.num_objects < 0:
            raise ValueError("--num-objects must be non-negative")
        if args.num_objects:
            names = names[: int(args.num_objects)]
        if not names:
            raise ValueError(f"No objects matched name filter {args.name_contains!r}")
        selected = [
            (
                name,
                _resolve_object(
                    mesh_dir,
                    _strip_manifest_scale_suffix(name) if from_manifest else name,
                ),
            )
            for name in names
        ]

    if args.expected_num_objects and len(selected) != int(args.expected_num_objects):
        raise ValueError(f"Expected {args.expected_num_objects} objects, selected {len(selected)}")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_ray_intersector(mesh, engine: str):
    if engine in {"auto", "embree"}:
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            return RayMeshIntersector(mesh), "embree"
        except Exception:
            if engine == "embree":
                raise
    from trimesh.ray.ray_triangle import RayMeshIntersector

    return RayMeshIntersector(mesh), "triangle"


def _camera_basis(view_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(reference, view_direction))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(reference, view_direction)
    right /= np.linalg.norm(right)
    up = np.cross(view_direction, right)
    up /= np.linalg.norm(up)
    return right, up


def _first_hit_candidates(mesh, intersector, view_directions: np.ndarray, resolution: int):
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max())
    half_extent = radius * 1.02
    camera_distance = radius * 3.0
    pixel = ((np.arange(resolution, dtype=np.float64) + 0.5) / resolution * 2.0 - 1.0) * half_extent
    grid_u, grid_v = np.meshgrid(pixel, pixel, indexing="xy")
    grid_u = grid_u.reshape(-1, 1)
    grid_v = grid_v.reshape(-1, 1)

    hits: list[np.ndarray] = []
    hit_counts: list[int] = []
    for view_direction in view_directions:
        view_direction = np.asarray(view_direction, dtype=np.float64)
        view_direction /= np.linalg.norm(view_direction)
        right, up = _camera_basis(view_direction)
        camera_center = center + camera_distance * view_direction
        origins = camera_center + grid_u * right.reshape(1, 3) + grid_v * up.reshape(1, 3)
        directions = np.repeat((-view_direction).reshape(1, 3), origins.shape[0], axis=0)
        locations, _, _ = intersector.intersects_location(origins, directions, multiple_hits=False)
        locations = np.asarray(locations, dtype=np.float64)
        hits.append(locations)
        hit_counts.append(int(len(locations)))
    if not hits or not any(len(hit) for hit in hits):
        raise ValueError("Ray casting produced no first-hit points")
    return np.concatenate(hits, axis=0), hit_counts


def _voxel_mean(points: np.ndarray, voxel_size: float) -> np.ndarray:
    origin = points.min(axis=0)
    voxel_indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts.reshape(-1, 1)


def _fps_with_extent_anchors(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"FPS needs {count} points but only {len(points)} candidates are available")

    anchor_indices: list[int] = []
    for axis in range(3):
        anchor_indices.extend((int(np.argmin(points[:, axis])), int(np.argmax(points[:, axis]))))
    selected: list[int] = []
    selected_set: set[int] = set()
    for index in anchor_indices:
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    min_dist_sq = np.full(len(points), np.inf, dtype=np.float64)
    for index in selected:
        delta = points - points[index]
        min_dist_sq = np.minimum(min_dist_sq, np.einsum("ij,ij->i", delta, delta))
    min_dist_sq[np.asarray(selected, dtype=np.int64)] = -1.0

    while len(selected) < count:
        index = int(np.argmax(min_dist_sq))
        selected.append(index)
        delta = points - points[index]
        min_dist_sq = np.minimum(min_dist_sq, np.einsum("ij,ij->i", delta, delta))
        min_dist_sq[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def _nearest_neighbor_stats(points: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=2)
    nearest = distances[:, 1]
    return {
        "min": float(nearest.min()),
        "median": float(np.median(nearest)),
        "mean": float(nearest.mean()),
        "p95": float(np.quantile(nearest, 0.95)),
        "max": float(nearest.max()),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_export_pointcloud(points: np.ndarray, path: Path) -> None:
    if _WORKER_TRIMESH is None:
        raise RuntimeError("Point-cloud worker was not initialized")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        payload = _WORKER_TRIMESH.PointCloud(points).export(file_type="ply")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        with temporary.open("wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npy(points: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, points, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_VIEW_DIRECTIONS, _WORKER_TRIMESH
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required; run with the GraspGen environment") from exc
    _WORKER_CONFIG = config
    _WORKER_TRIMESH = trimesh
    _WORKER_VIEW_DIRECTIONS = np.asarray(
        trimesh.creation.icosphere(
            subdivisions=int(config["view_subdivisions"]),
            radius=1.0,
        ).vertices,
        dtype=np.float64,
    )


def _load_resumable_metadata(
    object_id: str,
    obj_path: Path,
    metadata_path: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not config["resume"] or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        expected_counts = {str(int(count)) for count in config["num_points"]}
        pointclouds = row.get("pointclouds", {})
        numpy_pointclouds = row.get("numpy_pointclouds", {})
        matches = (
            row.get("method") == "multiview_first_hit_voxel_fps"
            and row.get("object_name") == object_id
            and row.get("mesh_name") == obj_path.stem
            and Path(row.get("source_obj", "")).resolve() == obj_path.resolve()
            and int(row.get("view_subdivisions", -1)) == int(config["view_subdivisions"])
            and int(row.get("ray_resolution", -1)) == int(config["ray_resolution"])
            and np.isclose(float(row.get("voxel_size_ratio", -1.0)), float(config["voxel_size_ratio"]))
            and np.isclose(float(row.get("scale", -1.0)), float(config["scale"]))
            and set(pointclouds) == expected_counts
            and set(numpy_pointclouds) == expected_counts
            and all(Path(pointclouds[count]).is_file() for count in expected_counts)
            and all(Path(numpy_pointclouds[count]).is_file() for count in expected_counts)
        )
        if not config["no_save_candidates"]:
            candidate_path = row.get("candidate_pointcloud")
            matches = matches and bool(candidate_path) and Path(candidate_path).is_file()
        return row if matches else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _process_object(
    object_index: int,
    object_id: str,
    obj_path_value: str,
) -> tuple[int, dict[str, Any]]:
    if _WORKER_CONFIG is None or _WORKER_VIEW_DIRECTIONS is None or _WORKER_TRIMESH is None:
        raise RuntimeError("Point-cloud worker was not initialized")
    config = _WORKER_CONFIG
    trimesh = _WORKER_TRIMESH
    obj_path = Path(obj_path_value)

    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"OBJ did not load as a triangle mesh: {obj_path}")

    intersector, resolved_ray_engine = _make_ray_intersector(mesh, str(config["ray_engine"]))
    raw_hits, hit_counts = _first_hit_candidates(
        mesh,
        intersector,
        _WORKER_VIEW_DIRECTIONS,
        int(config["ray_resolution"]),
    )
    bbox_diagonal = float(np.linalg.norm(np.asarray(mesh.extents, dtype=np.float64)))
    if not np.isfinite(bbox_diagonal) or bbox_diagonal <= 0.0:
        raise ValueError(f"Invalid mesh AABB diagonal {bbox_diagonal}: {obj_path}")
    voxel_size = bbox_diagonal * float(config["voxel_size_ratio"])
    candidates = _voxel_mean(raw_hits, voxel_size)
    max_point_count = max(int(count) for count in config["num_points"])
    fps_indices = _fps_with_extent_anchors(candidates, max_point_count)
    fps_points = candidates[fps_indices]

    ply_pointcloud_dir = Path(config["ply_pointcloud_dir"])
    numpy_pointcloud_dir = Path(config["numpy_pointcloud_dir"])
    candidate_dir = Path(config["candidate_dir"])
    replace_existing = bool(config["overwrite"] or config["resume"])
    outputs: dict[str, str] = {}
    numpy_outputs: dict[str, str] = {}
    nn_stats: dict[str, dict[str, float]] = {}
    for point_count in config["num_points"]:
        point_count = int(point_count)
        points = np.asarray(fps_points[:point_count] * float(config["scale"]), dtype=np.float32)
        output_path = ply_pointcloud_dir / f"{object_id}_first_hit_fps_{point_count}.ply"
        numpy_output_path = numpy_pointcloud_dir / f"{object_id}_first_hit_fps_{point_count}.npy"
        existing_paths = [path for path in (output_path, numpy_output_path) if path.exists()]
        if existing_paths and not replace_existing:
            existing_text = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                f"Output already exists; pass --overwrite or --resume: {existing_text}"
            )
        _atomic_export_pointcloud(points, output_path)
        _atomic_save_npy(points, numpy_output_path)
        outputs[str(point_count)] = str(output_path)
        numpy_outputs[str(point_count)] = str(numpy_output_path)
        nn_stats[str(point_count)] = _nearest_neighbor_stats(points)

    candidate_path = None
    if not config["no_save_candidates"]:
        candidate_path = candidate_dir / f"{object_id}_first_hit_voxel.ply"
        if candidate_path.exists() and not replace_existing:
            raise FileExistsError(f"Output already exists; pass --overwrite or --resume: {candidate_path}")
        candidate_points = np.asarray(candidates * float(config["scale"]), dtype=np.float32)
        _atomic_export_pointcloud(candidate_points, candidate_path)

    row = {
        "object_name": object_id,
        "mesh_name": obj_path.stem,
        "source_obj": str(obj_path),
        "source_sha256": _sha256(obj_path),
        "method": "multiview_first_hit_voxel_fps",
        "scale": float(config["scale"]),
        "source_components": int(len(mesh.split(only_watertight=False))),
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "view_subdivisions": int(config["view_subdivisions"]),
        "views": int(len(_WORKER_VIEW_DIRECTIONS)),
        "ray_resolution": int(config["ray_resolution"]),
        "rays_per_view": int(config["ray_resolution"]) ** 2,
        "ray_engine": resolved_ray_engine,
        "raw_first_hits": int(len(raw_hits)),
        "per_view_hit_counts": hit_counts,
        "bbox_diagonal": bbox_diagonal,
        "voxel_size_ratio": float(config["voxel_size_ratio"]),
        "voxel_size": voxel_size,
        "voxel_candidates": int(len(candidates)),
        "candidate_pointcloud": None if candidate_path is None else str(candidate_path),
        "fps_extent_anchors": True,
        "fps_deterministic": True,
        "pointclouds": outputs,
        "numpy_pointclouds": numpy_outputs,
        "nearest_neighbor_distance": nn_stats,
    }
    return object_index, row


def _validate_args(args: argparse.Namespace) -> None:
    if any(count <= 0 for count in args.num_points):
        raise ValueError("Every --num-points value must be positive")
    if args.view_subdivisions < 0:
        raise ValueError("--view-subdivisions must be non-negative")
    if args.ray_resolution <= 0:
        raise ValueError("--ray-resolution must be positive")
    if args.voxel_size_ratio <= 0.0:
        raise ValueError("--voxel-size-ratio must be positive")
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.expected_num_objects < 0:
        raise ValueError("--expected-num-objects must be non-negative")
    if args.overwrite and args.resume:
        raise ValueError("Use only one of --overwrite or --resume")


def main() -> int:
    args = _parse_args()
    args.num_points = list(dict.fromkeys(int(count) for count in args.num_points))
    _validate_args(args)

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for the total progress bar") from exc

    output_dir = args.output_dir.expanduser().resolve()
    ply_pointcloud_dir = output_dir / "ply"
    numpy_pointcloud_dir = output_dir / "npy"
    candidate_dir = output_dir / "voxel_candidates"
    metadata_dir = output_dir / "metadata"
    ply_pointcloud_dir.mkdir(parents=True, exist_ok=True)
    numpy_pointcloud_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    selected_objects = _select_objects(args)
    config: dict[str, Any] = {
        "ply_pointcloud_dir": str(ply_pointcloud_dir),
        "numpy_pointcloud_dir": str(numpy_pointcloud_dir),
        "candidate_dir": str(candidate_dir),
        "num_points": tuple(args.num_points),
        "view_subdivisions": int(args.view_subdivisions),
        "ray_resolution": int(args.ray_resolution),
        "voxel_size_ratio": float(args.voxel_size_ratio),
        "ray_engine": str(args.ray_engine),
        "scale": float(args.scale),
        "seed": int(args.seed),
        "no_save_candidates": bool(args.no_save_candidates),
        "overwrite": bool(args.overwrite),
        "resume": bool(args.resume),
    }

    metadata_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, str, Path]] = []
    for object_index, (object_id, obj_path) in enumerate(selected_objects):
        metadata_path = metadata_dir / f"{object_id}.json"
        existing = _load_resumable_metadata(object_id, obj_path, metadata_path, config)
        if existing is None:
            pending.append((object_index, object_id, obj_path))
        else:
            metadata_by_index[object_index] = existing

    total = len(selected_objects)
    resumed = total - len(pending)
    print(
        f"selected={total} pending={len(pending)} resumed={resumed} workers={args.workers} "
        f"points={args.num_points} output={output_dir}",
        flush=True,
    )

    failures: list[dict[str, Any]] = []
    completed_this_run = 0
    progress = tqdm(
        total=total,
        initial=resumed,
        desc="first-hit FPS",
        unit="object",
        dynamic_ncols=True,
    )

    def record_success(object_index: int, row: dict[str, Any]) -> None:
        nonlocal completed_this_run
        metadata_by_index[object_index] = row
        _atomic_write_json(metadata_dir / f"{row['object_name']}.json", row)
        completed_this_run += 1
        progress.update(1)
        progress.set_postfix(
            success=len(metadata_by_index),
            failed=len(failures),
            latest=row["object_name"],
            refresh=False,
        )

    if args.workers == 1:
        _init_worker(config)
        for object_index, object_id, obj_path in pending:
            try:
                result_index, row = _process_object(object_index, object_id, str(obj_path))
                record_success(result_index, row)
            except Exception as exc:
                failures.append(
                    {
                        "object_index": object_index,
                        "object_name": object_id,
                        "mesh_name": obj_path.stem,
                        "source_obj": str(obj_path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                progress.update(1)
                progress.set_postfix(success=len(metadata_by_index), failed=len(failures), refresh=False)
                progress.write(f"[ERROR] object={object_id}: {type(exc).__name__}: {exc}")
    elif pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=context,
            initializer=_init_worker,
            initargs=(config,),
        ) as executor:
            future_to_object = {
                executor.submit(_process_object, object_index, object_id, str(obj_path)): (
                    object_index,
                    object_id,
                    obj_path,
                )
                for object_index, object_id, obj_path in pending
            }
            for future in concurrent.futures.as_completed(future_to_object):
                object_index, object_id, obj_path = future_to_object[future]
                try:
                    result_index, row = future.result()
                    record_success(result_index, row)
                except Exception as exc:
                    failures.append(
                        {
                            "object_index": object_index,
                            "object_name": object_id,
                            "mesh_name": obj_path.stem,
                            "source_obj": str(obj_path),
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    progress.update(1)
                    progress.set_postfix(
                        success=len(metadata_by_index),
                        failed=len(failures),
                        refresh=False,
                    )
                    progress.write(f"[ERROR] object={object_id}: {type(exc).__name__}: {exc}")

    progress.close()

    metadata = [metadata_by_index[index] for index in sorted(metadata_by_index)]
    metadata_path = output_dir / "samples.json"
    failures_path = output_dir / "failures.json"
    _atomic_write_json(metadata_path, metadata)
    _atomic_write_json(failures_path, failures)
    print(
        f"wrote metadata={metadata_path} successful={len(metadata)} failed={len(failures)}",
        flush=True,
    )
    if failures:
        print(f"failure details={failures_path}; rerun with --resume after addressing them", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

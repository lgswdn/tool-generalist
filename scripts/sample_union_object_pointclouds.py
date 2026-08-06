#!/usr/bin/env python3
"""Boolean-union COACD components and export test point-cloud PLY files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_MESH_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")
DEFAULT_OUTPUT_DIR = Path("scripts/outputs/union_object_pointcloud_samples")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a COACD OBJ into connected convex components, boolean-union the components, "
            "and sample only the resulting union boundary. This script does not change runtime APIs."
        )
    )
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument("--name-contains", default="bowl")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-points", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--engine", default="manifold")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
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
            if name is not None and str(name) not in names:
                names.append(str(name))
    return names


def _select_objects(args: argparse.Namespace) -> list[Path]:
    mesh_dir = args.mesh_dir.expanduser().resolve()
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    if args.objects:
        return [_resolve_object(mesh_dir, value) for value in args.objects]

    if args.source_jsonl is not None:
        names = _jsonl_object_names(args.source_jsonl.expanduser().resolve())
    else:
        names = sorted(path.stem for path in mesh_dir.glob("*.obj"))
    substring = str(args.name_contains).lower()
    if substring:
        names = [name for name in names if substring in name.lower()]
    if args.num_objects <= 0:
        raise ValueError("--num-objects must be positive")
    names = names[: int(args.num_objects)]
    if not names:
        raise ValueError(f"No objects matched name filter {args.name_contains!r}")
    return [_resolve_object(mesh_dir, name) for name in names]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_components(mesh) -> list:
    components = list(mesh.split(only_watertight=False))
    if not components:
        raise ValueError("Mesh has no connected components")
    prepared = []
    for component_index, component in enumerate(components):
        component = component.copy()
        if not component.is_watertight:
            raise ValueError(f"COACD component {component_index} is not watertight")
        if not component.is_winding_consistent:
            component.fix_normals()
        if float(component.volume) < 0.0:
            component.invert()
        if float(component.volume) <= 0.0:
            raise ValueError(f"COACD component {component_index} has non-positive volume")
        prepared.append(component)
    return prepared


def _as_single_mesh(value, trimesh):
    if isinstance(value, trimesh.Scene):
        value = value.dump(concatenate=True)
    if not isinstance(value, trimesh.Trimesh):
        raise TypeError(f"Boolean union returned unsupported type: {type(value).__name__}")
    return value


def main() -> int:
    args = _parse_args()
    if any(count <= 0 for count in args.num_points):
        raise ValueError("Every --num-points value must be positive")
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required; run with the Isaac conda environment") from exc

    output_dir = args.output_dir.expanduser().resolve()
    union_mesh_dir = output_dir / "union_meshes"
    pointcloud_dir = output_dir / "pointclouds"
    union_mesh_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    for object_index, obj_path in enumerate(_select_objects(args)):
        raw_mesh = _as_single_mesh(
            trimesh.load(str(obj_path), force="mesh", process=False),
            trimesh,
        )
        components = _prepare_components(raw_mesh)
        union_mesh = trimesh.boolean.union(
            components,
            engine=str(args.engine),
            check_volume=True,
        )
        if union_mesh is None:
            raise RuntimeError(f"Boolean union returned no mesh: {obj_path}")
        union_mesh = _as_single_mesh(union_mesh, trimesh)
        if len(union_mesh.faces) == 0 or float(union_mesh.volume) <= 0.0:
            raise RuntimeError(f"Boolean union produced an invalid solid: {obj_path}")
        if not union_mesh.is_watertight:
            print(
                f"[WARNING] union mesh is not watertight object={obj_path.stem}; "
                "exporting it for visual inspection",
                flush=True,
            )

        bounds_error = float(np.max(np.abs(np.asarray(raw_mesh.bounds) - np.asarray(union_mesh.bounds))))
        if bounds_error > 1.0e-5:
            raise RuntimeError(f"Boolean union changed bounds by {bounds_error:.6g}: {obj_path}")

        export_mesh = union_mesh.copy()
        export_mesh.apply_scale(float(args.scale))
        union_mesh_path = union_mesh_dir / f"{obj_path.stem}_union.ply"
        if union_mesh_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {union_mesh_path}")
        export_mesh.export(union_mesh_path)

        outputs: dict[str, str] = {}
        for point_count_index, point_count in enumerate(args.num_points):
            sample_seed = int(args.seed) + 1009 * object_index + 17 * point_count_index
            rng = np.random.default_rng(sample_seed)
            points, _ = trimesh.sample.sample_surface(union_mesh, int(point_count), seed=rng)
            points = np.asarray(points, dtype=np.float32) * np.float32(args.scale)
            pointcloud_path = pointcloud_dir / f"{obj_path.stem}_union_{int(point_count)}.ply"
            if pointcloud_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists; pass --overwrite: {pointcloud_path}")
            trimesh.PointCloud(points).export(pointcloud_path)
            outputs[str(int(point_count))] = str(pointcloud_path)

        union_components = list(union_mesh.split(only_watertight=False))
        row = {
            "object_name": obj_path.stem,
            "source_obj": str(obj_path),
            "source_sha256": _sha256(obj_path),
            "method": "coacd_boolean_union_then_area_sample",
            "boolean_engine": str(args.engine),
            "scale": float(args.scale),
            "input_components": int(len(components)),
            "input_vertices": int(len(raw_mesh.vertices)),
            "input_faces": int(len(raw_mesh.faces)),
            "input_volume": float(sum(float(component.volume) for component in components)),
            "union_components": int(len(union_components)),
            "union_vertices": int(len(union_mesh.vertices)),
            "union_faces": int(len(union_mesh.faces)),
            "union_volume": float(union_mesh.volume),
            "union_watertight": bool(union_mesh.is_watertight),
            "bounds_error": bounds_error,
            "bounds_after_scale": (np.asarray(union_mesh.bounds) * float(args.scale)).tolist(),
            "union_mesh": str(union_mesh_path),
            "pointclouds": outputs,
        }
        metadata.append(row)
        print(
            f"union-sampled object={obj_path.stem} input_components={len(components)} "
            f"union_components={len(union_components)} input_faces={len(raw_mesh.faces)} "
            f"union_faces={len(union_mesh.faces)} outputs={len(outputs)}",
            flush=True,
        )

    metadata_path = output_dir / "samples.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"wrote metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

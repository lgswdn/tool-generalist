#!/usr/bin/env python3
"""Export PLY point clouds with the repository's current raw-mesh method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MESH_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")
DEFAULT_OUTPUT_DIR = Path("scripts/outputs/current_object_pointcloud_samples")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load COACD OBJ files with trimesh and directly sample all triangle surfaces, "
            "matching the current policy point-cloud cache generation method."
        )
    )
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Object names or OBJ paths. The .obj suffix is optional for object names.",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="Optionally select object_name values from an evaluation JSONL.",
    )
    parser.add_argument(
        "--name-contains",
        default="bowl",
        help="Case-insensitive selection filter used when --objects is omitted.",
    )
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument(
        "--num-points",
        type=int,
        nargs="+",
        default=[512],
        help="One or more point counts to generate per object.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale applied after sampling. Keep 1.0 to match canonical cache files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _object_name_from_jsonl_row(row: dict) -> str | None:
    name = row.get("object_name")
    if name is None and isinstance(row.get("record"), dict):
        name = row["record"].get("object_name")
    return None if name is None else str(name)


def _resolve_explicit_objects(mesh_dir: Path, values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value).expanduser()
        if not candidate.is_file():
            filename = candidate.name if candidate.suffix == ".obj" else f"{candidate.name}.obj"
            candidate = mesh_dir / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Object OBJ not found: {value}")
        paths.append(candidate.resolve())
    return paths


def _select_objects(args: argparse.Namespace) -> list[Path]:
    mesh_dir = args.mesh_dir.expanduser().resolve()
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    if args.objects:
        return _resolve_explicit_objects(mesh_dir, args.objects)

    names: list[str] = []
    if args.source_jsonl is not None:
        jsonl_path = args.source_jsonl.expanduser().resolve()
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {jsonl_path}:{line_number}")
                name = _object_name_from_jsonl_row(row)
                if name is not None and name not in names:
                    names.append(name)
    else:
        names = sorted(path.stem for path in mesh_dir.glob("*.obj"))

    substring = str(args.name_contains).lower()
    if substring:
        names = [name for name in names if substring in name.lower()]
    if args.num_objects <= 0:
        raise ValueError("--num-objects must be positive")
    names = names[: int(args.num_objects)]
    if not names:
        raise ValueError(
            f"No objects selected from {mesh_dir}; name filter was {args.name_contains!r}"
        )
    return _resolve_explicit_objects(mesh_dir, names)


def main() -> int:
    args = _parse_args()
    if any(count <= 0 for count in args.num_points):
        raise ValueError("Every --num-points value must be positive")
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required. Run this script with the Isaac conda environment."
        ) from exc

    object_paths = _select_objects(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] = []

    for object_index, obj_path in enumerate(object_paths):
        mesh = trimesh.load(str(obj_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            raise ValueError(f"Mesh has no triangle faces: {obj_path}")

        components = mesh.split(only_watertight=False)
        outputs: dict[str, str] = {}
        for point_count_index, point_count in enumerate(args.num_points):
            sample_seed = int(args.seed) + 1009 * object_index + 17 * point_count_index
            np.random.seed(sample_seed)
            points = np.asarray(mesh.sample(int(point_count)), dtype=np.float32)
            points *= np.float32(args.scale)
            output_path = output_dir / f"{obj_path.stem}_{int(point_count)}.ply"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")
            trimesh.PointCloud(points).export(output_path)
            outputs[str(int(point_count))] = str(output_path)
            print(
                f"sampled object={obj_path.stem} points={point_count} "
                f"components={len(components)} output={output_path}",
                flush=True,
            )

        metadata.append(
            {
                "object_name": obj_path.stem,
                "obj_path": str(obj_path),
                "sampling_method": "trimesh_mesh_sample_all_faces",
                "scale": float(args.scale),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "connected_components": int(len(components)),
                "bounds_after_scale": (np.asarray(mesh.bounds) * float(args.scale)).tolist(),
                "outputs": outputs,
            }
        )

    metadata_path = output_dir / "samples.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"wrote metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

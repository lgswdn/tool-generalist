#!/usr/bin/env python3
"""Filter contact cases by convex-hull overlap and export each match as OBJ.

For one contact_pt_env_v1 file, this script reconstructs every recorded object
and tool pose, computes the intersection volume of their convex hulls, and
exports cases whose overlap volume is above a threshold.  Each exported OBJ uses
the same three groups as export_contact_obj_viz.py: floor, pre_object, pre_tool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from contact_generation.export_contact_obj_viz import (
        load_centered_mesh,
        make_floor,
        np_array,
        safe_stem,
        transform_vertices,
        write_mtl,
        write_obj,
    )
except ModuleNotFoundError as exc:
    if exc.name != "contact_generation":
        raise
    from export_contact_obj_viz import (
        load_centered_mesh,
        make_floor,
        np_array,
        safe_stem,
        transform_vertices,
        write_mtl,
        write_obj,
    )


@dataclass(frozen=True)
class HullData:
    equations: np.ndarray


@dataclass(frozen=True)
class CaseResult:
    contact_index: int
    overlap_volume: float
    obj_path: str | None
    mtl_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_path", help="Path to one contact_pt_env_v1 .pt file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where selected case OBJ/MTL files and manifest are written. "
            "Defaults to contact_generation/obj_exports/<pt-stem>_convex_overlap_cases."
        ),
    )
    parser.add_argument(
        "--overlap-volume-threshold",
        type=float,
        default=1e-6,
        help="Export cases with convex-hull intersection volume strictly above this value.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on scanned contact cases, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-exports",
        type=int,
        default=None,
        help="Optional cap on exported cases after thresholding.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Scan every Nth contact case. Defaults to all cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute overlap volumes and write only the manifest, without OBJ/MTL exports.",
    )
    parser.add_argument("--floor-z", type=float, default=0.0, help="Floor plane z coordinate.")
    parser.add_argument("--floor-margin", type=float, default=0.20, help="Extra XY margin around the case meshes.")
    parser.add_argument("--floor-min-size", type=float, default=0.60, help="Minimum square floor side length.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of largest-overlap cases to include in the printed summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt_path = Path(args.pt_path).expanduser()
    if not pt_path.exists():
        raise FileNotFoundError(f"contact .pt does not exist: {pt_path}")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if args.max_cases is not None and int(args.max_cases) <= 0:
        raise ValueError("--max-cases must be positive when provided")
    if args.max_exports is not None and int(args.max_exports) <= 0:
        raise ValueError("--max-exports must be positive when provided")
    if float(args.overlap_volume_threshold) < 0.0:
        raise ValueError("--overlap-volume-threshold must be non-negative")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("contact_generation") / "obj_exports" / f"{safe_stem(pt_path.stem)}_convex_overlap_cases"
    )
    result = filter_and_export_cases(
        pt_path=pt_path,
        output_dir=output_dir,
        overlap_volume_threshold=float(args.overlap_volume_threshold),
        max_cases=int(args.max_cases) if args.max_cases is not None else None,
        max_exports=int(args.max_exports) if args.max_exports is not None else None,
        stride=int(args.stride),
        dry_run=bool(args.dry_run),
        floor_z=float(args.floor_z),
        floor_margin=float(args.floor_margin),
        floor_min_size=float(args.floor_min_size),
        top_n=int(args.top_n),
    )

    print(
        "[export_contact_all_cases_obj] "
        f"scanned={result['num_scanned_cases']} selected={result['num_selected_cases']} "
        f"exported={result['num_exported_cases']} threshold={result['overlap_volume_threshold']:.9g}",
        flush=True,
    )
    print(f"[export_contact_all_cases_obj] manifest={result['manifest_path']}", flush=True)
    if result["top_cases"]:
        summary = ", ".join(
            f"c{item['contact_index']}={item['overlap_volume']:.9g}" for item in result["top_cases"]
        )
        print(f"[export_contact_all_cases_obj] top_overlap {summary}", flush=True)
    return 0


def filter_and_export_cases(
    *,
    pt_path: Path,
    output_dir: Path,
    overlap_volume_threshold: float,
    max_cases: int | None,
    max_exports: int | None,
    stride: int,
    dry_run: bool,
    floor_z: float,
    floor_margin: float,
    floor_min_size: float,
    top_n: int,
) -> dict[str, Any]:
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"payload is not a dict: {pt_path}")
    if data.get("schema_version") != "contact_pt_env_v1":
        raise ValueError(f"unexpected schema_version={data.get('schema_version')!r}: {pt_path}")
    if str(data.get("generation_status", "")) != "complete":
        raise ValueError(f"generation_status={data.get('generation_status')!r}: {pt_path}")

    num_contacts = int(data["num_contacts"])
    if num_contacts <= 0:
        raise ValueError(f"contact file has no contact cases: {pt_path}")
    contact_indices = list(range(0, num_contacts, int(stride)))
    if max_cases is not None:
        contact_indices = contact_indices[: int(max_cases)]
    if not contact_indices:
        raise ValueError("no contact cases selected for scanning")

    object_local, object_faces = load_centered_mesh(
        data["object_mesh_path"],
        scale=float(data["object_scale"]),
        bbox_center=np_array(data["object_bbox_center_M"], (3,), "object_bbox_center_M"),
    )
    tool_local, tool_faces = load_centered_mesh(
        data["tool_mesh_path"],
        scale=np_array(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
        bbox_center=np_array(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M"),
    )
    object_hull = build_hull_data(object_local, "object")
    tool_hull = build_hull_data(tool_local, "tool")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases: list[dict[str, Any]] = []
    selected: list[CaseResult] = []
    exported = 0

    for scanned, contact_index in enumerate(contact_indices, start=1):
        object_R = np_array(data["object_rotation_E"][contact_index], (3, 3), "object_rotation_E")
        object_t = np_array(data["object_bbox_center_E"][contact_index], (3,), "object_bbox_center_E")
        tool_R = np_array(data["tool_rotation_E"][contact_index], (3, 3), "tool_rotation_E")
        tool_t = np_array(data["tool_translation_E"][contact_index], (3,), "tool_translation_E")

        object_vertices = transform_vertices(object_local, object_R, object_t)
        tool_vertices = transform_vertices(tool_local, tool_R, tool_t)
        overlap_volume = convex_hull_overlap_volume(
            object_hull,
            object_vertices,
            object_R,
            object_t,
            tool_hull,
            tool_vertices,
            tool_R,
            tool_t,
        )

        case_entry = {
            "contact_index": int(contact_index),
            "overlap_volume": float(overlap_volume),
            "selected": bool(overlap_volume > overlap_volume_threshold),
            "obj_path": None,
            "mtl_path": None,
        }
        if overlap_volume > overlap_volume_threshold:
            if not dry_run and (max_exports is None or exported < int(max_exports)):
                obj_path = output_dir / f"case_{contact_index:06d}_overlap_{format_volume_for_name(overlap_volume)}.obj"
                mtl_path = obj_path.with_suffix(".mtl")
                export_case_obj(
                    obj_path=obj_path,
                    mtl_path=mtl_path,
                    pt_path=pt_path,
                    data=data,
                    contact_index=contact_index,
                    overlap_volume=overlap_volume,
                    object_vertices=object_vertices,
                    object_faces=object_faces,
                    tool_vertices=tool_vertices,
                    tool_faces=tool_faces,
                    floor_z=floor_z,
                    floor_margin=floor_margin,
                    floor_min_size=floor_min_size,
                )
                exported += 1
                case_entry["obj_path"] = str(obj_path)
                case_entry["mtl_path"] = str(mtl_path)
            selected.append(
                CaseResult(
                    contact_index=int(contact_index),
                    overlap_volume=float(overlap_volume),
                    obj_path=case_entry["obj_path"],
                    mtl_path=case_entry["mtl_path"],
                )
            )
        all_cases.append(case_entry)
        if scanned == len(contact_indices) or scanned % 25 == 0:
            print(
                "[export_contact_all_cases_obj] "
                f"scanned {scanned}/{len(contact_indices)} selected={len(selected)} exported={exported}",
                flush=True,
            )

    top_cases = sorted(all_cases, key=lambda item: float(item["overlap_volume"]), reverse=True)
    top_cases = top_cases[: max(0, int(top_n))]
    manifest = {
        "schema_version": "contact_convex_overlap_export_manifest_v1",
        "pt_path": str(pt_path.resolve()),
        "output_dir": str(output_dir),
        "object_id": str(data.get("object_id", "")),
        "tool_id": str(data.get("tool_id", "")),
        "object_mesh_path": str(data.get("object_mesh_path", "")),
        "tool_mesh_path": str(data.get("tool_mesh_path", "")),
        "num_contacts": int(num_contacts),
        "num_scanned_cases": int(len(contact_indices)),
        "num_selected_cases": int(len(selected)),
        "num_exported_cases": int(exported),
        "overlap_volume_threshold": float(overlap_volume_threshold),
        "stride": int(stride),
        "max_cases": max_cases,
        "max_exports": max_exports,
        "dry_run": bool(dry_run),
        "floor_z": float(floor_z),
        "floor_margin": float(floor_margin),
        "floor_min_size": float(floor_min_size),
        "selected_cases": [case.__dict__ for case in selected],
        "top_cases": top_cases,
        "all_cases": all_cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_path": str(manifest_path)}


def build_hull_data(vertices: np.ndarray, label: str) -> HullData:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.asarray(vertices, dtype=np.float64))
    equations = np.asarray(hull.equations, dtype=np.float64)
    if equations.ndim != 2 or equations.shape[1] != 4:
        raise ValueError(f"{label} convex hull equations must have shape (N, 4), got {equations.shape}")
    return HullData(equations=equations)


def convex_hull_overlap_volume(
    object_hull: HullData,
    object_vertices: np.ndarray,
    object_R: np.ndarray,
    object_t: np.ndarray,
    tool_hull: HullData,
    tool_vertices: np.ndarray,
    tool_R: np.ndarray,
    tool_t: np.ndarray,
) -> float:
    if not aabb_intersects(object_vertices, tool_vertices):
        return 0.0

    object_halfspaces = transform_hull_halfspaces(object_hull, object_R, object_t)
    tool_halfspaces = transform_hull_halfspaces(tool_hull, tool_R, tool_t)
    halfspaces = np.concatenate([object_halfspaces, tool_halfspaces], axis=0)
    return halfspace_intersection_volume(halfspaces)


def aabb_intersects(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(a.min(axis=0) <= b.max(axis=0) + eps) and np.all(b.min(axis=0) <= a.max(axis=0) + eps))


def transform_hull_halfspaces(hull: HullData, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    local_normals = hull.equations[:, :3]
    local_offsets = hull.equations[:, 3]
    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    world_normals = local_normals @ rot.T
    world_offsets = local_offsets - world_normals @ trans
    return np.concatenate([world_normals, world_offsets[:, None]], axis=1)


def halfspace_intersection_volume(halfspaces: np.ndarray) -> float:
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

    hs = np.asarray(halfspaces, dtype=np.float64)
    normals = hs[:, :3]
    offsets = hs[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(hs).all(axis=1) & (norms > 1e-12)
    normals = normals[valid]
    offsets = offsets[valid]
    norms = norms[valid]
    if normals.shape[0] < 4:
        return 0.0

    # Chebyshev center gives a strict interior point for HalfspaceIntersection.
    aub = np.concatenate([normals, norms[:, None]], axis=1)
    bub = -offsets
    c = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    result = linprog(
        c,
        A_ub=aub,
        b_ub=bub,
        bounds=[(None, None), (None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not result.success or not np.isfinite(result.x).all() or float(result.x[3]) <= 1e-10:
        return 0.0

    try:
        intersection = HalfspaceIntersection(np.concatenate([normals, offsets[:, None]], axis=1), result.x[:3])
        points = np.asarray(intersection.intersections, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 4:
            return 0.0
        return float(max(0.0, ConvexHull(points).volume))
    except (QhullError, ValueError):
        return 0.0


def export_case_obj(
    *,
    obj_path: Path,
    mtl_path: Path,
    pt_path: Path,
    data: dict[str, Any],
    contact_index: int,
    overlap_volume: float,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    tool_vertices: np.ndarray,
    tool_faces: np.ndarray,
    floor_z: float,
    floor_margin: float,
    floor_min_size: float,
) -> None:
    floor_vertices, floor_faces = make_floor(
        [object_vertices, tool_vertices],
        floor_z=floor_z,
        margin=floor_margin,
        min_size=floor_min_size,
    )
    write_mtl(mtl_path)
    write_obj(
        obj_path,
        mtl_path=mtl_path,
        groups=[
            ("floor", floor_vertices, floor_faces),
            ("pre_object", object_vertices, object_faces),
            ("pre_tool", tool_vertices, tool_faces),
        ],
        comments=[
            f"source_pt {pt_path}",
            f"contact_index {contact_index}",
            f"convex_hull_overlap_volume {overlap_volume:.12g}",
            f"tool_id {data.get('tool_id', '')}",
            f"object_id {data.get('object_id', '')}",
            f"object_mesh_path {data.get('object_mesh_path', '')}",
            f"tool_mesh_path {data.get('tool_mesh_path', '')}",
            "include_post_contact false",
            "mesh_transform scaled_vertices_minus_bbox_center_then_env_pose",
        ],
    )


def format_volume_for_name(value: float) -> str:
    return f"{float(value):.6g}".replace("+", "").replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())

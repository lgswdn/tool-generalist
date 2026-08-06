#!/usr/bin/env python3
"""Measure how much of each contact object lies between the gripper fingers.

The metric is the fraction of the sampled object surface that lies inside the
convex hull of the exact left/right finger and tip meshes, while remaining
outside the solid gripper mesh according to the cached signed distance.  This
is a surface-sample capture fraction, not an object-volume fraction.

This analyzer intentionally accepts only geometry candidate artifacts with
cached mesh SDF.  Missing fields or unrecognized finger meshes are errors.
"""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.geometry.gripper_cavity import finger_hull_halfspaces


SCHEMA_VERSION = "contact_candidate_v1"
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Contact artifact directory containing *.candidate.pt files.")
    parser.add_argument("--output", required=True, help="Summary JSON path.")
    parser.add_argument(
        "--cases-output",
        default=None,
        help="Optional JSONL path containing one record per analyzed contact case.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated capture fractions used for P(capture >= x).",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 analyzes every candidate file.")
    parser.add_argument(
        "--max-cases-per-file",
        type=int,
        default=0,
        help="0 analyzes every candidate in each file.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Analyze every Nth candidate.")
    parser.add_argument(
        "--case-chunk-size",
        type=int,
        default=32,
        help="Cases transformed and tested against the hull together.",
    )
    parser.add_argument(
        "--hull-tolerance",
        type=float,
        default=1e-7,
        help="Numerical tolerance for point-in-hull tests in meters.",
    )
    parser.add_argument(
        "--material-tolerance",
        type=float,
        default=0.0,
        help=(
            "A point is treated as outside gripper material when its cached "
            "object-to-tool signed distance is at least -this value."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N files. 0 disables progress output.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(token.strip()) for token in raw.split(",") if token.strip()}))
    if not values or values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("--thresholds must contain values in [0, 1]")
    return values


def candidate_files(data_dir: str | Path, max_files: int) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Contact dataset directory does not exist: {root}")
    paths: list[Path] = []
    for tool_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(tool_dir.glob("*.candidate.pt")):
            paths.append(path)
            if int(max_files) > 0 and len(paths) >= int(max_files):
                return paths
    if not paths:
        raise RuntimeError(f"No tool/*.candidate.pt files found under {root}")
    return paths


def load_candidate(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected {SCHEMA_VERSION} payload: {path}")
    count = int(payload.get("num_candidates", 0))
    if count <= 0:
        raise ValueError(f"Candidate payload contains no cases: {path}")
    if payload.get("object_point_tool_signed_sdf") is None:
        raise ValueError(f"Candidate payload has no cached object-to-tool mesh SDF: {path}")
    if not isinstance(payload.get("candidates"), dict):
        raise ValueError(f"Candidate payload has no candidates mapping: {path}")
    return payload


def numpy_value(value: Any, *, dtype: np.dtype[Any] = np.dtype(np.float64)) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def finger_hull_equations(payload: dict[str, Any]) -> np.ndarray:
    scale = numpy_value(payload["tool_scale_xyz"]).reshape(3)
    center = numpy_value(payload["tool_bbox_center_M"]).reshape(3)
    return finger_hull_halfspaces(
        payload["tool_mesh_path"],
        scale_xyz=tuple(float(value) for value in scale),
        bbox_center=tuple(float(value) for value in center),
        )


def selected_indices(count: int, stride: int, max_cases_per_file: int) -> np.ndarray:
    if int(stride) <= 0:
        raise ValueError("--stride must be positive")
    indices = np.arange(0, int(count), int(stride), dtype=np.int64)
    if int(max_cases_per_file) > 0:
        indices = indices[: int(max_cases_per_file)]
    if indices.size == 0:
        raise ValueError("Case selection is empty")
    return indices


def points_inside_hull(
    points: np.ndarray,
    equations: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    normals = equations[:, :3]
    offsets = equations[:, 3]
    signed_planes = np.einsum("bki,fi->bkf", points, normals, optimize=True)
    signed_planes += offsets.reshape(1, 1, -1)
    return np.all(signed_planes <= float(tolerance), axis=-1)


def analyze_payload(
    payload: dict[str, Any],
    *,
    indices: np.ndarray,
    case_chunk_size: int,
    hull_tolerance: float,
    material_tolerance: float,
) -> list[dict[str, Any]]:
    if int(case_chunk_size) <= 0:
        raise ValueError("--case-chunk-size must be positive")

    count = int(payload["num_candidates"])
    candidates = payload["candidates"]
    object_points_O = numpy_value(payload["object_points_O"])
    sdf = numpy_value(payload["object_point_tool_signed_sdf"])
    if object_points_O.ndim != 2 or object_points_O.shape[1] != 3:
        raise ValueError(f"object_points_O must have shape (K, 3), got {object_points_O.shape}")
    if sdf.shape != (count, object_points_O.shape[0]):
        raise ValueError(
            "object_point_tool_signed_sdf shape does not match candidate/object point counts: "
            f"sdf={sdf.shape} candidates={count} points={object_points_O.shape[0]}"
        )

    object_R = numpy_value(candidates["object_rotation_E"])
    object_t = numpy_value(candidates["object_bbox_center_E"])
    tool_R = numpy_value(candidates["tool_rotation_E"])
    tool_t = numpy_value(candidates["tool_translation_E"])
    expected = {
        "object_rotation_E": (count, 3, 3),
        "object_bbox_center_E": (count, 3),
        "tool_rotation_E": (count, 3, 3),
        "tool_translation_E": (count, 3),
    }
    actual = {
        "object_rotation_E": object_R.shape,
        "object_bbox_center_E": object_t.shape,
        "tool_rotation_E": tool_R.shape,
        "tool_translation_E": tool_t.shape,
    }
    if actual != expected:
        raise ValueError(f"Candidate pose tensor shapes are invalid: expected={expected} actual={actual}")

    equations = finger_hull_equations(payload)
    records: list[dict[str, Any]] = []
    for start in range(0, int(indices.size), int(case_chunk_size)):
        chunk_indices = indices[start : start + int(case_chunk_size)]
        obj_R = object_R[chunk_indices]
        obj_t = object_t[chunk_indices]
        grip_R = tool_R[chunk_indices]
        grip_t = tool_t[chunk_indices]
        object_E = np.einsum("bij,kj->bki", obj_R, object_points_O, optimize=True)
        object_E += obj_t[:, None, :]
        object_T = np.einsum(
            "bki,bij->bkj",
            object_E - grip_t[:, None, :],
            grip_R,
            optimize=True,
        )
        in_finger_span = points_inside_hull(
            object_T,
            equations,
            tolerance=float(hull_tolerance),
        )
        chunk_sdf = sdf[chunk_indices]
        outside_material = chunk_sdf >= -float(material_tolerance)
        in_cavity = in_finger_span & outside_material

        object_center_E = obj_t
        object_center_T = np.einsum(
            "bi,bij->bj",
            object_center_E - grip_t,
            grip_R,
            optimize=True,
        )
        center_inside = points_inside_hull(
            object_center_T[:, None, :],
            equations,
            tolerance=float(hull_tolerance),
        )[:, 0]

        for local_index, case_index in enumerate(chunk_indices):
            records.append(
                {
                    "contact_index": int(case_index),
                    "capture_surface_fraction": float(in_cavity[local_index].mean()),
                    "finger_span_surface_fraction": float(in_finger_span[local_index].mean()),
                    "material_penetration_surface_fraction": float(
                        (chunk_sdf[local_index] < 0.0).mean()
                    ),
                    "object_center_inside_finger_span": bool(center_inside[local_index]),
                }
            )
    return records


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def threshold_proportions(values: Iterable[float], thresholds: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot threshold an empty metric")
    return {
        f"{100.0 * threshold:g}%": float(np.mean(array >= float(threshold)))
        for threshold in thresholds
    }


def new_accumulator() -> dict[str, Any]:
    return {
        "capture": array("f"),
        "span": array("f"),
        "penetration": array("f"),
        "center_inside_count": 0,
        "count": 0,
    }


def accumulate(accumulator: dict[str, Any], records: list[dict[str, Any]]) -> None:
    accumulator["capture"].extend(
        float(record["capture_surface_fraction"]) for record in records
    )
    accumulator["span"].extend(
        float(record["finger_span_surface_fraction"]) for record in records
    )
    accumulator["penetration"].extend(
        float(record["material_penetration_surface_fraction"]) for record in records
    )
    accumulator["center_inside_count"] += sum(
        bool(record["object_center_inside_finger_span"]) for record in records
    )
    accumulator["count"] += len(records)


def summarize_accumulator(
    accumulator: dict[str, Any],
    *,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("Cannot summarize zero contact cases")
    capture = accumulator["capture"]
    return {
        "num_cases": count,
        "capture_surface_fraction": distribution(capture),
        "case_proportion_at_or_above_capture_fraction": threshold_proportions(
            capture, thresholds
        ),
        "finger_span_surface_fraction": distribution(accumulator["span"]),
        "material_penetration_surface_fraction": distribution(accumulator["penetration"]),
        "object_center_inside_finger_span_fraction": float(
            int(accumulator["center_inside_count"]) / count
        ),
    }


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    paths = candidate_files(args.data_dir, args.max_files)
    output_path = Path(args.output).expanduser().resolve()
    cases_path = (
        Path(args.cases_output).expanduser().resolve()
        if args.cases_output is not None
        else None
    )

    overall_accumulator = new_accumulator()
    per_tool_accumulators: dict[str, dict[str, Any]] = defaultdict(new_accumulator)
    case_handle = None
    try:
        if cases_path is not None:
            cases_path.parent.mkdir(parents=True, exist_ok=True)
            case_handle = cases_path.open("w", encoding="utf-8")
        for file_index, path in enumerate(paths, start=1):
            payload = load_candidate(path)
            indices = selected_indices(
                int(payload["num_candidates"]),
                args.stride,
                args.max_cases_per_file,
            )
            records = analyze_payload(
                payload,
                indices=indices,
                case_chunk_size=args.case_chunk_size,
                hull_tolerance=args.hull_tolerance,
                material_tolerance=args.material_tolerance,
            )
            tool_id = str(payload["tool_id"])
            for record in records:
                record.update(
                    {
                        "candidate_path": str(path),
                        "tool_id": tool_id,
                        "object_id": str(payload["object_id"]),
                    }
                )
                if case_handle is not None:
                    case_handle.write(json.dumps(record, sort_keys=True) + "\n")
            accumulate(overall_accumulator, records)
            accumulate(per_tool_accumulators[tool_id], records)
            if int(args.progress_every) > 0 and (
                file_index == len(paths) or file_index % int(args.progress_every) == 0
            ):
                print(
                    "[gripper-cavity] "
                    f"files={file_index}/{len(paths)} "
                    f"cases={int(overall_accumulator['count'])}",
                    flush=True,
                )
    finally:
        if case_handle is not None:
            case_handle.close()

    summary = {
        "schema_version": "gripper_cavity_occupancy_analysis_v1",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "definition": (
            "Fraction of sampled object surface points inside the convex hull of "
            "the exact left/right finger and tip meshes and outside solid gripper material."
        ),
        "interpretation": (
            "This estimates between-finger capture/concavity occupancy. It is not "
            "the fraction of object mesh volume enclosed by the hand."
        ),
        "num_files": len(paths),
        "stride": int(args.stride),
        "max_cases_per_file": int(args.max_cases_per_file),
        "hull_tolerance_m": float(args.hull_tolerance),
        "material_tolerance_m": float(args.material_tolerance),
        "thresholds": list(thresholds),
        "overall": summarize_accumulator(overall_accumulator, thresholds=thresholds),
        "per_tool": {
            tool_id: summarize_accumulator(accumulator, thresholds=thresholds)
            for tool_id, accumulator in sorted(per_tool_accumulators.items())
        },
        "cases_output": str(cases_path) if cases_path is not None else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[gripper-cavity] wrote {output_path}", flush=True)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

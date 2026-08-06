#!/usr/bin/env python3
"""Materialize legacy contact poses with canonical 128-bin gripper clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

from utils.assets.tool_assets import load_tool_kinematic_cloud


SCHEMA_VERSION = "canonical_gripper_candidate_dataset_v1"
CANDIDATE_SCHEMA_VERSION = "contact_candidate_v1"
STALE_POINT_LABELS = (
    "tool_point_inside_object",
    "object_point_inside_tool",
    "tool_point_object_signed_sdf",
    "object_point_tool_signed_sdf",
)


def _tensor_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value, dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _canonical_tool_points(payload: Mapping[str, Any]) -> tuple[float, torch.Tensor]:
    tool_id = str(payload["tool_id"])
    if not tool_id.startswith(("generated_gripper_", "one_dof_gripper_")):
        raise RuntimeError(
            f"Canonical materialization only accepts generated grippers, got {tool_id!r}"
        )
    mesh_path = Path(str(payload["tool_mesh_path"])).expanduser().resolve()
    try:
        tools_json = mesh_path.parents[3] / "tools_adjusted.json"
    except IndexError as exc:
        raise RuntimeError(f"Cannot locate tools_adjusted.json from {mesh_path}") from exc
    opening, cloud = load_tool_kinematic_cloud(tools_json, tool_id)
    center = torch.as_tensor(
        payload["tool_bbox_center_M"], dtype=torch.float32
    )
    if tuple(center.shape) != (3,):
        raise RuntimeError(
            f"{tool_id!r} tool_bbox_center_M must have shape (3,), "
            f"got {tuple(center.shape)}"
        )
    points = (cloud.to(dtype=torch.float32) - center).contiguous()
    if tuple(points.shape) != (512, 3) or not bool(points.isfinite().all()):
        raise RuntimeError(
            f"{tool_id!r} canonical cloud must be finite (512, 3), "
            f"got {tuple(points.shape)}"
        )
    return opening, points


def _materialize_file(
    source: Path,
    destination: Path,
    *,
    recorded_destination: Path | None = None,
) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"Invalid geometry candidate artifact: {source}")
    if int(payload.get("num_candidates", 0)) <= 0:
        raise RuntimeError(f"Candidate artifact has no contact poses: {source}")
    saved_points = torch.as_tensor(
        payload["tool_points_T"], dtype=torch.float32
    ).contiguous()
    if tuple(saved_points.shape) != (512, 3):
        raise RuntimeError(
            f"{source} tool_points_T must have shape (512, 3), "
            f"got {tuple(saved_points.shape)}"
        )
    opening, canonical_points = _canonical_tool_points(payload)
    result = dict(payload)
    for key in STALE_POINT_LABELS:
        result[key] = None
    result.update(
        {
            "generator": "scripts.materialize_canonical_gripper_candidates",
            "candidate_artifact_path": str(
                (recorded_destination or destination).resolve()
            ),
            "tool_points_T": canonical_points,
            "source_candidate_artifact_path": str(source.resolve()),
            "source_tool_points_sha256": _tensor_sha256(saved_points),
            "canonical_tool_points_sha256": _tensor_sha256(canonical_points),
            "canonical_opening_fraction": float(opening),
            "canonical_gripper_cloud_contract": "128_bins_512_corresponding_points",
            "precomputed_point_labels_removed": list(STALE_POINT_LABELS),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    _, expected = _canonical_tool_points(reloaded)
    actual = torch.as_tensor(reloaded["tool_points_T"], dtype=torch.float32)
    if not torch.equal(actual, expected):
        raise RuntimeError(
            f"Materialized cloud failed exact canonical validation: {destination}"
        )
    if any(reloaded.get(key) is not None for key in STALE_POINT_LABELS):
        raise RuntimeError(f"Stale point labels remain in {destination}")
    return {
        "source": str(source.resolve()),
        "path": destination.name,
        "tool_id": str(payload["tool_id"]),
        "object_id": str(payload["object_id"]),
        "num_candidates": int(payload["num_candidates"]),
        "cloud_changed": not torch.equal(saved_points, canonical_points),
    }


def materialize(source_dir: Path, output_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source candidate directory does not exist: {source_dir}")
    source_files = sorted(source_dir.rglob("*.candidate.pt"))
    if not source_files:
        raise RuntimeError(f"No candidate files found under {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        records = []
        for index, source in enumerate(source_files, start=1):
            relative = source.relative_to(source_dir)
            record = _materialize_file(
                source,
                temporary / relative,
                recorded_destination=output_dir / relative,
            )
            record["path"] = str(relative)
            records.append(record)
            if index % 25 == 0 or index == len(source_files):
                print(
                    f"[canonical-candidates] files={index}/{len(source_files)}",
                    flush=True,
                )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "file_count": len(records),
            "candidate_count": sum(item["num_candidates"] for item in records),
            "changed_cloud_file_count": sum(
                int(item["cloud_changed"]) for item in records
            ),
            "canonical_gripper_cloud_contract": (
                "128_bins_512_corresponding_points"
            ),
            "precomputed_point_labels_removed": list(STALE_POINT_LABELS),
            "files": records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    materialize(args.source_dir, args.output_dir)

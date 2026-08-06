#!/usr/bin/env python3
"""Analyze convex-hull overlap volumes for a contact dataset.

This is the dataset-wide reporting counterpart of export_contact_all_cases_obj.py.
For every formal contact_pt_env_v1 .pt file under a dataset directory, it computes
the intersection volume of the posed object convex hull and posed tool convex
hull for each recorded contact pose pair, then writes per-contact JSONL and a
dataset summary JSON.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import heapq
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_generation.export_contact_all_cases_obj import build_hull_data, convex_hull_overlap_volume
from contact_generation.export_contact_obj_viz import load_centered_mesh, np_array, transform_vertices


BLOCKED_PT_SUFFIXES = (
    ".candidate.pt",
    ".physics_debug.pt",
    ".stabilized_success.pt",
    ".stabilized.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Root contact dataset/artifact directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reports are written. Filenames use the contact dataset hash.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of formal .pt files, useful for smoke tests. 0 means all.",
    )
    parser.add_argument(
        "--max-contacts-per-file",
        type=int,
        default=0,
        help="Optional cap on contact pose pairs analyzed per .pt file. 0 means all.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Analyze every Nth contact pose pair in each .pt file. Defaults to all.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of largest-overlap cases to keep in the summary JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes. Defaults to min(8, CPU count).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N files. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Skip invalid/incomplete contact files instead of failing.",
    )
    return parser.parse_args()


def iter_contact_pt_files(data_dir: str | Path, *, max_files: int = 0) -> Iterable[Path]:
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data dir does not exist: {root}")
    emitted = 0
    for path in root.rglob("*.pt"):
        if any(str(path).endswith(suffix) for suffix in BLOCKED_PT_SUFFIXES):
            continue
        yield path
        emitted += 1
        if int(max_files) > 0 and emitted >= int(max_files):
            return


def dataset_hash_from_path(data_dir: str | Path) -> str:
    name = Path(data_dir).expanduser().resolve().name
    if not name:
        raise ValueError(f"Could not infer dataset hash from data_dir={data_dir}")
    return name


def resolve_report_paths(data_dir: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir).expanduser()
    dataset_hash = dataset_hash_from_path(data_dir)
    return (
        out / f"{dataset_hash}.convex_overlap.jsonl",
        out / f"{dataset_hash}.convex_overlap.summary.json",
    )


def load_contact_payload(pt_path: Path) -> dict[str, Any]:
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"payload is not a dict: {pt_path}")
    if data.get("schema_version") != "contact_pt_env_v1":
        raise ValueError(f"unexpected schema_version={data.get('schema_version')!r}: {pt_path}")
    if str(data.get("generation_status", "")) != "complete":
        raise ValueError(f"generation_status={data.get('generation_status')!r}: {pt_path}")
    if int(data.get("num_contacts", 0)) <= 0:
        raise ValueError(f"contact file has no contact cases: {pt_path}")
    return data


def contact_indices_for_file(num_contacts: int, *, stride: int, max_contacts_per_file: int) -> list[int]:
    indices = list(range(0, int(num_contacts), int(stride)))
    if int(max_contacts_per_file) > 0:
        indices = indices[: int(max_contacts_per_file)]
    return indices


def analyze_file(
    pt_path: Path,
    *,
    stride: int,
    max_contacts_per_file: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_contact_payload(pt_path)
    num_contacts = int(data["num_contacts"])
    contact_indices = contact_indices_for_file(
        num_contacts,
        stride=int(stride),
        max_contacts_per_file=int(max_contacts_per_file),
    )
    if not contact_indices:
        raise ValueError(f"no contact cases selected for scanning: {pt_path}")

    object_local, _object_faces = load_centered_mesh(
        data["object_mesh_path"],
        scale=float(data["object_scale"]),
        bbox_center=np_array(data["object_bbox_center_M"], (3,), "object_bbox_center_M"),
    )
    tool_local, _tool_faces = load_centered_mesh(
        data["tool_mesh_path"],
        scale=np_array(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
        bbox_center=np_array(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M"),
    )
    object_hull = build_hull_data(object_local, "object")
    tool_hull = build_hull_data(tool_local, "tool")

    records: list[dict[str, Any]] = []
    overlap_values: list[float] = []
    for contact_index in contact_indices:
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
        overlap_volume = float(overlap_volume)
        overlap_values.append(overlap_volume)
        records.append(
            {
                "pt_path": str(pt_path),
                "object_id": str(data["object_id"]),
                "tool_id": str(data["tool_id"]),
                "contact_index": int(contact_index),
                "overlap_volume": overlap_volume,
            }
        )

    file_summary = summarize_values(
        overlap_values,
        prefix="overlap_volume",
        extra={
            "pt_path": str(pt_path),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "num_contacts_in_file": num_contacts,
            "num_contacts_analyzed": len(contact_indices),
            "stride": int(stride),
        },
    )
    return records, file_summary


def summarize_values(values: list[float], *, prefix: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = dict(extra or {})
    if not values:
        summary.update(
            {
                f"{prefix}_count": 0,
                f"{prefix}_mean": None,
                f"{prefix}_min": None,
                f"{prefix}_p05": None,
                f"{prefix}_p50": None,
                f"{prefix}_p95": None,
                f"{prefix}_max": None,
            }
        )
        return summary
    arr = np.asarray(values, dtype=np.float64)
    summary.update(
        {
            f"{prefix}_count": int(arr.size),
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_p05": float(np.quantile(arr, 0.05)),
            f"{prefix}_p50": float(np.quantile(arr, 0.50)),
            f"{prefix}_p95": float(np.quantile(arr, 0.95)),
            f"{prefix}_max": float(arr.max()),
        }
    )
    return summary


def update_top_cases(heap: list[tuple[float, str, dict[str, Any]]], record: dict[str, Any], *, top_n: int) -> None:
    if int(top_n) <= 0:
        return
    tie_breaker = f"{record.get('pt_path', '')}:{int(record.get('contact_index', -1))}"
    item = (float(record["overlap_volume"]), tie_breaker, dict(record))
    if len(heap) < int(top_n):
        heapq.heappush(heap, item)
        return
    if item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def main() -> int:
    args = parse_args()
    if int(args.max_files) < 0:
        raise ValueError("--max-files must be non-negative")
    if int(args.max_contacts_per_file) < 0:
        raise ValueError("--max-contacts-per-file must be non-negative")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.top_n) < 0:
        raise ValueError("--top-n must be non-negative")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")

    output_path, summary_path = resolve_report_paths(args.data_dir, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[analyze_contact_convex_overlap] scanning "
        f"data_dir={Path(args.data_dir).expanduser()} output_dir={output_path.parent} "
        f"max_files={int(args.max_files)} workers={int(args.workers)}",
        flush=True,
    )

    num_contacts = 0
    overlap_values: list[float] = []
    file_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    top_heap: list[tuple[float, str, dict[str, Any]]] = []
    progress_every = int(args.progress_every)
    files_submitted = 0
    files_done = 0
    max_pending = max(1, int(args.workers) * 2)
    file_iter = iter(iter_contact_pt_files(args.data_dir, max_files=int(args.max_files)))
    with output_path.open("w", encoding="utf-8") as output_f:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            pending = {}
            input_exhausted = False
            while True:
                while not input_exhausted and len(pending) < max_pending:
                    try:
                        pt_path = next(file_iter)
                    except StopIteration:
                        input_exhausted = True
                        break
                    future = executor.submit(
                        analyze_file,
                        pt_path,
                        stride=int(args.stride),
                        max_contacts_per_file=int(args.max_contacts_per_file),
                    )
                    pending[future] = pt_path
                    files_submitted += 1

                if not pending:
                    break

                done, _not_done = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    pt_path = pending.pop(future)
                    files_done += 1
                    try:
                        records, file_summary = future.result()
                    except Exception as exc:
                        if not bool(args.allow_invalid):
                            for pending_future in pending:
                                pending_future.cancel()
                            raise
                        skipped.append({"pt_path": str(pt_path), "error": str(exc)})
                        records = []
                        file_summary = None
                    if file_summary is not None:
                        for record in records:
                            output_f.write(json.dumps(record, sort_keys=True) + "\n")
                            overlap = float(record["overlap_volume"])
                            overlap_values.append(overlap)
                            num_contacts += 1
                            update_top_cases(top_heap, record, top_n=int(args.top_n))
                        file_summaries.append(file_summary)
                    if progress_every > 0 and files_done % progress_every == 0:
                        print(
                            "[analyze_contact_convex_overlap] progress "
                            f"files_done={files_done} files_submitted={files_submitted} "
                            f"contacts={num_contacts} skipped={len(skipped)}",
                            flush=True,
                        )

    if files_submitted == 0:
        raise RuntimeError(f"No training .pt files found under {args.data_dir}")

    top_cases = [item[2] for item in sorted(top_heap, key=lambda x: x[0], reverse=True)]
    summary = summarize_values(
        overlap_values,
        prefix="overlap_volume",
        extra={
            "data_dir": str(Path(args.data_dir).expanduser()),
            "num_files_seen": int(files_submitted),
            "num_files_done": int(files_done),
            "num_files_analyzed": int(len(file_summaries)),
            "num_contacts": int(num_contacts),
            "num_skipped": int(len(skipped)),
            "stride": int(args.stride),
            "max_files": int(args.max_files),
            "max_contacts_per_file": int(args.max_contacts_per_file),
            "workers": int(args.workers),
            "top_cases": top_cases,
            "file_summaries": file_summaries,
            "skipped": skipped,
        },
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    printable = {k: v for k, v in summary.items() if k not in {"file_summaries", "skipped", "top_cases"}}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    print(f"[analyze_contact_convex_overlap] wrote per-contact JSONL: {output_path}", flush=True)
    print(f"[analyze_contact_convex_overlap] wrote summary JSON: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Count objects with zero successful GraspGen lift episodes."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULTS_DIR = Path("/mnt/project/world_model/tool_generalist/ungrasp_result/full_5676_ep10")


@dataclass
class ObjectStats:
    object_index: int
    object_name: str | None = None
    successes: int = 0
    failures: int = 0
    episode_indices: set[int] = field(default_factory=set)
    worker_ids: set[int] = field(default_factory=set)

    @property
    def attempts(self) -> int:
        return self.successes + self.failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read GraspGen lift-eval JSONL results and report objects that had no "
            "successful grasp in the evaluated episodes."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing failures_rank_*.jsonl/all_results_rank_*.jsonl. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help=(
            "Expected evaluated episodes per object. If omitted, this is inferred from "
            "the largest episode_index present in the input."
        ),
    )
    parser.add_argument(
        "--total-objects",
        type=int,
        default=None,
        help="Optional total object count. Used only for summary completeness checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/zero_success_graspgen"),
        help="Directory for summary JSON and zero-success JSONL outputs.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the result directory name.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print the summary; do not write output files.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help=(
            "For failures-only inputs, count objects with fewer than expected observed "
            "failure episodes as zero-success too. By default these are kept separate "
            "because a missing failure row may mean that episode succeeded."
        ),
    )
    return parser.parse_args()


def _paths_for_pattern(results_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(str(results_dir / pattern)))


def _choose_input_paths(results_dir: Path) -> tuple[str, list[Path]]:
    all_results_paths = _paths_for_pattern(results_dir, "all_results_rank_*.jsonl")
    if all_results_paths:
        return "all_results", all_results_paths

    failure_paths = _paths_for_pattern(results_dir, "failures_rank_*.jsonl")
    if failure_paths:
        return "failures_only", failure_paths

    raise FileNotFoundError(
        f"No all_results_rank_*.jsonl or failures_rank_*.jsonl files found in {results_dir}"
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}, got {type(row).__name__}")
            yield row


def _row_object_index(row: dict[str, Any], path: Path) -> int:
    value = row.get("object_index")
    if value is None:
        raise ValueError(f"{path} contains a row without object_index")
    return int(value)


def _scan_rows(paths: list[Path]) -> tuple[dict[int, ObjectStats], dict[str, Any]]:
    stats_by_object: dict[int, ObjectStats] = {}
    seen_episode_keys: set[tuple[int, int]] = set()
    raw_rows = 0
    duplicate_rows = 0
    max_episode_index: int | None = None
    worker_ids: set[int] = set()

    for path in paths:
        for row in _iter_jsonl(path):
            raw_rows += 1
            object_index = _row_object_index(row, path)
            episode_index = int(row.get("episode_index", 0))
            episode_key = (object_index, episode_index)
            if episode_key in seen_episode_keys:
                duplicate_rows += 1
                continue
            seen_episode_keys.add(episode_key)

            item = stats_by_object.setdefault(object_index, ObjectStats(object_index=object_index))
            object_name = row.get("object_name")
            if isinstance(object_name, str):
                item.object_name = object_name

            worker_id = row.get("worker_id")
            if worker_id is not None:
                worker_id_int = int(worker_id)
                item.worker_ids.add(worker_id_int)
                worker_ids.add(worker_id_int)

            item.episode_indices.add(episode_index)
            max_episode_index = episode_index if max_episode_index is None else max(max_episode_index, episode_index)
            if bool(row.get("success", False)):
                item.successes += 1
            else:
                item.failures += 1

    metadata = {
        "raw_rows": raw_rows,
        "deduped_rows": raw_rows - duplicate_rows,
        "duplicate_rows": duplicate_rows,
        "max_episode_index": max_episode_index,
        "worker_ids": sorted(worker_ids),
    }
    return stats_by_object, metadata


def _object_row(item: ObjectStats) -> dict[str, Any]:
    return {
        "object_index": item.object_index,
        "object_name": item.object_name,
        "successes": item.successes,
        "failures": item.failures,
        "attempts": item.attempts,
        "episode_indices": sorted(item.episode_indices),
        "worker_ids": sorted(item.worker_ids),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_names(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["object_name"]) for row in rows if row.get("object_name")]
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    input_mode, paths = _choose_input_paths(results_dir)
    stats_by_object, scan_metadata = _scan_rows(paths)

    inferred_episodes = (
        int(scan_metadata["max_episode_index"]) + 1 if scan_metadata["max_episode_index"] is not None else 0
    )
    expected_episodes = int(args.expected_episodes) if args.expected_episodes is not None else inferred_episodes
    if expected_episodes <= 0:
        raise ValueError("Could not infer a positive episode count; pass --expected-episodes explicitly.")

    complete_zero_success: list[ObjectStats] = []
    partial_zero_success: list[ObjectStats] = []
    partial_failure_objects: list[ObjectStats] = []

    for item in stats_by_object.values():
        if item.successes > 0:
            continue
        if input_mode == "all_results":
            complete_zero_success.append(item)
        elif item.failures >= expected_episodes or args.include_partial:
            complete_zero_success.append(item)
        else:
            partial_zero_success.append(item)
            partial_failure_objects.append(item)

    complete_zero_success.sort(key=lambda item: item.object_index)
    partial_zero_success.sort(key=lambda item: item.object_index)

    total_objects = args.total_objects
    if total_objects is None and stats_by_object:
        total_objects = max(stats_by_object) + 1

    zero_success_rows = [_object_row(item) for item in complete_zero_success]
    partial_zero_success_rows = [_object_row(item) for item in partial_zero_success]
    summary = {
        "results_dir": str(results_dir),
        "input_mode": input_mode,
        "source_files": [str(path) for path in paths],
        "expected_episodes": expected_episodes,
        "inferred_episodes_from_input": inferred_episodes,
        "total_objects": total_objects,
        "observed_objects_in_input": len(stats_by_object),
        "zero_success_object_count": len(zero_success_rows),
        "partial_zero_success_object_count": len(partial_zero_success_rows),
        "objects_with_success_count": sum(1 for item in stats_by_object.values() if item.successes > 0),
        "objects_with_any_failure_count": sum(1 for item in stats_by_object.values() if item.failures > 0),
        "raw_rows": scan_metadata["raw_rows"],
        "deduped_rows": scan_metadata["deduped_rows"],
        "duplicate_rows": scan_metadata["duplicate_rows"],
        "worker_ids": scan_metadata["worker_ids"],
    }

    if total_objects is not None:
        summary["unobserved_or_all_success_objects"] = int(total_objects) - len(stats_by_object)

    prefix = args.prefix or results_dir.name
    if not args.no_write:
        output_dir = args.output_dir.expanduser().resolve() / prefix
        summary_path = output_dir / "zero_success_summary.json"
        zero_success_path = output_dir / "zero_success_objects.jsonl"
        zero_success_names_path = output_dir / "zero_success_object_names.txt"
        partial_path = output_dir / "partial_failure_no_known_success_objects.jsonl"
        summary["summary_path"] = str(summary_path)
        summary["zero_success_objects_path"] = str(zero_success_path)
        summary["zero_success_object_names_path"] = str(zero_success_names_path)
        summary["partial_failure_no_known_success_objects_path"] = str(partial_path)
        _write_json(summary_path, summary)
        _write_jsonl(zero_success_path, zero_success_rows)
        _write_names(zero_success_names_path, zero_success_rows)
        _write_jsonl(partial_path, partial_zero_success_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

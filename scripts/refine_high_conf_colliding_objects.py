#!/usr/bin/env python3
"""Refine an existing object manifest using stricter GraspGen confidence criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("scripts/outputs/graspgen_episode_consistency_full_1024_top256.jsonl")
DEFAULT_INPUT = Path(
    "configs/object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_listed_scales.json"
)
DEFAULT_OUTPUT = Path(
    "configs/object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_conf_gt_0p9_listed_scales.json"
)
COLLISION_KEYS = (
    "panda_hand_object_collision",
    "panda_hand_ground_collision",
    "panda_fingers_object_collision",
    "panda_fingers_ground_collision",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-jsonl", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON list: {path}")

    rows: list[dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object/scale mapping at {path}[{index}]")
        name = item.get("object")
        scale = item.get("scale")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Missing object name at {path}[{index}]")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or float(scale) <= 0.0:
            raise ValueError(f"Invalid scale at {path}[{index}]: {scale!r}")
        if name in seen:
            raise ValueError(f"Duplicate object in {path}: {name}")
        seen.add(name)
        rows.append(dict(item))
    return rows


def _qualifying_sets(
    report_path: Path,
    base_names: set[str],
    threshold: float,
) -> tuple[set[str], set[str], set[str]]:
    reported = set()
    high_conf_free = set()
    high_conf_colliding = set()

    with report_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("object")
            if name not in base_names:
                continue
            if bool(row.get("episode_success")):
                raise ValueError(
                    f"Base manifest object unexpectedly has episode_success=true at "
                    f"{report_path}:{line_number}: {name}"
                )
            reported.add(name)
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"Missing candidates at {report_path}:{line_number}")
            for candidate in candidates:
                confidence = float(candidate.get("confidence", float("-inf")))
                if confidence <= threshold:
                    continue
                if bool(candidate.get("hand_and_fingers_collision_free")):
                    high_conf_free.add(name)
                if bool(candidate.get("valid_se3")) and any(
                    bool(candidate.get(key)) for key in COLLISION_KEYS
                ):
                    high_conf_colliding.add(name)

    return reported, high_conf_free, high_conf_colliding


def refine_manifest(
    report_path: Path,
    input_path: Path,
    output_path: Path,
    threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"confidence threshold must be in [0, 1], got {threshold}")

    base_rows = _load_manifest(input_path)
    base_names = {str(row["object"]) for row in base_rows}
    reported, high_conf_free, high_conf_colliding = _qualifying_sets(
        report_path,
        base_names,
        threshold,
    )
    missing = base_names.difference(reported)
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"Report is missing {len(missing)} base-manifest objects; first missing objects: {preview}"
        )

    selected_names = high_conf_colliding.difference(high_conf_free)
    selected_rows = [row for row in base_rows if row["object"] in selected_names]
    removed_names = [str(row["object"]) for row in base_rows if row["object"] not in selected_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_report": str(report_path.resolve()),
        "source_manifest": str(input_path.resolve()),
        "output_manifest": str(output_path.resolve()),
        "selection_scope": "refine_source_manifest_only",
        "confidence_operator": ">",
        "confidence_threshold": threshold,
        "criteria": {
            "episode_success": False,
            "has_hand_and_fingers_collision_free_grasp_above_threshold": False,
            "has_valid_colliding_grasp_above_threshold": True,
        },
        "source_object_count": len(base_rows),
        "selected_object_count": len(selected_rows),
        "removed_object_count": len(removed_names),
        "high_conf_free_object_count": len(high_conf_free),
        "high_conf_colliding_object_count": len(high_conf_colliding),
        "removed_objects": removed_names,
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _args()
    summary = refine_manifest(
        args.report_jsonl.expanduser().resolve(),
        args.input_manifest.expanduser().resolve(),
        args.output_manifest.expanduser().resolve(),
        float(args.confidence_threshold),
    )
    print(
        f"[DONE] selected={summary['selected_object_count']}/"
        f"{summary['source_object_count']} removed={summary['removed_object_count']} "
        f"confidence>{summary['confidence_threshold']:g}"
    )
    print(f"[DONE] manifest={summary['output_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split failed eval objects into low-scale and high-scale name lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read eval_objects_summary.json and write two object-name-only files "
            "for failed objects split by evaluated scale."
        )
    )
    parser.add_argument("summary", type=Path, help="Path to eval_objects_summary.json.")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Objects with success_rate below this value are treated as failures.",
    )
    parser.add_argument(
        "--scale-threshold",
        type=float,
        default=0.16,
        help="Scale cutoff used to split failed objects.",
    )
    parser.add_argument(
        "--scale-field",
        choices=("scale_mean", "scale_min", "scale_max"),
        default="scale_mean",
        help="Per-object scale field used for the split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/eval_scale_split"),
        help="Directory where the two object-name files are written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the summary parent directory name.",
    )
    return parser.parse_args()


def _load_rows(summary_path: Path) -> list[dict[str, Any]]:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("per_object")
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path} does not contain a per_object list.")
    return rows


def _scale_value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    if field == "scale_mean":
        values = row.get("scale_values") or row.get("episode_scales")
        if isinstance(values, list) and values:
            numeric_values = [float(item) for item in values if isinstance(item, (int, float))]
            if numeric_values:
                return sum(numeric_values) / len(numeric_values)
    return None


def _object_group_key(name: str) -> str:
    if name.startswith("ddg-"):
        parts = name.split("_", 2)
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return name

    parts = name.split("-", 2)
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return name


def _write_grouped_names(path: Path, groups: dict[str, set[str]]) -> None:
    lines = []
    for key in sorted(groups):
        lines.append("\t".join(sorted(groups[key])))
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _object_count(groups: dict[str, set[str]]) -> int:
    return sum(len(names) for names in groups.values())


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.summary)

    low_scale: dict[str, set[str]] = {}
    high_scale: dict[str, set[str]] = {}
    missing_scale: list[str] = []
    equal_scale: list[str] = []

    for row in rows:
        name = row.get("name") or row.get("object")
        success_rate = row.get("success_rate")
        if not isinstance(name, str) or not isinstance(success_rate, (int, float)):
            continue
        if float(success_rate) >= args.success_threshold:
            continue

        scale = _scale_value(row, args.scale_field)
        group_key = _object_group_key(name)
        if scale is None:
            missing_scale.append(name)
        elif scale < args.scale_threshold:
            low_scale.setdefault(group_key, set()).add(name)
        elif scale > args.scale_threshold:
            high_scale.setdefault(group_key, set()).add(name)
        else:
            equal_scale.append(name)

    overlapping_groups = set(low_scale).intersection(high_scale)
    for group_key in sorted(overlapping_groups):
        low_scale[group_key].update(high_scale.pop(group_key))
    missing_scale.sort()
    equal_scale.sort()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.summary.parent.name
    low_path = args.output_dir / f"{prefix}_failed_scale_lt_{args.scale_threshold:g}.txt"
    high_path = args.output_dir / f"{prefix}_failed_scale_gt_{args.scale_threshold:g}.txt"
    _write_grouped_names(low_path, low_scale)
    _write_grouped_names(high_path, high_scale)

    print(f"summary: {args.summary}")
    print(f"failure criterion: success_rate < {args.success_threshold:g}")
    print(f"scale field: {args.scale_field}")
    print(f"scale split: < {args.scale_threshold:g} / > {args.scale_threshold:g}")
    print(f"overlapping groups assigned to low-scale file: {len(overlapping_groups)}")
    print(f"low-scale failure groups: {len(low_scale)} groups, {_object_count(low_scale)} objects -> {low_path}")
    print(f"high-scale failure groups: {len(high_scale)} groups, {_object_count(high_scale)} objects -> {high_path}")
    if equal_scale:
        print(f"equal-to-threshold failures skipped: {len(equal_scale)}")
    if missing_scale:
        print(f"missing-scale failures skipped: {len(missing_scale)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert grouped object text into a flat JSON object manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert whitespace/tab-separated object names into a JSON list."
    )
    parser.add_argument("input_txt", type=Path, help="Input text file with object names.")
    parser.add_argument("output_json", type=Path, help="Output JSON manifest path.")
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Keep first-seen order instead of sorting names.",
    )
    parser.add_argument(
        "--source-candidates",
        type=Path,
        default=None,
        help=(
            "Optional source candidates JSON with '<object>-<scale>' entries. "
            "Input names without scale suffix are expanded back to matching full candidates."
        ),
    )
    parser.add_argument(
        "--object-scale-records",
        action="store_true",
        help=(
            "Write {'object': <name>, 'scale': <scale>} records instead of "
            "'<object>-<scale>' strings. Requires candidates with scale suffixes."
        ),
    )
    parser.add_argument(
        "--input-object-scale-table",
        action="store_true",
        help=(
            "Parse a whitespace-delimited table with the exact header 'object scale' "
            "and write its listed scales directly as object-scale records."
        ),
    )
    return parser.parse_args()


def _load_names(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        for name in line.split():
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _expand_from_source_candidates(names: list[str], source_path: Path) -> list[str]:
    with source_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise ValueError(f"source candidates must be a JSON list of strings: {source_path}")

    by_base: dict[str, list[str]] = {}
    for item in candidates:
        if "-" not in item:
            continue
        base, _scale = item.rsplit("-", 1)
        by_base.setdefault(base, []).append(item)

    expanded: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in names:
        matches = by_base.get(name, [name])
        if matches == [name] and name not in candidates:
            missing.append(name)
            continue
        for match in matches:
            if match not in seen:
                expanded.append(match)
                seen.add(match)

    if missing:
        sample = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"Input names not found in source candidates: {sample}{suffix}")
    return expanded


def _object_scale_records(candidates: list[str]) -> list[dict[str, str | float]]:
    records: list[dict[str, str | float]] = []
    for candidate in candidates:
        if "-" not in candidate:
            raise ValueError(f"Candidate is missing a scale suffix: {candidate!r}")
        object_name, scale_text = candidate.rsplit("-", 1)
        try:
            scale = float(scale_text)
        except ValueError as exc:
            raise ValueError(f"Candidate has an invalid scale suffix: {candidate!r}") from exc
        if not object_name or scale <= 0.0:
            raise ValueError(f"Candidate has an invalid object or scale: {candidate!r}")
        records.append({"object": object_name, "scale": scale})
    return records


def _load_object_scale_table(path: Path) -> list[dict[str, str | float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].split() != ["object", "scale"]:
        raise ValueError(f"object-scale table must start with the exact header 'object scale': {path}")

    records: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(f"Expected exactly two columns at {path}:{line_number}: {line!r}")
        object_name, scale_text = columns
        if object_name in seen:
            raise ValueError(f"Duplicate object at {path}:{line_number}: {object_name!r}")
        try:
            scale = float(scale_text)
        except ValueError as exc:
            raise ValueError(f"Invalid scale at {path}:{line_number}: {scale_text!r}") from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Scale must be finite and positive at {path}:{line_number}: {scale!r}")
        records.append({"object": object_name, "scale": scale})
        seen.add(object_name)
    if not records:
        raise ValueError(f"No object-scale rows found in {path}")
    return records


def main() -> None:
    args = _parse_args()
    if args.input_object_scale_table:
        if args.source_candidates is not None or args.object_scale_records:
            raise ValueError(
                "--input-object-scale-table cannot be combined with "
                "--source-candidates or --object-scale-records"
            )
        output = _load_object_scale_table(args.input_txt)
        if not args.preserve_order:
            output.sort(key=lambda record: str(record["object"]))
    else:
        names = _load_names(args.input_txt)
        if args.source_candidates is not None:
            names = _expand_from_source_candidates(names, args.source_candidates)
        if not args.preserve_order:
            names = sorted(names)
        if not names:
            raise ValueError(f"No object names found in {args.input_txt}")
        output = _object_scale_records(names) if args.object_scale_records else names

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} objects to {args.output_json}")


if __name__ == "__main__":
    main()

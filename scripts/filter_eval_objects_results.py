#!/usr/bin/env python3
"""Filter an existing per-object evaluation summary by an object manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def filter_results(source_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Expected a non-empty object manifest: {manifest_path}")
    names = {
        item["object"] if isinstance(item, dict) else str(item).rsplit("-", 1)[0]
        for item in manifest
    }
    rows = source.get("per_object")
    if not isinstance(rows, list):
        raise ValueError(f"Source summary has no per_object list: {source_path}")
    rows_by_name = {str(row["name"]): row for row in rows}
    missing = names.difference(rows_by_name)
    if missing:
        raise ValueError(
            f"Source evaluation is missing {len(missing)} selected objects; "
            f"first missing objects: {sorted(missing)[:10]}"
        )

    selected = [rows_by_name[name] for name in sorted(names)]
    episodes = sum(int(row["episodes"]) for row in selected)
    successes = sum(int(row["successes"]) for row in selected)
    payload = {
        **{key: value for key, value in source.items() if key != "per_object"},
        "derived_from_eval_summary": str(source_path.resolve()),
        "filter_object_manifest": str(manifest_path.resolve()),
        "result_kind": "filtered_existing_evaluation",
        "source_object_count": len(rows),
        "objects": len(selected),
        "episodes": episodes,
        "successes": successes,
        "success_rate": float(successes) / float(episodes) if episodes else 0.0,
        "per_object": selected,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_objects_summary.json"
    csv_path = output_dir / "eval_objects_per_object.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "object",
                "episodes",
                "successes",
                "success_rate",
                "scale_mean",
                "scale_min",
                "scale_max",
                "ranks",
            ]
        )
        for row in selected:
            writer.writerow(
                [
                    row["name"],
                    row["episodes"],
                    row["successes"],
                    row["success_rate"],
                    row.get("scale_mean"),
                    row.get("scale_min"),
                    row.get("scale_max"),
                    " ".join(str(rank) for rank in row.get("ranks", [])),
                ]
            )
    return payload


def main() -> int:
    args = _args()
    result = filter_results(
        args.source_summary.expanduser().resolve(),
        args.object_manifest.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(
        f"[DONE] objects={result['objects']} episodes={result['episodes']} "
        f"successes={result['successes']} success_rate={result['success_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

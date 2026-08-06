#!/usr/bin/env python3
"""Summarize the completed 2x2 PointNet offline ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts/probes/pointnet_normalization_bottleneck_ablation"
)
VARIANTS = (
    "normalized_direct128",
    "normalized_rank10",
    "unnormalized_direct128",
    "unnormalized_rank10",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows = []
    for variant in VARIANTS:
        path = root / variant / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"ablation variant is incomplete: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = max(payload["history"], key=lambda item: item["r2"])
        rows.append(
            {
                "variant": variant,
                "normalized": payload["normalized"],
                "rank10_bottleneck": payload["rank10_bottleneck"],
                "best_epoch": best["epoch"],
                "validation_r2": best["r2"],
                "validation_cosine": best["mean_cosine"],
            }
        )
    summary = {
        "schema_version": "pointnet_normalization_bottleneck_ablation_summary_v1",
        "variants": rows,
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("variant                              val_r2    val_cos")
    for row in sorted(rows, key=lambda item: item["validation_r2"], reverse=True):
        print(
            f"{row['variant']:<36} "
            f"{row['validation_r2']:.6f}  {row['validation_cosine']:.6f}"
        )
    print(f"[offline-ablation] wrote {path}")


if __name__ == "__main__":
    main()

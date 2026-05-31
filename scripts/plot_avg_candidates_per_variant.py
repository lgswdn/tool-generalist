#!/usr/bin/env python3
"""Plot average pre-stabilize candidate count per tool variant."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CANDIDATE_CSV = Path(
    "outputs/contact_counts/tool_contact_candidate_counts_before_stabilize.csv"
)


@dataclass
class ToolRow:
    tool_key: str
    tool_label: str
    variants: int
    candidate_cases: int

    @property
    def avg_candidates_per_variant(self) -> float:
        if self.variants <= 0:
            return 0.0
        return float(self.candidate_cases) / float(self.variants)


def load_rows(path: Path) -> list[ToolRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"tool_key", "tool_label", "variants", "contact_cases"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [
            ToolRow(
                tool_key=row["tool_key"],
                tool_label=row["tool_label"],
                variants=int(row["variants"]),
                candidate_cases=int(row["contact_cases"]),
            )
            for row in reader
        ]
    return sorted(rows, key=lambda row: (-row.avg_candidates_per_variant, -row.candidate_cases, row.tool_key))


def write_csv(path: Path, rows: list[ToolRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tool_key",
                "tool_label",
                "variants",
                "candidate_cases",
                "avg_candidates_per_variant",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.tool_key,
                    row.tool_label,
                    row.variants,
                    row.candidate_cases,
                    f"{row.avg_candidates_per_variant:.6f}",
                ]
            )


def plot(rows: list[ToolRow], output: Path, *, dpi: int, top_n: int | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache = output.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.fspath(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = rows[:top_n] if top_n is not None else rows
    labels = [row.tool_label.replace("_", " ") for row in plot_rows]
    values = [row.avg_candidates_per_variant for row in plot_rows]

    height = max(7.0, min(30.0, 0.28 * len(plot_rows) + 2.0))
    fig, ax = plt.subplots(figsize=(14.0, height))
    y_positions = range(len(plot_rows))
    ax.barh(y_positions, values, color="#54A24B")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average geometry-qualified candidates per tool variant")
    ax.set_title("Average Pre-stabilize Candidates per Tool Variant")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    max_value = max(values) if values else 0.0
    ax.set_xlim(0.0, max_value * 1.08 if max_value > 0 else 1.0)
    pad = max(max_value * 0.006, 1.0)
    for y, value in zip(y_positions, values):
        ax.text(value + pad, y, f"{value:,.0f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average candidate count per tool variant.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/contact_counts/tool_avg_candidates_per_variant.png"),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    candidate_csv = args.candidate_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    csv_output = output.with_suffix(".csv") if args.csv is None else args.csv.expanduser().resolve()

    rows = load_rows(candidate_csv)
    write_csv(csv_output, rows)
    plot(rows, output, dpi=args.dpi, top_n=args.top_n)

    total_candidates = sum(row.candidate_cases for row in rows)
    total_variants = sum(row.variants for row in rows)
    overall_avg = float(total_candidates) / float(total_variants) if total_variants else 0.0
    print(f"Candidate CSV: {candidate_csv}")
    print(f"Base tools: {len(rows)}")
    print(f"Tool variants: {total_variants}")
    print(f"Candidate cases: {total_candidates}")
    print(f"Overall avg candidates per variant: {overall_avg:.2f}")
    print(f"Saved plot: {output}")
    print(f"Saved CSV: {csv_output}")


if __name__ == "__main__":
    main()

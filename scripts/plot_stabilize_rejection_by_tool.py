#!/usr/bin/env python3
"""Plot rejection rate from existing contact-count CSVs.

This temporary helper intentionally does not scan the artifact directory. It
uses:

* candidate CSV: pre-stabilize geometry-qualified counts
* final CSV: final successful contact counts

The plotted rejection rate is `(candidate_cases - final_success_cases) /
candidate_cases`, grouped by base tool.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CANDIDATE_CSV = Path(
    "outputs/contact_counts/tool_contact_candidate_counts_before_stabilize.csv"
)
DEFAULT_FINAL_CSV = Path("outputs/contact_counts/tool_contact_success_counts.csv")


@dataclass
class ToolRow:
    tool_key: str
    tool_label: str
    variants: int
    candidate_cases: int = 0
    final_success_cases: int = 0

    @property
    def rejected_cases(self) -> int:
        return max(self.candidate_cases - self.final_success_cases, 0)

    @property
    def rejection_rate(self) -> float:
        if self.candidate_cases <= 0:
            return 0.0
        return float(self.rejected_cases) / float(self.candidate_cases)


def _read_count_csv(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"tool_key", "tool_label", "variants"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if "contact_cases" not in (reader.fieldnames or []) and "successful_contacts" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain either contact_cases or successful_contacts")
        return {row["tool_key"]: row for row in reader}


def _count_value(row: dict) -> int:
    if "contact_cases" in row and row["contact_cases"] != "":
        return int(row["contact_cases"])
    return int(row["successful_contacts"])


def load_rows(candidate_csv: Path, final_csv: Path) -> list[ToolRow]:
    candidate_rows = _read_count_csv(candidate_csv)
    final_rows = _read_count_csv(final_csv)
    rows: list[ToolRow] = []

    for tool_key, cand in candidate_rows.items():
        final = final_rows.get(tool_key)
        rows.append(
            ToolRow(
                tool_key=tool_key,
                tool_label=cand["tool_label"],
                variants=int(cand["variants"]),
                candidate_cases=_count_value(cand),
                final_success_cases=_count_value(final) if final is not None else 0,
            )
        )

    return sorted(rows, key=lambda row: (-row.rejection_rate, -row.candidate_cases, row.tool_key))


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
                "final_success_cases",
                "rejected_cases",
                "rejection_rate",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.tool_key,
                    row.tool_label,
                    row.variants,
                    row.candidate_cases,
                    row.final_success_cases,
                    row.rejected_cases,
                    f"{row.rejection_rate:.8f}",
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
    rates = [row.rejection_rate * 100.0 for row in plot_rows]

    height = max(7.0, min(30.0, 0.28 * len(plot_rows) + 2.0))
    fig, ax = plt.subplots(figsize=(14.0, height))
    y_positions = range(len(plot_rows))
    ax.barh(y_positions, rates, color="#E45756")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Rejection rate from candidate to final success (%)")
    ax.set_title("Contact Rejection Rate per Tool")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for y, rate in zip(y_positions, rates):
        ax.text(min(rate + 0.6, 99.0), y, f"{rate:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rejection rate from existing contact-count CSVs.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/contact_counts/tool_candidate_to_final_rejection_rate.png"),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    candidate_csv = args.candidate_csv.expanduser().resolve()
    final_csv = args.final_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    csv_output = output.with_suffix(".csv") if args.csv is None else args.csv.expanduser().resolve()

    rows = load_rows(candidate_csv, final_csv)
    write_csv(csv_output, rows)
    plot(rows, output, dpi=args.dpi, top_n=args.top_n)

    total_candidates = sum(row.candidate_cases for row in rows)
    total_final = sum(row.final_success_cases for row in rows)
    total_rejected = max(total_candidates - total_final, 0)
    overall_rate = float(total_rejected) / float(total_candidates) if total_candidates else 0.0
    print(f"Candidate CSV: {candidate_csv}")
    print(f"Final CSV: {final_csv}")
    print(f"Base tools: {len(rows)}")
    print(f"Candidate cases: {total_candidates}")
    print(f"Final success cases: {total_final}")
    print(f"Rejected cases: {total_rejected}")
    print(f"Overall rejection rate: {overall_rate * 100.0:.2f}%")
    print(f"Saved plot: {output}")
    print(f"Saved CSV: {csv_output}")


if __name__ == "__main__":
    main()

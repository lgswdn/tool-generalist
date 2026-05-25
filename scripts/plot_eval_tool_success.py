#!/usr/bin/env python3
"""Plot per-base-tool success rates from eval_tools results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


def _base_tool_name(tool_name: str) -> str:
    match = re.search(r"_(?:var|variant)_", tool_name)
    if match:
        return tool_name[: match.start()]
    match = re.search(r"(?:^|[_-])variant(?:[_-]|\d|$)", tool_name)
    if match:
        return tool_name[: match.start()].rstrip("_-")
    return tool_name


def _read_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "episodes", "successes"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "name": row["name"],
                    "episodes": int(row["episodes"]),
                    "successes": int(row["successes"]),
                }
            )
    return rows


def _read_summary_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    per_tool = payload.get("per_tool")
    if not isinstance(per_tool, list):
        raise ValueError(f"{path} does not contain a per_tool list")
    rows = []
    for row in per_tool:
        rows.append(
            {
                "name": str(row["name"]),
                "episodes": int(row["episodes"]),
                "successes": int(row["successes"]),
            }
        )
    return rows


def _load_eval_rows(result_path: Path) -> list[dict]:
    if result_path.is_file():
        if result_path.suffix == ".json":
            return _read_summary_rows(result_path)
        if result_path.suffix == ".csv":
            return _read_csv_rows(result_path)
        raise ValueError(f"{result_path} is not a supported eval result file")

    prefixes = ("eval_tools", "eval_tools_steps")
    for prefix in prefixes:
        per_tool_csv = result_path / f"{prefix}_per_tool.csv"
        if per_tool_csv.is_file():
            return _read_csv_rows(per_tool_csv)

        summary_json = result_path / f"{prefix}_summary.json"
        if summary_json.is_file():
            return _read_summary_rows(summary_json)

        rank_paths = sorted(Path(path) for path in glob.glob(str(result_path / f"{prefix}_rank_*.csv")))
        if rank_paths:
            rows = []
            for path in rank_paths:
                rows.extend(_read_csv_rows(path))
            return rows

    expected = []
    for prefix in prefixes:
        expected.extend([f"{prefix}_per_tool.csv", f"{prefix}_summary.json", f"{prefix}_rank_*.csv"])
    raise FileNotFoundError(f"No eval result files found in {result_path}. Expected one of: {', '.join(expected)}.")


def _aggregate_by_base_tool(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        base_name = _base_tool_name(str(row["name"]))
        item = grouped.setdefault(base_name, {"name": base_name, "episodes": 0, "successes": 0, "variants": 0})
        item["episodes"] += int(row["episodes"])
        item["successes"] += int(row["successes"])
        item["variants"] += 1

    aggregated = []
    for item in grouped.values():
        episodes = int(item["episodes"])
        successes = int(item["successes"])
        item["success_rate"] = float(successes) / float(episodes) if episodes > 0 else 0.0
        aggregated.append(item)
    aggregated.sort(key=lambda item: (-item["success_rate"], -item["episodes"], item["name"]))
    return aggregated


def _write_aggregated_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "variants", "episodes", "successes", "success_rate"])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["variants"],
                    row["episodes"],
                    row["successes"],
                    f"{row['success_rate']:.8f}",
                ]
            )


def _plot(rows: list[dict], output: Path, title: str, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required: install it in the eval environment first.") from exc

    if not rows:
        raise ValueError("No rows to plot")

    names = [row["name"] for row in rows]
    rates = [row["success_rate"] * 100.0 for row in rows]

    width = max(10.0, min(48.0, 0.38 * len(rows)))
    fig, ax = plt.subplots(figsize=(width, 6.5))
    ax.bar(range(len(rows)), rates, color="#4C78A8")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.margins(x=0.005)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot success rates grouped by base tool name from eval_tools outputs."
    )
    parser.add_argument("result_path", type=Path, help="Directory or file containing eval result data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <result_dir>/eval_tools_base_tool_success.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV for aggregated base-tool success rates.",
    )
    parser.add_argument("--title", default="Base Tool Success Rate", help="Plot title.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    args = parser.parse_args()

    result_path = args.result_path.expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result_dir = result_path if result_path.is_dir() else result_path.parent

    output = args.output
    if output is None:
        output = result_dir / "eval_tools_base_tool_success.png"
    else:
        output = output.expanduser().resolve()

    rows = _aggregate_by_base_tool(_load_eval_rows(result_path))
    _plot(rows, output, args.title, args.dpi)

    csv_output = args.csv
    if csv_output is not None:
        _write_aggregated_csv(csv_output.expanduser().resolve(), rows)

    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    overall = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(f"Loaded base tools: {len(rows)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {overall * 100.0:.2f}%")
    print(f"Saved plot: {os.fspath(output)}")
    if csv_output is not None:
        print(f"Saved CSV: {os.fspath(csv_output.expanduser().resolve())}")


if __name__ == "__main__":
    main()

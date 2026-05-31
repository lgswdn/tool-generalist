#!/usr/bin/env python3
"""Plot contact case counts grouped by base tool.

The input is a contact artifact directory whose immediate children are tool
variant directories, for example:

    000_ball_peen_hammer_end_effector_var_000/

Final contact artifacts have sibling manifests named:

    <object_id>.pt.manifest.json

Geometry candidates before stabilization have sibling manifests named:

    <object_id>.pt.candidate.manifest.json

The script can sum either final `num_contacts` or pre-stabilization
`num_candidates` across all variants of the same base tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONTACT_ROOT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_full_tool/"
    "281987b90b894c5a84c97b9b0c89bca2d8711036c52e2d2b3f7f0a65f7d94535"
)

VARIANT_RE = re.compile(r"^(?P<base>.+)_end_effector_var_(?P<variant>\d+)$")
TOOL_INDEX_RE = re.compile(r"^\d{3}_(?P<label>.+)$")


@dataclass
class ToolCount:
    tool_key: str
    tool_label: str
    variants_seen: set[str]
    manifest_files: int = 0
    contact_cases: int = 0

    @property
    def variants(self) -> int:
        return len(self.variants_seen)


@dataclass(frozen=True)
class StageSpec:
    name: str
    value_field: str
    manifest_globs: tuple[str, ...]
    default_output_name: str
    default_title: str
    x_label: str


STAGES = {
    "final": StageSpec(
        name="final",
        value_field="num_contacts",
        manifest_globs=(
            "*.pt.manifest.json",
            "!*.candidate.manifest.json",
            "!*.stabilized_success.manifest.json",
        ),
        default_output_name="tool_contact_success_counts.png",
        default_title="Successful Contact Cases per Tool",
        x_label="Successful contact cases",
    ),
    "candidate": StageSpec(
        name="candidate",
        value_field="num_candidates",
        manifest_globs=("*.candidate.manifest.json",),
        default_output_name="tool_contact_candidate_counts_before_stabilize.png",
        default_title="Geometry-qualified Contact Cases per Tool (Before Stabilize)",
        x_label="Geometry-qualified contact candidates before stabilize",
    ),
}


def _variant_parts(dirname: str) -> tuple[str, str] | None:
    match = VARIANT_RE.match(dirname)
    if match is None:
        return None
    return match.group("base"), match.group("variant")


def _tool_label(tool_key: str, *, strip_index: bool) -> str:
    if not strip_index:
        return tool_key
    match = TOOL_INDEX_RE.match(tool_key)
    return match.group("label") if match else tool_key


def _is_stage_manifest(path: Path, stage: StageSpec) -> bool:
    name = path.name
    if stage.name == "candidate":
        return name.endswith(".candidate.manifest.json")
    if stage.name == "final":
        return (
            name.endswith(".pt.manifest.json")
            and not name.endswith(".candidate.manifest.json")
            and not name.endswith(".stabilized_success.manifest.json")
        )
    raise ValueError(f"Unsupported stage: {stage.name}")


def _iter_stage_manifests(root: Path, stage: StageSpec) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if _is_stage_manifest(path, stage):
                yield path


def _collect_counts_with_ripgrep(
    root: Path,
    counts: dict[str, ToolCount],
    *,
    stage: StageSpec,
    strip_index: bool,
) -> bool:
    """Fast path for large NFS directories.

    `rg` is much faster than Python JSON parsing for the common case because
    manifests only need one numeric field line.
    """

    if shutil.which("rg") is None:
        return False

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        f'"{stage.value_field}"',
        os.fspath(root),
    ]
    for glob_pattern in stage.manifest_globs:
        cmd.extend(["--glob", glob_pattern])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        print("rg scan failed; falling back to Python manifest parsing.")
        if result.stderr:
            print(result.stderr.strip())
        return False

    line_re = re.compile(rf'"{re.escape(stage.value_field)}"\s*:\s*(\d+)')
    for raw_line in result.stdout.splitlines():
        path_text, _, rest = raw_line.partition(":")
        _, _, json_line = rest.partition(":")
        match = line_re.search(json_line)
        if match is None:
            continue

        manifest_path = Path(path_text)
        parts = _variant_parts(manifest_path.parent.name)
        if parts is None:
            continue

        tool_key, variant = parts
        item = counts.setdefault(
            tool_key,
            ToolCount(
                tool_key=tool_key,
                tool_label=_tool_label(tool_key, strip_index=strip_index),
                variants_seen=set(),
            ),
        )
        item.variants_seen.add(variant)
        item.manifest_files += 1
        item.contact_cases += int(match.group(1))

    return True


def collect_counts(root: Path, *, stage: StageSpec, strip_index: bool) -> list[ToolCount]:
    if not root.is_dir():
        raise NotADirectoryError(root)

    counts: dict[str, ToolCount] = {}

    for child in root.iterdir():
        if not child.is_dir():
            continue
        parts = _variant_parts(child.name)
        if parts is None:
            continue
        tool_key, variant = parts
        item = counts.setdefault(
            tool_key,
            ToolCount(
                tool_key=tool_key,
                tool_label=_tool_label(tool_key, strip_index=strip_index),
                variants_seen=set(),
            ),
        )
        item.variants_seen.add(variant)

    skipped = 0
    used_rg = _collect_counts_with_ripgrep(root, counts, stage=stage, strip_index=strip_index)
    if not used_rg:
        for manifest_path in _iter_stage_manifests(root, stage):
            parts = _variant_parts(manifest_path.parent.name)
            if parts is None:
                skipped += 1
                continue

            tool_key, variant = parts
            item = counts.setdefault(
                tool_key,
                ToolCount(
                    tool_key=tool_key,
                    tool_label=_tool_label(tool_key, strip_index=strip_index),
                    variants_seen=set(),
                ),
            )
            item.variants_seen.add(variant)

            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            if stage.name == "final" and manifest.get("status") != "complete":
                skipped += 1
                continue

            item.manifest_files += 1
            item.contact_cases += int(manifest.get(stage.value_field, 0))

    rows = sorted(
        counts.values(),
        key=lambda row: (-row.contact_cases, -row.manifest_files, row.tool_key),
    )
    if skipped:
        print(f"Skipped manifests: {skipped}")
    return rows


def write_csv(path: Path, rows: list[ToolCount], *, stage: StageSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "tool_key", "tool_label", "variants", "manifest_files", "contact_cases"])
        for row in rows:
            writer.writerow(
                [
                    stage.name,
                    row.tool_key,
                    row.tool_label,
                    row.variants,
                    row.manifest_files,
                    row.contact_cases,
                ]
            )


def plot_counts(
    rows: list[ToolCount],
    output: Path,
    *,
    title: str,
    x_label: str,
    dpi: int,
    top_n: int | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache = output.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.fspath(mpl_cache))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required. Try: conda activate isaac") from exc

    if top_n is not None:
        rows = rows[:top_n]
    if not rows:
        raise ValueError("No tool rows to plot")

    labels = [row.tool_label.replace("_", " ") for row in rows]
    counts = [row.contact_cases for row in rows]

    height = max(7.0, min(30.0, 0.28 * len(rows) + 2.0))
    fig, ax = plt.subplots(figsize=(14.0, height))
    y_positions = range(len(rows))
    ax.barh(y_positions, counts, color="#4C78A8")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    max_count = max(counts)
    text_pad = max(max_count * 0.006, 1.0)
    ax.set_xlim(0, max_count * 1.08)
    for y, count in zip(y_positions, counts):
        ax.text(count + text_pad, y, f"{count:,}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot contact case counts grouped by base tool."
    )
    parser.add_argument(
        "contact_root",
        type=Path,
        nargs="?",
        default=DEFAULT_CONTACT_ROOT,
        help="Contact artifact root. Defaults to the fork_sdf/contact_gen_full_tool artifact inspected in this repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to outputs/contact_counts/<stage-specific-name>.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to the PNG path with .csv suffix.",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        default="final",
        help="final sums final num_contacts; candidate sums pre-stabilize geometry num_candidates.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=None, help="Only plot the top N tools.")
    parser.add_argument(
        "--keep-tool-index",
        action="store_true",
        help="Keep the leading numeric tool index in plot labels.",
    )
    args = parser.parse_args()

    contact_root = args.contact_root.expanduser().resolve()
    stage = STAGES[args.stage]
    output = (
        (Path("outputs/contact_counts") / stage.default_output_name).resolve()
        if args.output is None
        else args.output.expanduser().resolve()
    )
    csv_output = output.with_suffix(".csv") if args.csv is None else args.csv.expanduser().resolve()
    title = stage.default_title if args.title is None else args.title

    rows = collect_counts(contact_root, stage=stage, strip_index=not args.keep_tool_index)
    plot_counts(rows, output, title=title, x_label=stage.x_label, dpi=args.dpi, top_n=args.top_n)
    write_csv(csv_output, rows, stage=stage)

    total_contacts = sum(row.contact_cases for row in rows)
    total_manifest_files = sum(row.manifest_files for row in rows)
    total_variants = sum(row.variants for row in rows)
    print(f"Contact root: {contact_root}")
    print(f"Stage: {stage.name}")
    print(f"Base tools: {len(rows)}")
    print(f"Tool variants: {total_variants}")
    print(f"Manifest files: {total_manifest_files}")
    print(f"Contact cases: {total_contacts}")
    print(f"Saved plot: {output}")
    print(f"Saved CSV: {csv_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render many geometry-only contact candidates into one headless PNG sheet."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
import sys
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/tool_generalist_matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing tool/*.candidate.pt files.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--same-tool",
        action="store_true",
        help="Sample all panels from one tool instead of using distinct tools.",
    )
    parser.add_argument("--tool-filter", default="", help="Optional tool-directory substring.")
    parser.add_argument("--max-faces", type=int, default=3500)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from pretrain.dataset import NewPretrainDataset
    from contact_generation.visualize_post_patch import (
        _add_mesh,
        _axis_center_radius,
        _subsample_faces,
    )

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Contact candidate directory does not exist: {data_dir}")
    output = Path(args.output).expanduser().resolve()
    count = int(args.rows) * int(args.cols)
    rng = random.Random(int(args.seed))
    selected_paths = _select_candidate_files(
        data_dir,
        count=count,
        rng=rng,
        same_tool=bool(args.same_tool),
        tool_filter=str(args.tool_filter),
    )
    print(
        f"[contact-candidate-sheet] selected files={len(selected_paths)} "
        f"from data_dir={data_dir}",
        flush=True,
    )

    dataset = NewPretrainDataset(
        selected_paths,
        augment=False,
        require_movement=False,
        num_points=512,
        num_precontact_steps=0,
        include_meshes=True,
        max_contacts_per_file=0,
    )
    indices_by_path: dict[str, list[int]] = {}
    for dataset_i, (pt_path, _candidate_i) in enumerate(dataset._index):
        indices_by_path.setdefault(str(pt_path), []).append(dataset_i)

    selected_items: list[dict[str, Any]] = []
    for pt_path in selected_paths:
        indices = indices_by_path[str(pt_path)]
        dataset_i = rng.choice(indices)
        item = dataset[dataset_i]
        selected_items.append(_materialize_viz(item, torch))

    fig = plt.figure(
        figsize=(2.65 * int(args.cols), 2.55 * int(args.rows)),
        dpi=int(args.dpi),
        facecolor="white",
    )
    records: list[dict[str, Any]] = []
    for panel_i, viz in enumerate(selected_items):
        ax = fig.add_subplot(
            int(args.rows),
            int(args.cols),
            panel_i + 1,
            projection="3d",
        )
        object_faces = _subsample_faces(viz["object_faces"], int(args.max_faces))
        tool_faces = _subsample_faces(viz["tool_faces"], int(args.max_faces))
        _add_mesh(
            ax,
            viz["object_vertices"],
            object_faces,
            color=(0.12, 0.42, 0.88),
            alpha=0.48,
        )
        _add_mesh(
            ax,
            viz["tool_vertices"],
            tool_faces,
            color=(0.96, 0.36, 0.06),
            alpha=0.65,
        )
        closest = np.stack((viz["closest_tool"], viz["closest_object"]))
        ax.plot(
            closest[:, 0],
            closest[:, 1],
            closest[:, 2],
            color=(0.88, 0.05, 0.58),
            linewidth=1.2,
        )
        ax.scatter(
            closest[:, 0],
            closest[:, 1],
            closest[:, 2],
            color=[(0.88, 0.05, 0.58)],
            s=7,
            depthshade=False,
        )
        xyz = np.concatenate((viz["object_vertices"], viz["tool_vertices"]), axis=0)
        center, radius = _axis_center_radius(xyz, margin=0.60)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title(
            f"{_short_id(viz['tool_id'], 25)}\n"
            f"{_short_id(viz['object_id'], 25)}  "
            f"c{viz['candidate_index']}  sample gap={viz['sample_gap_m'] * 1000:.2f} mm",
            fontsize=5.3,
            pad=-1.5,
        )
        records.append(
            {
                "tool_id": viz["tool_id"],
                "object_id": viz["object_id"],
                "candidate_index": viz["candidate_index"],
                "sample_gap_m": viz["sample_gap_m"],
                "candidate_path": viz["candidate_path"],
            }
        )

    fig.suptitle(
        "Nonpenetrating contact candidates — blue: object, orange: gripper, "
        "magenta: closest sampled surface pair",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.965,
        wspace=-0.20,
        hspace=-0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "contact_candidate_sheet_v1",
                "data_dir": str(data_dir),
                "seed": int(args.seed),
                "rows": int(args.rows),
                "cols": int(args.cols),
                "note": (
                    "sample_gap_m is the nearest distance between the two stored "
                    "512-point surface clouds; it is a visualization aid, not the "
                    "mesh-SDF penetration audit."
                ),
                "cases": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[contact-candidate-sheet] wrote {output}", flush=True)
    print(f"[contact-candidate-sheet] wrote {manifest_path}", flush=True)
    return 0


def _select_candidate_files(
    data_dir: Path,
    *,
    count: int,
    rng: random.Random,
    same_tool: bool,
    tool_filter: str,
) -> list[Path]:
    tool_dirs = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_dir() and (not tool_filter or tool_filter in path.name)
    ]
    if not tool_dirs:
        raise RuntimeError(
            f"No matching tool directories under {data_dir}; filter={tool_filter!r}"
        )
    if same_tool:
        tool_dir = rng.choice(tool_dirs)
        files = sorted(tool_dir.glob("*.candidate.pt"))
        if not files:
            raise RuntimeError(f"No candidate files under {tool_dir}")
        return [rng.choice(files) for _ in range(count)]

    rng.shuffle(tool_dirs)
    selected: list[Path] = []
    for tool_dir in tool_dirs:
        files = sorted(tool_dir.glob("*.candidate.pt"))
        if files:
            selected.append(rng.choice(files))
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Requested {count} distinct tools but found only {len(selected)} "
            "with candidate files"
        )
    return selected


def _materialize_viz(item: dict[str, Any], torch_module) -> dict[str, Any]:
    object_rotation = item["object_rotation_E"]
    object_translation = item["object_bbox_center_E"]
    tool_rotation = item["contact_tool_rotation_E"]
    tool_translation = item["contact_tool_translation_E"]
    object_vertices = (
        item["object_mesh_vertices"] @ object_rotation.transpose(0, 1)
        + object_translation.reshape(1, 3)
    )
    tool_vertices = (
        item["tool_mesh_vertices"] @ tool_rotation.transpose(0, 1)
        + tool_translation.reshape(1, 3)
    )
    object_points = (
        item["object_points_O"] @ object_rotation.transpose(0, 1)
        + object_translation.reshape(1, 3)
    )
    tool_points = (
        item["tool_points_T"] @ tool_rotation.transpose(0, 1)
        + tool_translation.reshape(1, 3)
    )
    distances = torch_module.cdist(tool_points, object_points)
    flat_i = int(distances.argmin())
    object_count = int(object_points.shape[0])
    tool_i, object_i = divmod(flat_i, object_count)
    return {
        "object_vertices": object_vertices.detach().cpu().numpy(),
        "object_faces": item["object_mesh_faces"].detach().cpu().numpy(),
        "tool_vertices": tool_vertices.detach().cpu().numpy(),
        "tool_faces": item["tool_mesh_faces"].detach().cpu().numpy(),
        "closest_tool": tool_points[tool_i].detach().cpu().numpy(),
        "closest_object": object_points[object_i].detach().cpu().numpy(),
        "sample_gap_m": float(distances[tool_i, object_i]),
        "tool_id": str(item["tool_id"]),
        "object_id": str(item["object_id"]),
        "candidate_index": int(item["contact_index"]),
        "candidate_path": str(item["pt_path"]),
    }


def _short_id(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= int(limit):
        return text
    return text[: max(1, int(limit) - 1)] + "…"


if __name__ == "__main__":
    raise SystemExit(main())

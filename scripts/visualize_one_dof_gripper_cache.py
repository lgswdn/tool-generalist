#!/usr/bin/env python3
"""Visual proof for canonical one-DoF gripper clouds and aperture motion."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tool_generalist_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_graspgenx_grippers import _revolute_inward_extent
from scripts.render_generated_gripper_contact_sheet import (
    _one_dof_gripper_named_polygons,
)
from utils.assets import load_one_dof_gripper_manifest
from utils.geometry.gripper_cloud_cache import (
    cache_path_for_asset,
    load_gripper_cloud_cache,
)
from utils.geometry.one_dof_gripper_kinematics import (
    one_dof_body_poses,
    transform_points,
)


DEFAULT_MANIFEST = ROOT / "gripper/generated_graspgenx_matched_128/two_finger_revolute.json"
DEFAULT_CONTACT_ASSETS = (
    ROOT / "configs/generated_gripper_contact_assets_general_128/tools_adjusted.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/visualization/revolute_128_cache_proof"
STATE_FRACTIONS = (0.0, 0.5, 1.0)


def _body_color(name: str) -> str:
    if name.startswith("left_"):
        return "#1677ff"
    if name.startswith("right_"):
        return "#f59e0b"
    return "#6b7280"


def _set_limits(axis, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float((high - low).max()) * 0.56, 0.045)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _draw_state(
    axis,
    entry: dict,
    asset,
    manifest_dir: Path,
    cache,
    fraction: float,
) -> None:
    named = _one_dof_gripper_named_polygons(entry, fraction, manifest_dir)
    for body_name, polygons, _ in named:
        if polygons:
            axis.add_collection3d(
                Poly3DCollection(
                    polygons,
                    facecolor=_body_color(body_name),
                    edgecolor="none",
                    alpha=0.18,
                )
            )
    cloud = cache.cloud_at_fraction(fraction).numpy()
    body_index = cache.point_body_index.numpy()
    for index, body_name in enumerate(cache.body_names):
        rows = body_index == index
        axis.scatter(
            cloud[rows, 0],
            cloud[rows, 1],
            cloud[rows, 2],
            s=4.0,
            c=_body_color(body_name),
            alpha=0.9,
            depthshade=False,
        )
    poses = one_dof_body_poses(
        asset, fraction, device="cpu", dtype=torch.float64
    )
    tip_length = (
        float(asset.params["tip_length"])
        if asset.params["tip_shape"] != "none"
        else 0.0
    )
    local_tip = torch.tensor(
        [[0.0, 0.0, float(asset.params["top_size"][2]) + tip_length]],
        dtype=torch.float64,
    )
    interaction_center = torch.stack(
        [
            transform_points(local_tip, poses[body_name])[0]
            for body_name in ("left_top_link", "right_top_link")
        ]
    ).mean(dim=0).numpy()
    axis.scatter(
        *interaction_center,
        s=70,
        c="#ef4444",
        marker="X",
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
    )
    all_mesh = [polygon for _, polygons, _ in named for polygon in polygons]
    _set_limits(axis, np.concatenate(all_mesh + [cloud], axis=0))
    axis.view_init(elev=18, azim=-48)
    axis.set_axis_off()


def _surface_gap(entry: dict, fraction: float) -> float:
    params = entry["params"]
    mid = params["mid_size"]
    top = params["top_size"]
    outer = params["outer_size"]
    travel = float(params["travel_angle_rad"]) * (1.0 - fraction)
    mid_angle = float(params["open_angle_rad"]) + travel
    top_angle = (
        0.0
        if params["closure_mode"] == "parallel_tip"
        else float(params["open_angle_rad"]) + 2.0 * travel
    )
    extent = _revolute_inward_extent(
        length_scale=1.0,
        mid_length=float(mid[2]),
        top_length=float(top[2]),
        tip_length=float(params["tip_length"]),
        mid_y=float(mid[1]),
        top_y=float(top[1]),
        tip_width=float(params["tip_width"]),
        tip_shape=str(params["tip_shape"]),
        add_outer=bool(params["has_outer_finger"]),
        outer_y=float(outer[1]),
        outer_length_ratio=float(outer[2]) / float(mid[2]),
        mid_angle=mid_angle,
        top_angle=top_angle,
    )
    return float(params["finger_separation"]) - 2.0 * extent


def _quat_matrix(quat: tuple[float, float, float, float]) -> torch.Tensor:
    w, x, y, z = quat
    return torch.tensor(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=torch.float64,
    )


def _part_counts(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def _surface_residual(asset, cache) -> float:
    """Maximum distance of a cached local point from its declared primitive surface."""

    start = 0
    residuals = []
    counts = _part_counts(cache.points_body.shape[0], len(asset.cloud_parts))
    points_body = cache.points_body.double()
    for part, count in zip(asset.cloud_parts, counts):
        points = points_body[start : start + count]
        start += count
        rotation = _quat_matrix(part.geometry_to_body.quat_wxyz)
        translation = torch.tensor(
            part.geometry_to_body.translation, dtype=torch.float64
        )
        local = (points - translation) @ rotation
        if part.geometry_type == "box":
            half = 0.5 * torch.tensor(part.box_size, dtype=torch.float64)
            residual = torch.min(torch.abs(torch.abs(local) - half), dim=1).values
        elif part.geometry_type == "cylinder":
            radial = torch.linalg.vector_norm(local[:, :2], dim=1)
            residual = torch.minimum(
                torch.abs(radial - float(part.cylinder_radius)),
                torch.abs(torch.abs(local[:, 2]) - 0.5 * float(part.cylinder_length)),
            )
        else:
            raise ValueError(
                "This proof expects the generated revolute family to use only "
                f"box/cylinder primitives, got {part.geometry_type!r}"
            )
        residuals.append(residual)
    return float(torch.cat(residuals).max())


def _cache_motion_errors(asset, cache) -> tuple[float, float]:
    node_error = 0.0
    midpoint_error = 0.0
    bins = cache.state_clouds_palm.shape[0]
    for bin_index in range(bins):
        fraction = bin_index / (bins - 1)
        poses = one_dof_body_poses(
            asset, fraction, device="cpu", dtype=torch.float64
        )
        direct = torch.empty_like(cache.points_body, dtype=torch.float64)
        for body_index, body_name in enumerate(cache.body_names):
            rows = cache.point_body_index == body_index
            direct[rows] = transform_points(
                cache.points_body[rows].double(), poses[body_name]
            )
        node_error = max(
            node_error,
            float((direct.float() - cache.state_clouds_palm[bin_index]).abs().max()),
        )
        if bin_index + 1 < bins:
            midpoint = (bin_index + 0.5) / (bins - 1)
            poses_mid = one_dof_body_poses(
                asset, midpoint, device="cpu", dtype=torch.float64
            )
            direct_mid = torch.empty_like(direct)
            for body_index, body_name in enumerate(cache.body_names):
                rows = cache.point_body_index == body_index
                direct_mid[rows] = transform_points(
                    cache.points_body[rows].double(), poses_mid[body_name]
                )
            midpoint_error = max(
                midpoint_error,
                float(
                    (
                        direct_mid.float()
                        - cache.cloud_at_fraction(midpoint)
                    )
                    .abs()
                    .max()
                ),
            )
    return node_error, midpoint_error


def _metrics(entries: list[dict], assets, contact_assets: Path) -> tuple[dict, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, 128)
    gaps = np.asarray(
        [[_surface_gap(entry, float(value)) for value in fractions] for entry in entries]
    )
    node_errors = []
    midpoint_errors = []
    surface_residuals = []
    for asset in assets:
        cache = load_gripper_cloud_cache(
            cache_path_for_asset(asset),
            expected_gripper_id=asset.gripper_id,
            expected_source_manifest=asset.manifest_path,
            expected_source_asset_root=asset.root_dir,
        )
        node_error, midpoint_error = _cache_motion_errors(asset, cache)
        node_errors.append(node_error)
        midpoint_errors.append(midpoint_error)
        surface_residuals.append(_surface_residual(asset, cache))

    adjusted = json.loads(contact_assets.read_text(encoding="utf-8"))
    openings = [
        float(entry["opening_fraction"])
        for entry in adjusted
        if "source_one_dof_gripper_id" in entry
    ]
    off_bin = [
        value
        for value in openings
        if abs(value * 127.0 - round(value * 127.0)) > 1.0e-10
    ]
    monotonic = np.diff(gaps, axis=1) >= -1.0e-10
    metrics = {
        "gripper_count": len(entries),
        "cache_shape": [128, 512, 3],
        "closed_gap_target_mm": 0.0,
        "closed_gap_max_abs_error_mm": float(np.abs(gaps[:, 0]).max() * 1000.0),
        "open_gap_target_mm": 80.0,
        "open_gap_max_abs_error_mm": float(
            np.abs(gaps[:, -1] - 0.08).max() * 1000.0
        ),
        "monotonic_gripper_count": int(monotonic.all(axis=1).sum()),
        "cache_node_max_error_mm": max(node_errors) * 1000.0,
        "mid_bin_quantization_max_error_mm": max(midpoint_errors) * 1000.0,
        "primitive_surface_max_residual_mm": max(surface_residuals) * 1000.0,
        "contact_revolute_count": len(openings),
        "contact_openings_off_128_bins": len(off_bin),
        "contact_opening_unique_bins": len(
            {int(round(value * 127.0)) for value in openings}
        ),
    }
    return metrics, gaps


def _render_state_sheet(
    entries: list[dict],
    assets,
    selected_indices: list[int],
    manifest_dir: Path,
    output: Path,
) -> None:
    figure = plt.figure(
        figsize=(12.0, 3.45 * len(selected_indices)),
        facecolor="white",
    )
    for row, index in enumerate(selected_indices):
        entry = entries[index]
        asset = assets[index]
        cache = load_gripper_cloud_cache(
            cache_path_for_asset(asset),
            expected_gripper_id=asset.gripper_id,
            expected_source_manifest=asset.manifest_path,
            expected_source_asset_root=asset.root_dir,
        )
        for column, fraction in enumerate(STATE_FRACTIONS):
            axis = figure.add_subplot(
                len(selected_indices), 3, row * 3 + column + 1, projection="3d"
            )
            _draw_state(
                axis, entry, asset, manifest_dir, cache, fraction
            )
            state = ("closed", "half", "open")[column]
            gap_mm = 1000.0 * _surface_gap(entry, fraction)
            axis.set_title(
                f"{entry['id']} · {state}\n"
                f"joint openness={fraction:.1f}  surface gap={gap_mm:.2f} mm",
                fontsize=9,
            )
    figure.suptitle(
        "URDF visual geometry (transparent) + exact 512-point policy cloud\n"
        "Blue/Orange=left/right fixed point identities; red X=RL interaction center",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_metrics(metrics: dict, gaps: np.ndarray, output: Path) -> None:
    fractions = np.linspace(0.0, 1.0, gaps.shape[1])
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
    axis = axes[0, 0]
    axis.plot(fractions, gaps.T * 1000.0, color="#1677ff", alpha=0.11, linewidth=0.8)
    axis.plot(fractions, np.median(gaps, axis=0) * 1000.0, color="black", linewidth=2)
    axis.set_title("All 200 exact surface-gap trajectories")
    axis.set_xlabel("normalized revolute joint angle (0=closed, 1=open)")
    axis.set_ylabel("surface gap [mm]")
    axis.grid(alpha=0.2)

    axis = axes[0, 1]
    axis.scatter(
        np.arange(gaps.shape[0]),
        gaps[:, 0] * 1000.0,
        s=8,
        label="closed error from 0 mm",
    )
    axis.scatter(
        np.arange(gaps.shape[0]),
        (gaps[:, -1] - 0.08) * 1000.0,
        s=8,
        label="open error from 80 mm",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title("Endpoint errors for every gripper")
    axis.set_xlabel("gripper index")
    axis.set_ylabel("error [mm]")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)

    axis = axes[1, 0]
    labels = ("cache nodes", "mid-bin quantization", "primitive surface")
    values = (
        metrics["cache_node_max_error_mm"],
        metrics["mid_bin_quantization_max_error_mm"],
        metrics["primitive_surface_max_residual_mm"],
    )
    axis.bar(labels, values, color=("#1677ff", "#22a06b", "#ef4444"))
    axis.set_yscale("log")
    axis.set_ylabel("maximum error [mm, log scale]")
    axis.set_title("Worst case over all 200 grippers")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    axis.axis("off")
    lines = [
        "STRICT CHECKS",
        "",
        f"Grippers checked: {metrics['gripper_count']} / 200",
        f"Cache: {metrics['cache_shape']}",
        f"Monotonic trajectories: {metrics['monotonic_gripper_count']} / 200",
        f"Closed max error: {metrics['closed_gap_max_abs_error_mm']:.3e} mm",
        f"Open max error: {metrics['open_gap_max_abs_error_mm']:.3e} mm",
        f"Cache-node max error: {metrics['cache_node_max_error_mm']:.3e} mm",
        f"Mid-bin quantization: {metrics['mid_bin_quantization_max_error_mm']:.3e} mm",
        f"Surface max residual: {metrics['primitive_surface_max_residual_mm']:.3e} mm",
        f"Contact assets on bins: "
        f"{metrics['contact_revolute_count'] - metrics['contact_openings_off_128_bins']}"
        f" / {metrics['contact_revolute_count']}",
        f"Contact bins represented: {metrics['contact_opening_unique_bins']} / 128",
    ]
    axis.text(
        0.04,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    figure.suptitle("Canonical 128-bin revolute gripper verification", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_video(
    entry: dict,
    asset,
    manifest_dir: Path,
    output: Path,
) -> None:
    cache = load_gripper_cloud_cache(
        cache_path_for_asset(asset),
        expected_gripper_id=asset.gripper_id,
        expected_source_manifest=asset.manifest_path,
        expected_source_asset_root=asset.root_dir,
    )
    figure = plt.figure(figsize=(7, 7), facecolor="white")
    writer = FFMpegWriter(fps=30, metadata={"title": "one-DoF canonical cloud proof"})
    frame_fractions = np.concatenate(
        (np.linspace(0.0, 1.0, 128), np.linspace(1.0, 0.0, 128)[1:])
    )
    with writer.saving(figure, str(output), dpi=150):
        for fraction in frame_fractions:
            figure.clear()
            axis = figure.add_subplot(111, projection="3d")
            _draw_state(
                axis, entry, asset, manifest_dir, cache, float(fraction)
            )
            position = float(fraction) * 127.0
            axis.set_title(
                f"{entry['id']}\n"
                f"joint openness={fraction:.4f}  cache_position={position:.2f}/127  "
                f"gap={1000.0 * _surface_gap(entry, float(fraction)):.2f} mm"
            )
            writer.grab_frame()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contact-assets", type=Path, default=DEFAULT_CONTACT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--indices", default="0,17,63,101,157,199")
    parser.add_argument("--video-index", type=int, default=63)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--sheet-only", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    contact_assets = args.contact_assets.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(manifest.read_text(encoding="utf-8"))["grippers"]
    assets = load_one_dof_gripper_manifest(manifest, require_usd=False)
    indices = [int(value) for value in args.indices.split(",")]
    if any(index < 0 or index >= len(entries) for index in indices):
        parser.error(f"--indices must be within [0, {len(entries) - 1}]")
    if not 0 <= args.video_index < len(entries):
        parser.error(f"--video-index must be within [0, {len(entries) - 1}]")

    metrics = None
    if args.video_only and args.sheet_only:
        parser.error("--video-only and --sheet-only are mutually exclusive")
    if args.sheet_only:
        _render_state_sheet(
            entries,
            assets,
            indices,
            manifest.parent,
            output_dir / "state_sheet.png",
        )
    elif not args.video_only:
        metrics, gaps = _metrics(entries, assets, contact_assets)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        _render_state_sheet(
            entries,
            assets,
            indices,
            manifest.parent,
            output_dir / "state_sheet.png",
        )
        _render_metrics(metrics, gaps, output_dir / "metrics.png")
    if not args.no_video and not args.sheet_only:
        _render_video(
            entries[args.video_index],
            assets[args.video_index],
            manifest.parent,
            output_dir / "sweep.mp4",
        )
    if metrics is not None:
        print(json.dumps(metrics, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()

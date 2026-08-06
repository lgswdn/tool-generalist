#!/usr/bin/env python3
"""Export selected canonical gripper-cache bins as colored ASCII PLY files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARALLEL_CACHE_DIR = (
    REPO_ROOT / "gripper/generated_parallel_128/kinematic_cloud_cache"
)
DEFAULT_REVOLUTE_CACHE_DIR = (
    REPO_ROOT
    / "gripper/generated_graspgenx_matched_128/kinematic_cloud_cache"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts/visualization/gripper_cache_ply_check"
)
BODY_COLORS = (
    (120, 120, 120),
    (230, 70, 70),
    (70, 130, 230),
    (80, 190, 100),
    (220, 160, 50),
    (170, 80, 210),
    (40, 190, 190),
    (230, 100, 170),
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _load_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    clouds = torch.as_tensor(payload["state_clouds_palm"]).float()
    body_index = torch.as_tensor(payload["point_body_index"]).long()
    fractions = torch.as_tensor(payload["opening_fractions"]).float()
    if clouds.shape != (128, 512, 3):
        raise ValueError(f"Unexpected cloud shape {tuple(clouds.shape)} in {path}")
    if body_index.shape != (512,):
        raise ValueError(
            f"Unexpected point_body_index shape {tuple(body_index.shape)} in {path}"
        )
    if fractions.shape != (128,):
        raise ValueError(
            f"Unexpected opening_fractions shape {tuple(fractions.shape)} in {path}"
        )
    return payload


def _write_ply(
    path: Path,
    *,
    points: torch.Tensor,
    point_body_index: torch.Tensor,
    body_names: tuple[str, ...],
    gripper_id: str,
    bin_index: int,
    opening_fraction: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"comment gripper_id {gripper_id}",
        f"comment bin_index {bin_index}",
        f"comment opening_fraction {opening_fraction:.9f}",
        "comment coordinates palm_frame_meters",
    ]
    header.extend(
        f"comment body_{index} {name}"
        for index, name in enumerate(body_names)
    )
    header.extend(
        [
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int body_index",
            "property int point_index",
            "end_header",
        ]
    )
    lines = header
    for point_index, (point, raw_body_index) in enumerate(
        zip(points.tolist(), point_body_index.tolist())
    ):
        body_index = int(raw_body_index)
        color = BODY_COLORS[body_index % len(BODY_COLORS)]
        lines.append(
            f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{color[0]} {color[1]} {color[2]} {body_index} {point_index}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _export_family(
    *,
    family: str,
    cache_dir: Path,
    indices: tuple[int, ...],
    bins: tuple[int, ...],
    output_dir: Path,
) -> list[Path]:
    exported = []
    for asset_index in indices:
        stem = (
            f"{asset_index:06d}"
            if family == "parallel"
            else f"two_finger_revolute_{asset_index:06d}"
        )
        cache_path = cache_dir / f"{stem}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing {family} cache: {cache_path}")
        payload = _load_cache(cache_path)
        clouds = torch.as_tensor(payload["state_clouds_palm"]).float()
        body_index = torch.as_tensor(payload["point_body_index"]).long()
        fractions = torch.as_tensor(payload["opening_fractions"]).float()
        body_names = tuple(str(name) for name in payload["body_names"])
        gripper_id = str(payload["gripper_id"])
        for bin_index in bins:
            if not 0 <= bin_index < clouds.shape[0]:
                raise ValueError(f"Bin index must be in [0, 127], got {bin_index}")
            fraction = float(fractions[bin_index])
            ply_path = (
                output_dir
                / family
                / gripper_id
                / f"bin_{bin_index:03d}_opening_{fraction:.6f}.ply"
            )
            _write_ply(
                ply_path,
                points=clouds[bin_index],
                point_body_index=body_index,
                body_names=body_names,
                gripper_id=gripper_id,
                bin_index=bin_index,
                opening_fraction=fraction,
            )
            exported.append(ply_path)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel-cache-dir",
        type=Path,
        default=DEFAULT_PARALLEL_CACHE_DIR,
    )
    parser.add_argument(
        "--revolute-cache-dir",
        type=Path,
        default=DEFAULT_REVOLUTE_CACHE_DIR,
    )
    parser.add_argument(
        "--indices",
        type=_parse_csv_ints,
        default=(0, 99, 199),
        help="Comma-separated asset indices.",
    )
    parser.add_argument(
        "--bins",
        type=_parse_csv_ints,
        default=(0, 32, 64, 96, 127),
        help="Comma-separated cache-bin indices.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    exported = []
    exported.extend(
        _export_family(
            family="parallel",
            cache_dir=args.parallel_cache_dir.expanduser().resolve(),
            indices=args.indices,
            bins=args.bins,
            output_dir=output_dir,
        )
    )
    exported.extend(
        _export_family(
            family="revolute",
            cache_dir=args.revolute_cache_dir.expanduser().resolve(),
            indices=args.indices,
            bins=args.bins,
            output_dir=output_dir,
        )
    )
    print(f"Exported {len(exported)} PLY files to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit rank-10 tokens from cheap analytic point-cloud patch summaries.

The source data uses a single 512-point-cloud to 512-point-cloud nearest-neighbor
query.  This probe then performs only analytic within-patch reductions; it does
not contain a PointNet and never queries either mesh.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.patch_oracle_probe import DeepPatchOracleToRankToken
from pretrain.pointcloud_patch_oracle import (
    FAST_POINTCLOUD_PATCH_FEATURE_NAMES,
    build_fast_pointcloud_patch_features,
)
from scripts.train_rank10_patch_oracle_probe import RunningMoments, _train_model


DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)


def _source_paths(source_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"fast point-cloud source data is missing: {manifest_path}; run "
            "scripts/train_rank10_minimal_pointnet.py --stage prepare first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = "rank10_fast_pointcloud11_dataset_v1"
    if manifest.get("schema_version") != expected:
        raise RuntimeError(
            f"source schema={manifest.get('schema_version')!r}, expected {expected!r}"
        )
    train = [source_dir / item["path"] for item in manifest["train"]]
    validation = [source_dir / item["path"] for item in manifest["validation"]]
    return train, validation, manifest


def _convert_shard(source: Path, destination: Path) -> tuple[int, torch.Tensor, torch.Tensor]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    point_features = payload["point_features"].float()
    targets = payload["targets"].float()
    features = build_fast_pointcloud_patch_features(point_features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.half(), "targets": targets}, destination)
    return features.shape[0], features, targets


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        if not args.overwrite_prepared:
            manifest = args.output_dir / "manifest.json"
            if manifest.is_file():
                print(f"[prepare] reusing {args.output_dir}", flush=True)
                return
            raise FileExistsError(
                f"incomplete output exists: {args.output_dir}; pass --overwrite-prepared"
            )
        shutil.rmtree(args.output_dir)

    train_paths, validation_paths, source_manifest = _source_paths(args.source_dir)
    feature_moments = RunningMoments(len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES))
    target_moments = RunningMoments(10)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, paths in (("train", train_paths), ("validation", validation_paths)):
        for index, source in enumerate(paths):
            destination = args.output_dir / split / source.name
            count, features, targets = _convert_shard(source, destination)
            if split == "train":
                feature_moments.update(features)
                target_moments.update(targets)
            splits[split].append(
                {"path": str(destination.relative_to(args.output_dir)), "patches": count}
            )
            print(f"[prepare:{split}] shard={index + 1}/{len(paths)} patches={count}", flush=True)

    feature_mean, feature_std = feature_moments.mean_std()
    target_mean, target_std = target_moments.mean_std()
    torch.save(
        {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        args.output_dir / "normalization.pt",
    )
    manifest = {
        "schema_version": "rank10_fast_pointcloud_patch_oracle35_v1",
        "source_dir": str(args.source_dir),
        "source_rl_checkpoint": source_manifest.get("source_rl_checkpoint"),
        "feature_count": len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES),
        "feature_names": FAST_POINTCLOUD_PATCH_FEATURE_NAMES,
        "distance": "unsigned_nearest_opposite_pointcloud_point",
        "mesh_queries": False,
        "pointnet": False,
        "strict_patch_reduction": True,
        **splits,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare the fast patch dataset first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mesh_queries") is not False or manifest.get("pointnet") is not False:
        raise RuntimeError("refusing a source that is not mesh-free and PointNet-free")
    train_paths = [args.output_dir / item["path"] for item in manifest["train"]]
    validation_paths = [args.output_dir / item["path"] for item in manifest["validation"]]
    normalization = torch.load(
        args.output_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    torch.manual_seed(args.seed)
    _train_model(
        "fast_patch_oracle35",
        DeepPatchOracleToRankToken(
            input_dim=len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES)
        ),
        args=args,
        train_paths=train_paths,
        validation_paths=validation_paths,
        normalization=normalization,
        feature_names=FAST_POINTCLOUD_PATCH_FEATURE_NAMES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--overwrite-prepared", action="store_true")
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.source_dir.parent.parent / "fast_patch_oracle35"
    args.output_dir = args.output_dir.resolve()
    if args.epochs < 1 or args.train_batch_size < 1:
        parser.error("epochs and train batch size must be positive")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"train", "all"}:
        train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit rank-10 tokens with fast unsigned point-cloud proximity inputs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.rank10_pointnet_contract import (
    PATCH_METADATA_NAMES as SOURCE_METADATA_NAMES,
    POINT_FEATURE_NAMES as SOURCE_POINT_FEATURE_NAMES,
)


DEFAULT_PROBE_DIR = REPO_ROOT / "artifacts/probes/rank10_patch_pointnet"
PATCHES_PER_BODY = 16
POINTS_PER_PATCH = 32
PATCHES_PER_FRAME = 2 * PATCHES_PER_BODY

FAST_POINT_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "patch_center_x",
    "patch_center_y",
    "patch_center_z",
    "nearest_opposite_point_distance",
    "nearest_opposite_point_direction_x",
    "nearest_opposite_point_direction_y",
    "nearest_opposite_point_direction_z",
    "patch_is_tool",
)


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.square_sum = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1, values.shape[-1]).to(device="cpu", dtype=torch.float64)
        self.count += flat.shape[0]
        self.sum += flat.sum(dim=0)
        self.square_sum += flat.square().sum(dim=0)

    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("normalization requires at least two point samples")
        mean = self.sum / self.count
        variance = (self.square_sum / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float().clamp_min(1e-6)


class FastPatchPointNet(nn.Module):
    """Shared per-point MLP and max pooling within exactly one patch."""

    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(11, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.patch_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.patch_mlp(self.point_mlp(point_features).amax(dim=-2))


def _source_paths(probe_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    manifest = json.loads((probe_dir / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest.get("point_feature_names", ())) != SOURCE_POINT_FEATURE_NAMES:
        raise RuntimeError("source point-feature contract does not match")
    if tuple(manifest.get("patch_metadata_names", ())) != SOURCE_METADATA_NAMES:
        raise RuntimeError("source patch-metadata contract does not match")
    train = [probe_dir / item["path"] for item in manifest["train"]["shards"]]
    validation = [probe_dir / item["path"] for item in manifest["validation"]["shards"]]
    return train, validation, manifest


def _fast_inputs_for_frames(
    point_features: torch.Tensor,
    patch_metadata: torch.Tensor,
) -> torch.Tensor:
    """Convert complete frames to unsigned nearest-opposite-cloud features."""

    if point_features.ndim != 4 or point_features.shape[1:] != (
        PATCHES_PER_FRAME,
        POINTS_PER_PATCH,
        20,
    ):
        raise ValueError(f"invalid framed point features: {point_features.shape}")
    if patch_metadata.shape != (point_features.shape[0], PATCHES_PER_FRAME, 5):
        raise ValueError(f"invalid framed patch metadata: {patch_metadata.shape}")
    tool_mask = patch_metadata[..., 3]
    if not (
        torch.all(tool_mask[:, :PATCHES_PER_BODY] > 0.5)
        and torch.all(tool_mask[:, PATCHES_PER_BODY:] < 0.5)
    ):
        raise RuntimeError("source shard does not preserve tool/object patch ordering")

    local = point_features[..., 0:3]
    centers = patch_metadata[..., :3]
    points = local + centers.unsqueeze(-2)
    tool_points = points[:, :PATCHES_PER_BODY].reshape(points.shape[0], -1, 3)
    object_points = points[:, PATCHES_PER_BODY:].reshape(points.shape[0], -1, 3)
    pairwise = torch.cdist(tool_points, object_points)

    tool_distance, tool_nearest_index = pairwise.min(dim=-1)
    nearest_object = object_points.gather(
        1, tool_nearest_index.unsqueeze(-1).expand(-1, -1, 3)
    )
    tool_direction = F.normalize(nearest_object - tool_points, dim=-1, eps=1e-8)

    object_distance, object_nearest_index = pairwise.min(dim=-2)
    nearest_tool = tool_points.gather(
        1, object_nearest_index.unsqueeze(-1).expand(-1, -1, 3)
    )
    object_direction = F.normalize(nearest_tool - object_points, dim=-1, eps=1e-8)

    patch_distance = torch.cat(
        (
            tool_distance.reshape(-1, PATCHES_PER_BODY, POINTS_PER_PATCH),
            object_distance.reshape(-1, PATCHES_PER_BODY, POINTS_PER_PATCH),
        ),
        dim=1,
    )
    patch_direction = torch.cat(
        (
            tool_direction.reshape(-1, PATCHES_PER_BODY, POINTS_PER_PATCH, 3),
            object_direction.reshape(-1, PATCHES_PER_BODY, POINTS_PER_PATCH, 3),
        ),
        dim=1,
    )
    repeated_centers = centers.unsqueeze(-2).expand(-1, -1, POINTS_PER_PATCH, -1)
    repeated_body = tool_mask.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, POINTS_PER_PATCH, 1
    )
    result = torch.cat(
        (
            local,
            repeated_centers,
            patch_distance.unsqueeze(-1),
            patch_direction,
            repeated_body,
        ),
        dim=-1,
    )
    if result.shape[-1] != len(FAST_POINT_FEATURE_NAMES):
        raise RuntimeError("fast point-feature contract mismatch")
    return result


def _convert_shard(
    source_path: Path,
    destination_path: Path,
    *,
    device: torch.device,
    frame_batch_size: int,
    moments: RunningMoments | None,
) -> int:
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    point_features = payload["point_features"]
    patch_metadata = payload["patch_metadata"]
    targets = payload["targets"].float()
    if targets.shape[0] % PATCHES_PER_FRAME:
        raise RuntimeError(f"shard does not contain complete frames: {source_path}")
    frame_count = targets.shape[0] // PATCHES_PER_FRAME
    point_features = point_features.reshape(
        frame_count, PATCHES_PER_FRAME, POINTS_PER_PATCH, 20
    )
    patch_metadata = patch_metadata.reshape(frame_count, PATCHES_PER_FRAME, 5)
    parts = []
    with torch.inference_mode():
        for start in range(0, frame_count, frame_batch_size):
            fast = _fast_inputs_for_frames(
                point_features[start : start + frame_batch_size].to(device).float(),
                patch_metadata[start : start + frame_batch_size].to(device).float(),
            )
            fast = fast.reshape(-1, POINTS_PER_PATCH, 11).half().cpu()
            if moments is not None:
                moments.update(fast)
            parts.append(fast)
    fast_features = torch.cat(parts, dim=0)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"point_features": fast_features, "targets": targets}, destination_path)
    return targets.shape[0]


def prepare(args: argparse.Namespace) -> None:
    data_dir = args.output_dir / "data"
    prepared_manifest_path = data_dir / "manifest.json"
    if prepared_manifest_path.is_file() and not args.overwrite_prepared:
        print(f"[prepare] reusing {data_dir}", flush=True)
        return
    if data_dir.exists():
        if not args.overwrite_prepared:
            raise FileExistsError(
                f"incomplete prepared data exists: {data_dir}; pass --overwrite-prepared"
            )
        shutil.rmtree(data_dir)
    train_paths, validation_paths, source_manifest = _source_paths(args.probe_dir)
    device = torch.device(args.device)
    moments = RunningMoments(11)
    prepared: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, paths in (("train", train_paths), ("validation", validation_paths)):
        for index, source_path in enumerate(paths):
            destination = data_dir / split / source_path.name
            patches = _convert_shard(
                source_path,
                destination,
                device=device,
                frame_batch_size=args.prepare_frame_batch_size,
                moments=moments if split == "train" else None,
            )
            prepared[split].append(
                {"path": str(destination.relative_to(data_dir)), "patches": patches}
            )
            print(
                f"[prepare:{split}] shard={index + 1}/{len(paths)} patches={patches}",
                flush=True,
            )
    input_mean, input_std = moments.mean_std()
    source_normalization = torch.load(
        args.probe_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    torch.save(
        {
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": source_normalization["target_mean"],
            "target_std": source_normalization["target_std"],
        },
        data_dir / "normalization.pt",
    )
    prepared_manifest = {
        "schema_version": "rank10_fast_pointcloud11_dataset_v1",
        "source_probe_dir": str(args.probe_dir),
        "source_rl_checkpoint": source_manifest.get("rl_checkpoint"),
        "point_feature_names": FAST_POINT_FEATURE_NAMES,
        "distance": "unsigned_nearest_opposite_patch_point",
        "direction": "normalized_vector_to_nearest_opposite_patch_point",
        "signed_distance": False,
        "log_distance": False,
        "patch_equivariant": True,
        "cross_patch_network": False,
        **prepared,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepared_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare fast point features first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("point_feature_names", ())) != FAST_POINT_FEATURE_NAMES:
        raise RuntimeError("prepared fast point-feature contract does not match")
    train = [data_dir / item["path"] for item in manifest["train"]]
    validation = [data_dir / item["path"] for item in manifest["validation"]]
    return train, validation


def _source_token_up(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = str(manifest.get("source_rl_checkpoint") or "")
    if not checkpoint_path:
        raise RuntimeError("prepared data does not record its source rank-10 RL checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"source RL checkpoint lacks model_state_dict: {checkpoint_path}")
    weight = state.get("encoder_token_bottleneck_up.weight")
    bias = state.get("encoder_token_bottleneck_up.bias")
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (128, 10):
        raise RuntimeError("source rank-10 checkpoint lacks a 10D-to-128D token-up weight")
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (128,):
        raise RuntimeError("source rank-10 checkpoint lacks a 128D token-up bias")
    return weight.float(), bias.float(), checkpoint_path


def _load_prepared_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["point_features"].float(), payload["targets"].float()


def _evaluate(
    model: FastPatchPointNet,
    paths: list[Path],
    *,
    device: torch.device,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    squared_error = torch.zeros(10, dtype=torch.float64)
    target_square = torch.zeros(10, dtype=torch.float64)
    cosine_sum = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for path in paths:
            features, targets = _load_prepared_shard(path)
            for start in range(0, targets.shape[0], batch_size):
                x = features[start : start + batch_size].to(device)
                y = targets[start : start + batch_size].to(device)
                x = ((x - input_mean) / input_std).clamp(-12, 12)
                prediction = model(x) * target_std + target_mean
                error = (prediction - y).double().cpu()
                centered_target = (y - target_mean).double().cpu()
                squared_error += error.square().sum(dim=0)
                target_square += centered_target.square().sum(dim=0)
                cosine_sum += F.cosine_similarity(
                    prediction, y, dim=-1, eps=1e-8
                ).sum().item()
                count += y.shape[0]
    return {
        "patches": count,
        "mse": float(squared_error.sum() / (count * 10)),
        "r2": float(1.0 - squared_error.sum() / target_square.sum().clamp_min(1e-12)),
        "mean_cosine": cosine_sum / count,
        "per_dimension_r2": (
            1.0 - squared_error / target_square.clamp_min(1e-12)
        ).tolist(),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    data_dir = args.output_dir / "data"
    train_paths, validation_paths = _prepared_paths(data_dir)
    normalization = torch.load(
        data_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    input_mean = normalization["input_mean"].to(device)
    input_std = normalization["input_std"].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    token_up_weight, token_up_bias, source_rl_checkpoint = _source_token_up(data_dir)
    model = FastPatchPointNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = random.Random(args.seed)
    history = []
    best_r2 = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        paths = list(train_paths)
        rng.shuffle(paths)
        total_loss = 0.0
        total_count = 0
        for path in paths:
            features, targets = _load_prepared_shard(path)
            order = torch.randperm(targets.shape[0])
            for start in range(0, targets.shape[0], args.batch_size):
                indices = order[start : start + args.batch_size]
                x = features[indices].to(device)
                y = targets[indices].to(device)
                x = ((x - input_mean) / input_std).clamp(-12, 12)
                y = (y - target_mean) / target_std
                prediction = model(x)
                mse = F.mse_loss(prediction, y)
                cosine = 1.0 - F.cosine_similarity(
                    prediction, y, dim=-1, eps=1e-8
                ).mean()
                loss = mse + args.cosine_weight * cosine
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * indices.numel()
                total_count += indices.numel()
        metrics = _evaluate(
            model,
            validation_paths,
            device=device,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.batch_size,
        )
        metrics.update({"epoch": epoch, "train_loss": total_loss / total_count})
        history.append(metrics)
        print(
            f"[train:fast_pointcloud11] epoch={epoch}/{args.epochs} "
            f"loss={metrics['train_loss']:.6f} val_r2={metrics['r2']:.6f} "
            f"val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        checkpoint = {
            "schema_version": "rank10_fast_pointcloud11_v2",
            "model_state_dict": model.state_dict(),
            "point_feature_names": FAST_POINT_FEATURE_NAMES,
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "token_up_weight": token_up_weight,
            "token_up_bias": token_up_bias,
            "source_rl_checkpoint": source_rl_checkpoint,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "fast_pointcloud11_last.pt")
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(checkpoint, args.output_dir / "fast_pointcloud11_best.pt")
    (args.output_dir / "fast_pointcloud11_metrics.json").write_text(
        json.dumps(
            {
                "point_feature_count": 11,
                "point_feature_names": FAST_POINT_FEATURE_NAMES,
                "best_validation_r2": best_r2,
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-frame-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-prepared", action="store_true")
    args = parser.parse_args()
    args.probe_dir = args.probe_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.probe_dir / "fast_pointcloud11"
    args.output_dir = args.output_dir.resolve()
    if args.prepare_frame_batch_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("frame batch size, epochs, and training batch size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"train", "all"}:
        train(args)


if __name__ == "__main__":
    main()

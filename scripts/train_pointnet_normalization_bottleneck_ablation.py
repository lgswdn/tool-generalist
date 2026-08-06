#!/usr/bin/env python3
"""Train one controlled PointNet normalization/bottleneck offline ablation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_rank10_minimal_pointnet import (  # noqa: E402
    FAST_POINT_FEATURE_NAMES,
)


DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts/probes/pointnet_normalization_bottleneck_ablation"
)
VARIANTS = {
    "normalized_direct128": (True, False),
    "normalized_rank10": (True, True),
    "unnormalized_direct128": (False, False),
    "unnormalized_rank10": (False, True),
}


class ControlledPointNet(nn.Module):
    """Exact shared PointNet stem with either a direct-128 or rank-10 token."""

    def __init__(self, *, rank10: bool) -> None:
        super().__init__()
        self.rank10 = bool(rank10)
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
        if self.rank10:
            self.patch_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 10),
            )
            self.prediction_head = nn.Identity()
        else:
            self.patch_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
            )
            # This head exists only for the offline teacher-token objective.
            # The 128D representation before it is what RL would consume.
            self.prediction_head = nn.Linear(128, 10)

    def forward(self, point_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.point_mlp(point_features).amax(dim=-2)
        representation = self.patch_mlp(pooled)
        return representation, self.prediction_head(representation)


def _dataset_paths(data_dir: Path, split: str, max_shards: int | None) -> list[Path]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("point_feature_names", ())) != FAST_POINT_FEATURE_NAMES:
        raise RuntimeError(f"fast-11 feature contract mismatch: {manifest_path}")
    entries = manifest.get(split)
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{manifest_path} lacks non-empty {split!r} data")
    if max_shards is not None:
        entries = entries[:max_shards]
    return [data_dir / entry["path"] for entry in entries]


def _load_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload.get("point_features")
    targets = payload.get("targets")
    if not isinstance(features, torch.Tensor) or tuple(features.shape[1:]) != (32, 11):
        raise RuntimeError(f"invalid fast-11 point features: {path}")
    if not isinstance(targets, torch.Tensor) or targets.shape != (features.shape[0], 10):
        raise RuntimeError(f"invalid rank-10 targets: {path}")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise RuntimeError(f"non-finite offline ablation data: {path}")
    return features.float(), targets.float()


def _input_normalization(
    normalization: dict[str, torch.Tensor],
    *,
    enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if enabled:
        return (
            normalization["input_mean"].to(device),
            normalization["input_std"].to(device),
        )
    return torch.zeros(11, device=device), torch.ones(11, device=device)


def _normalize_input(
    values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return ((values - mean) / std).clamp(-12.0, 12.0)


@torch.inference_mode()
def _evaluate(
    model: ControlledPointNet,
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
    representation_sum = None
    representation_square_sum = None
    count = 0
    model.eval()
    for path in paths:
        features, targets = _load_shard(path)
        for start in range(0, targets.shape[0], batch_size):
            x = features[start : start + batch_size].to(device)
            y = targets[start : start + batch_size].to(device)
            representation, standardized_prediction = model(
                _normalize_input(x, input_mean, input_std)
            )
            prediction = standardized_prediction * target_std + target_mean
            error = (prediction - y).double().cpu()
            centered_target = (y - target_mean).double().cpu()
            squared_error += error.square().sum(dim=0)
            target_square += centered_target.square().sum(dim=0)
            cosine_sum += F.cosine_similarity(
                prediction, y, dim=-1, eps=1e-8
            ).sum().item()
            representation_cpu = representation.double().cpu()
            if representation_sum is None:
                representation_sum = torch.zeros(representation.shape[1], dtype=torch.float64)
                representation_square_sum = torch.zeros_like(representation_sum)
            representation_sum += representation_cpu.sum(dim=0)
            representation_square_sum += representation_cpu.square().sum(dim=0)
            count += y.shape[0]
    if representation_sum is None or representation_square_sum is None:
        raise RuntimeError("validation produced no representations")
    representation_variance = (
        representation_square_sum / count - (representation_sum / count).square()
    ).clamp_min(0)
    return {
        "patches": count,
        "mse": float(squared_error.sum() / (count * 10)),
        "r2": float(1.0 - squared_error.sum() / target_square.sum().clamp_min(1e-12)),
        "mean_cosine": cosine_sum / count,
        "per_dimension_r2": (
            1.0 - squared_error / target_square.clamp_min(1e-12)
        ).tolist(),
        "representation_dimension": representation_sum.numel(),
        "mean_representation_variance": float(representation_variance.mean()),
    }


def _checkpoint(
    *,
    args: argparse.Namespace,
    model: ControlledPointNet,
    normalization: dict[str, torch.Tensor],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    encoder_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("prediction_head.")
    }
    encoder_state["input_mean"] = input_mean.detach().cpu()
    encoder_state["input_std"] = input_std.detach().cpu()
    return {
        "schema_version": "pointnet_normalization_bottleneck_ablation_v1",
        "variant": args.variant,
        "normalized": args.normalized,
        "rank10_bottleneck": args.rank10,
        "representation_dimension": 10 if args.rank10 else 128,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "encoder_state_dict": encoder_state,
        "normalization": {
            key: value.detach().cpu() for key, value in normalization.items()
        },
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "metrics": metrics,
    }


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_root / args.variant
    if output_dir.exists():
        raise FileExistsError(
            f"offline ablation output already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = _dataset_paths(args.data_dir, "train", args.max_train_shards)
    validation_paths = _dataset_paths(
        args.data_dir, "validation", args.max_validation_shards
    )
    normalization = torch.load(
        args.data_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    input_mean, input_std = _input_normalization(
        normalization, enabled=args.normalized, device=device
    )
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    model = ControlledPointNet(rank10=args.rank10).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_r2 = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_paths = list(enumerate(train_paths))
        random.Random(args.seed + epoch).shuffle(epoch_paths)
        total_loss = 0.0
        total_count = 0
        for shard_index, path in epoch_paths:
            features, targets = _load_shard(path)
            generator = torch.Generator().manual_seed(
                args.seed + epoch * 1_000_003 + shard_index
            )
            order = torch.randperm(targets.shape[0], generator=generator)
            for start in range(0, targets.shape[0], args.batch_size):
                indices = order[start : start + args.batch_size]
                x = features[indices].to(device)
                y = targets[indices].to(device)
                y = (y - target_mean) / target_std
                _, prediction = model(_normalize_input(x, input_mean, input_std))
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
            f"[offline-ablation:{args.variant}] epoch={epoch}/{args.epochs} "
            f"loss={metrics['train_loss']:.6f} val_r2={metrics['r2']:.6f} "
            f"val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        checkpoint = _checkpoint(
            args=args,
            model=model,
            normalization=normalization,
            input_mean=input_mean,
            input_std=input_std,
            metrics=metrics,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "pointnet_normalization_bottleneck_ablation_v1",
                "variant": args.variant,
                "normalized": args.normalized,
                "rank10_bottleneck": args.rank10,
                "best_validation_r2": best_r2,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-shards", type=int)
    parser.add_argument("--max-validation-shards", type=int)
    args = parser.parse_args()
    args.normalized, args.rank10 = VARIANTS[args.variant]
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    if min(args.epochs, args.batch_size) <= 0:
        parser.error("epochs and batch size must be positive")
    if args.max_train_shards is not None and args.max_train_shards <= 0:
        parser.error("max train shards must be positive")
    if args.max_validation_shards is not None and args.max_validation_shards <= 0:
        parser.error("max validation shards must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())

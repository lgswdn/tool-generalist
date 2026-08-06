#!/usr/bin/env python3
"""Fit a hand-deduplicated 36D patch oracle to the rank-10 token."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.patch_oracle_probe import PATCH_ORACLE_FEATURE_NAMES, PatchOracleToRankToken


DEFAULT_PROBE_DIR = REPO_ROOT / "artifacts/probes/rank10_patch_oracle"

LOG_SDF_MOMENTS = {
    "log_sdf_min",
    "log_sdf_max",
    "log_sdf_mean",
    "log_sdf_std",
}
QUADRATIC_CURVATURE_AND_FIT = {
    "quadratic_sdf_square_normal",
    "quadratic_sdf_normal_tangent1",
    "quadratic_sdf_normal_tangent2",
    "quadratic_sdf_square_tangent1",
    "quadratic_sdf_tangent1_tangent2",
    "quadratic_sdf_square_tangent2",
    "quadratic_sdf_residual_rms",
    "quadratic_sdf_fit_r2",
}


def compact36_feature_names() -> tuple[str, ...]:
    """Retain distinct location, shape, proximity, direction, and curvature cues."""

    selected = []
    for name in PATCH_ORACLE_FEATURE_NAMES:
        keep = (
            name == "patch_is_tool"
            or name.startswith("center_")
            or name.startswith("local_mean_")
            or name.startswith("pca_eigenvalue_")
            or name in LOG_SDF_MOMENTS
            or name.startswith("signed_sdf_q")
            or name.startswith("closest_mesh_direction_")
            or name.startswith("local_sdf_gradient_")
            or name in QUADRATIC_CURVATURE_AND_FIT
        )
        if keep:
            selected.append(name)
    result = tuple(selected)
    if len(result) != 36:
        raise RuntimeError(f"compact oracle contract must contain 36 features, got {len(result)}")
    return result


def _load_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"].float(), payload["targets"].float()


def _load_paths(probe_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest = json.loads((probe_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_names") != list(PATCH_ORACLE_FEATURE_NAMES):
        raise RuntimeError("probe dataset does not match the current named-feature contract")
    train_paths = [probe_dir / item["path"] for item in manifest["train"]["shards"]]
    validation_paths = [
        probe_dir / item["path"] for item in manifest["validation"]["shards"]
    ]
    return train_paths, validation_paths


def _evaluate(
    model: PatchOracleToRankToken,
    paths: list[Path],
    selected_indices: torch.Tensor,
    *,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
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
            features, targets = _load_shard(path)
            features = features[:, selected_indices]
            for start in range(0, targets.shape[0], batch_size):
                x = features[start : start + batch_size].to(device)
                y = targets[start : start + batch_size].to(device)
                x = ((x - feature_mean) / feature_std).clamp(-12, 12)
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
    feature_names = compact36_feature_names()
    name_to_index = {name: index for index, name in enumerate(PATCH_ORACLE_FEATURE_NAMES)}
    selected_indices_list = [name_to_index[name] for name in feature_names]
    selected_indices = torch.tensor(selected_indices_list, dtype=torch.long)
    train_paths, validation_paths = _load_paths(args.probe_dir)
    normalization = torch.load(
        args.probe_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    feature_mean = normalization["feature_mean"][selected_indices].to(device)
    feature_std = normalization["feature_std"][selected_indices].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    model = PatchOracleToRankToken(input_dim=36).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = random.Random(args.seed)
    history = []
    best_r2 = float("-inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        paths = list(train_paths)
        rng.shuffle(paths)
        total_loss = 0.0
        total_count = 0
        for path in paths:
            features, targets = _load_shard(path)
            order = torch.randperm(targets.shape[0])
            for start in range(0, targets.shape[0], args.batch_size):
                batch_indices = order[start : start + args.batch_size]
                x = features[batch_indices][:, selected_indices].to(device)
                y = targets[batch_indices].to(device)
                x = ((x - feature_mean) / feature_std).clamp(-12, 12)
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
                total_loss += loss.item() * batch_indices.numel()
                total_count += batch_indices.numel()
        metrics = _evaluate(
            model,
            validation_paths,
            selected_indices,
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.batch_size,
        )
        metrics.update({"epoch": epoch, "train_loss": total_loss / total_count})
        history.append(metrics)
        print(
            f"[train:compact36] epoch={epoch}/{args.epochs} "
            f"loss={metrics['train_loss']:.6f} val_r2={metrics['r2']:.6f} "
            f"val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        checkpoint = {
            "schema_version": "rank10_compact36_patch_oracle_v1",
            "model_state_dict": model.state_dict(),
            "selected_feature_indices": selected_indices_list,
            "selected_feature_names": feature_names,
            "normalization": {
                "feature_mean": feature_mean.cpu(),
                "feature_std": feature_std.cpu(),
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
            },
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "compact36_last.pt")
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(checkpoint, args.output_dir / "compact36_best.pt")
    summary = {
        "feature_count": 36,
        "feature_names": feature_names,
        "best_validation_r2": best_r2,
        "history": history,
    }
    (args.output_dir / "compact36_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.probe_dir = args.probe_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.probe_dir / "compact36"
    args.output_dir = args.output_dir.resolve()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit rank-10 tokens with an explicit unsigned 29D pointwise oracle.

Every added value is computed independently for one point from its patch-local
position and unsigned nearest-opposite-cloud result.  There are no mesh,
signed-distance, cross-patch, patch-reduction, or target-derived features.
"""

from __future__ import annotations

import argparse
import json
import math
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

from scripts.train_rank10_minimal_pointnet import (
    FAST_POINT_FEATURE_NAMES,
    _load_prepared_shard,
    _prepared_paths,
    _source_token_up,
)
from scripts.train_rank10_tiny_pointnet import (
    _finalize_metrics,
    _metrics_accumulator,
    _update_metrics,
)


DEFAULT_DATA_DIR = (
    REPO_ROOT / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts/probes/rank10_patch_pointnet/pointwise29_patch_mlp"
)

SOFT_PROXIMITY_SCALES_M = (0.002, 0.005, 0.010)
POINTWISE29_FEATURE_NAMES: tuple[str, ...] = FAST_POINT_FEATURE_NAMES + (
    "nearest_displacement_x",
    "nearest_displacement_y",
    "nearest_displacement_z",
    "soft_proximity_2mm",
    "soft_proximity_5mm",
    "soft_proximity_10mm",
    "soft_5mm_local_x",
    "soft_5mm_local_y",
    "soft_5mm_local_z",
    "soft_5mm_direction_x",
    "soft_5mm_direction_y",
    "soft_5mm_direction_z",
    "local_direction_product_x",
    "local_direction_product_y",
    "local_direction_product_z",
    "local_radius",
    "nearest_opposite_point_radius",
    "transverse_offset",
)

FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "relative_xyz": (0, 1, 2),
    "patch_center_xyz": (3, 4, 5),
    "unsigned_nearest_distance": (6,),
    "nearest_direction_xyz": (7, 8, 9),
    "patch_is_tool": (10,),
    "nearest_displacement_xyz": (11, 12, 13),
    "multiscale_soft_proximity": (14, 15, 16),
    "soft_5mm_local_xyz": (17, 18, 19),
    "soft_5mm_direction_xyz": (20, 21, 22),
    "local_direction_products": (23, 24, 25),
    "local_radius": (26,),
    "nearest_opposite_point_radius": (27,),
    "transverse_offset": (28,),
}


def build_pointwise29_features(point_features: torch.Tensor) -> torch.Tensor:
    """Expand ``(..., K, 11)`` fast point features to pointwise 29D."""

    if point_features.ndim < 2 or point_features.shape[-1] != 11:
        raise ValueError("point_features must have shape (..., K, 11)")
    local = point_features[..., 0:3]
    distance = point_features[..., 6].clamp_min(0)
    direction = point_features[..., 7:10]
    displacement = distance.unsqueeze(-1) * direction
    soft = torch.stack(
        [torch.exp(-distance / scale) for scale in SOFT_PROXIMITY_SCALES_M],
        dim=-1,
    )
    soft_5mm = soft[..., 1:2]
    local_direction_products = local * direction
    local_radius = torch.linalg.vector_norm(local, dim=-1, keepdim=True)
    opposite_radius = torch.linalg.vector_norm(
        local + displacement, dim=-1, keepdim=True
    )
    transverse_offset = torch.linalg.vector_norm(
        torch.cross(local, direction, dim=-1), dim=-1, keepdim=True
    )
    result = torch.cat(
        (
            point_features,
            displacement,
            soft,
            soft_5mm * local,
            soft_5mm * direction,
            local_direction_products,
            local_radius,
            opposite_radius,
            transverse_offset,
        ),
        dim=-1,
    )
    if result.shape[-1] != len(POINTWISE29_FEATURE_NAMES):
        raise RuntimeError(f"pointwise feature contract mismatch: {result.shape}")
    if not torch.isfinite(result).all():
        raise RuntimeError("pointwise29 features contain non-finite values")
    return result


class Pointwise29PointNet(nn.Module):
    """29D point oracle -> 16D max pool -> richer patch MLP -> rank 10."""

    def __init__(self, hidden_dim: int = 16, patch_hidden_dim: int = 32) -> None:
        super().__init__()
        self.point_linear = nn.Linear(29, hidden_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(hidden_dim, patch_hidden_dim),
            nn.SiLU(),
            nn.Linear(patch_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        point_hidden = F.silu(self.point_linear(point_features))
        return self.patch_mlp(point_hidden.amax(dim=-2))


class RunningFeatureMoments:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.square_sum = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1, values.shape[-1]).double()
        self.count += flat.shape[0]
        self.sum += flat.sum(dim=0).cpu()
        self.square_sum += flat.square().sum(dim=0).cpu()

    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("normalization requires at least two points")
        mean = self.sum / self.count
        variance = (self.square_sum / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float().clamp_min(1e-6)


def _normalization(
    args: argparse.Namespace,
    train_paths: list[Path],
    device: torch.device,
) -> dict[str, Any]:
    path = args.output_dir / "pointwise29_normalization.pt"
    if path.is_file() and not args.recompute_normalization:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload.get("point_feature_names", ())) != POINTWISE29_FEATURE_NAMES:
            raise RuntimeError(f"normalization feature contract mismatch: {path}")
        print(f"[normalize] reusing {path}", flush=True)
        return payload

    base = torch.load(
        args.data_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    moments = RunningFeatureMoments(29)
    patch_count = 0
    with torch.inference_mode():
        for path_index, shard_path in enumerate(train_paths):
            features, _ = _load_prepared_shard(shard_path)
            if args.normalization_max_patches:
                remaining = args.normalization_max_patches - patch_count
                if remaining <= 0:
                    break
                features = features[:remaining]
            expanded = build_pointwise29_features(features.to(device))
            moments.update(expanded)
            patch_count += features.shape[0]
            print(
                f"[normalize] shard={path_index + 1}/{len(train_paths)} "
                f"patches={patch_count}",
                flush=True,
            )
    input_mean, input_std = moments.mean_std()
    payload = {
        "schema_version": "rank10_pointwise29_normalization_v1",
        "point_feature_names": POINTWISE29_FEATURE_NAMES,
        "normalization_patches": patch_count,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": base["target_mean"],
        "target_std": base["target_std"],
    }
    torch.save(payload, path)
    print(f"[normalize] wrote {path}", flush=True)
    return payload


def _group_lasso(model: Pointwise29PointNet) -> torch.Tensor:
    weight = model.point_linear.weight
    penalties = []
    for indices in FEATURE_GROUPS.values():
        group = weight[:, list(indices)]
        penalties.append(group.square().sum().sqrt() / math.sqrt(len(indices)))
    return torch.stack(penalties).sum()


def _evaluate(
    model: Pointwise29PointNet,
    paths: list[Path],
    *,
    device: torch.device,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    batch_size: int,
    importance: bool = False,
    max_patches: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    baseline = _metrics_accumulator()
    ablations = {
        name: _metrics_accumulator() for name in FEATURE_GROUPS
    } if importance else {}
    model.eval()
    with torch.inference_mode():
        for path in paths:
            features, targets = _load_prepared_shard(path)
            for start in range(0, targets.shape[0], batch_size):
                if max_patches and baseline["count"] >= max_patches:
                    break
                stop = min(start + batch_size, targets.shape[0])
                if max_patches:
                    stop = min(stop, start + max_patches - baseline["count"])
                x = build_pointwise29_features(features[start:stop].to(device))
                y = targets[start:stop].to(device)
                x = ((x - input_mean) / input_std).clamp(-12, 12)
                prediction = model(x) * target_std + target_mean
                _update_metrics(baseline, prediction, y, target_mean)
                if importance:
                    for name, indices in FEATURE_GROUPS.items():
                        masked = x.clone()
                        masked[..., list(indices)] = 0.0
                        masked_prediction = model(masked) * target_std + target_mean
                        _update_metrics(
                            ablations[name], masked_prediction, y, target_mean
                        )
            if max_patches and baseline["count"] >= max_patches:
                break
    return _finalize_metrics(baseline), {
        name: _finalize_metrics(values) for name, values in ablations.items()
    }


def _weight_importance(model: Pointwise29PointNet) -> dict[str, Any]:
    weight = model.point_linear.weight.detach().cpu()
    feature_norms = weight.square().sum(dim=0).sqrt()
    group_norms = {
        name: float(weight[:, list(indices)].square().sum().sqrt())
        for name, indices in FEATURE_GROUPS.items()
    }
    total = sum(group_norms.values())
    return {
        "feature_column_l2": {
            name: float(feature_norms[index])
            for index, name in enumerate(POINTWISE29_FEATURE_NAMES)
        },
        "group_frobenius": group_norms,
        "group_fraction": {
            name: value / max(total, 1e-12) for name, value in group_norms.items()
        },
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_paths, validation_paths = _prepared_paths(args.data_dir)
    normalization = _normalization(args, train_paths, device)
    input_mean = normalization["input_mean"].to(device)
    input_std = normalization["input_std"].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    token_up_weight, token_up_bias, source_rl_checkpoint = _source_token_up(args.data_dir)

    model = Pointwise29PointNet(args.hidden_dim, args.patch_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    best_r2 = float("-inf")
    best_path = args.output_dir / "pointwise29_best.pt"
    start_epoch = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if resume.get("schema_version") != "rank10_pointwise29_pointnet_v1":
            raise RuntimeError(f"incompatible resume checkpoint: {args.resume_checkpoint}")
        if tuple(resume.get("point_feature_names", ())) != POINTWISE29_FEATURE_NAMES:
            raise RuntimeError("resume checkpoint has a different 29D feature contract")
        if int(resume.get("hidden_dim", -1)) != args.hidden_dim:
            raise RuntimeError("resume checkpoint --hidden-dim does not match")
        if int(resume.get("patch_hidden_dim", -1)) != args.patch_hidden_dim:
            raise RuntimeError("resume checkpoint --patch-hidden-dim does not match")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer_state = resume.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        resume_metrics = resume.get("metrics")
        if not isinstance(resume_metrics, dict):
            raise RuntimeError("resume checkpoint lacks metrics")
        start_epoch = int(resume_metrics.get("epoch", 0))
        metrics_path = args.output_dir / "pointwise29_metrics.json"
        if metrics_path.is_file():
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = [
                item for item in prior.get("history", [])
                if isinstance(item, dict) and int(item.get("epoch", 0)) <= start_epoch
            ]
        if not history:
            history = [dict(resume_metrics)]
        best_r2 = max(float(item["r2"]) for item in history)
        if best_path.is_file():
            existing_best = torch.load(best_path, map_location="cpu", weights_only=False)
            if isinstance(existing_best.get("metrics"), dict):
                best_r2 = max(best_r2, float(existing_best["metrics"]["r2"]))
        else:
            torch.save(resume, best_path)
        print(
            f"[train:pointwise29] resume={args.resume_checkpoint} epoch={start_epoch}",
            flush=True,
        )

    final_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
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
                x = build_pointwise29_features(features[indices].to(device))
                y = targets[indices].to(device)
                x = ((x - input_mean) / input_std).clamp(-12, 12)
                y = (y - target_mean) / target_std
                prediction = model(x)
                mse = F.mse_loss(prediction, y)
                cosine = 1.0 - F.cosine_similarity(
                    prediction, y, dim=-1, eps=1e-8
                ).mean()
                loss = (
                    mse
                    + args.cosine_weight * cosine
                    + args.group_lasso_weight * _group_lasso(model)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * indices.numel()
                total_count += indices.numel()

        metrics, _ = _evaluate(
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
            f"[train:pointwise29] epoch={epoch}/{final_epoch} "
            f"loss={metrics['train_loss']:.6f} val_r2={metrics['r2']:.6f} "
            f"val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        checkpoint = {
            "schema_version": "rank10_pointwise29_pointnet_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_dim": args.hidden_dim,
            "patch_hidden_dim": args.patch_hidden_dim,
            "point_feature_names": POINTWISE29_FEATURE_NAMES,
            "feature_groups": FEATURE_GROUPS,
            "normalization": {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in normalization.items()
            },
            "token_up_weight": token_up_weight,
            "token_up_bias": token_up_bias,
            "source_rl_checkpoint": source_rl_checkpoint,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "pointwise29_last.pt")
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(checkpoint, best_path)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    importance_baseline, ablations = _evaluate(
        model,
        validation_paths,
        device=device,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=args.batch_size,
        importance=True,
        max_patches=args.importance_max_patches,
    )
    importance = _weight_importance(model)
    importance["mean_replacement_group_ablation"] = {
        name: {
            **metrics,
            "r2_drop": importance_baseline["r2"] - metrics["r2"],
            "cosine_drop": importance_baseline["mean_cosine"] - metrics["mean_cosine"],
        }
        for name, metrics in ablations.items()
    }
    summary = {
        "architecture": (
            f"29->{args.hidden_dim}->max->{args.patch_hidden_dim}"
            f"->{args.hidden_dim}->10"
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strictly_pointwise_before_pool": True,
        "unsigned_distance": True,
        "signed_distance": False,
        "mesh_queries": False,
        "point_feature_names": POINTWISE29_FEATURE_NAMES,
        "best_validation_r2": best_r2,
        "history": history,
        "importance_subset_baseline": importance_baseline,
        "importance": importance,
    }
    metrics_path = args.output_dir / "pointwise29_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[analysis] wrote {metrics_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--patch-hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-lasso-weight", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--importance-max-patches", type=int, default=250_000)
    parser.add_argument(
        "--normalization-max-patches",
        type=int,
        default=1_000_000,
        help="number of training patches used for 29D input moments; 0 uses all",
    )
    parser.add_argument("--recompute-normalization", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if (
        args.epochs < 1
        or args.hidden_dim < 1
        or args.patch_hidden_dim < 1
        or args.batch_size < 1
        or args.importance_max_patches < 0
        or args.normalization_max_patches < 0
    ):
        parser.error("epochs, dimensions, batch size, and limits must be valid")
    return args


if __name__ == "__main__":
    train(parse_args())

#!/usr/bin/env python3
"""Fit rank-10 tokens with a tiny, interpretable unsigned 11D PointNet.

The input is the already prepared fast point-cloud dataset.  Each point uses
only patch-local geometry plus an unsigned nearest-opposite-cloud distance and
direction.  No mesh query or signed distance is used here.
"""

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

from scripts.train_rank10_minimal_pointnet import (
    FAST_POINT_FEATURE_NAMES,
    _load_prepared_shard,
    _prepared_paths,
    _source_token_up,
)


DEFAULT_DATA_DIR = (
    REPO_ROOT / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts/probes/rank10_patch_pointnet/tiny11_patch_mlp"
)

FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "relative_xyz": (0, 1, 2),
    "patch_center_xyz": (3, 4, 5),
    "unsigned_nearest_distance": (6,),
    "nearest_direction_xyz": (7, 8, 9),
    "patch_is_tool": (10,),
}


class TinyUnsignedPointNet(nn.Module):
    """One small point layer followed by a richer patch-level MLP."""

    def __init__(self, hidden_dim: int = 16, patch_hidden_dim: int = 32) -> None:
        super().__init__()
        self.point_linear = nn.Linear(11, hidden_dim)
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


def _group_lasso(model: TinyUnsignedPointNet) -> torch.Tensor:
    """Group sparsity over standardized input columns.

    Dividing by sqrt(group width) prevents XYZ groups from being penalized just
    because they contain three coordinates rather than one scalar.
    """

    weight = model.point_linear.weight
    penalties = []
    for indices in FEATURE_GROUPS.values():
        group = weight[:, list(indices)]
        penalties.append(group.square().sum().sqrt() / len(indices) ** 0.5)
    return torch.stack(penalties).sum()


def _metrics_accumulator() -> dict[str, Any]:
    return {
        "squared_error": torch.zeros(10, dtype=torch.float64),
        "target_square": torch.zeros(10, dtype=torch.float64),
        "cosine_sum": 0.0,
        "count": 0,
    }


def _update_metrics(
    accumulator: dict[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mean: torch.Tensor,
) -> None:
    error = (prediction - target).double().cpu()
    centered_target = (target - target_mean).double().cpu()
    accumulator["squared_error"] += error.square().sum(dim=0)
    accumulator["target_square"] += centered_target.square().sum(dim=0)
    accumulator["cosine_sum"] += F.cosine_similarity(
        prediction, target, dim=-1, eps=1e-8
    ).sum().item()
    accumulator["count"] += target.shape[0]


def _finalize_metrics(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = accumulator["count"]
    squared_error = accumulator["squared_error"]
    target_square = accumulator["target_square"]
    return {
        "patches": count,
        "mse": float(squared_error.sum() / (count * 10)),
        "r2": float(1.0 - squared_error.sum() / target_square.sum().clamp_min(1e-12)),
        "mean_cosine": accumulator["cosine_sum"] / count,
        "per_dimension_r2": (
            1.0 - squared_error / target_square.clamp_min(1e-12)
        ).tolist(),
    }


def _evaluate(
    model: TinyUnsignedPointNet,
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
                x = features[start:stop].to(device)
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


def _weight_importance(model: TinyUnsignedPointNet) -> dict[str, Any]:
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
            for index, name in enumerate(FAST_POINT_FEATURE_NAMES)
        },
        "group_frobenius": group_norms,
        "group_fraction": {
            name: value / max(total, 1e-12) for name, value in group_norms.items()
        },
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    train_paths, validation_paths = _prepared_paths(args.data_dir)
    normalization = torch.load(
        args.data_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    input_mean = normalization["input_mean"].to(device)
    input_std = normalization["input_std"].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    token_up_weight, token_up_bias, source_rl_checkpoint = _source_token_up(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = TinyUnsignedPointNet(args.hidden_dim, args.patch_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    best_r2 = float("-inf")
    best_path = args.output_dir / "tiny11_best.pt"
    start_epoch = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        if resume.get("schema_version") != "rank10_tiny_unsigned_pointcloud11_v2":
            raise RuntimeError(
                "resume checkpoint is not a compatible tiny unsigned PointNet v2: "
                f"{args.resume_checkpoint}"
            )
        if tuple(resume.get("point_feature_names", ())) != FAST_POINT_FEATURE_NAMES:
            raise RuntimeError("resume checkpoint has a different 11D feature contract")
        if int(resume.get("hidden_dim", -1)) != args.hidden_dim:
            raise RuntimeError("resume checkpoint --hidden-dim does not match")
        if int(resume.get("patch_hidden_dim", -1)) != args.patch_hidden_dim:
            raise RuntimeError("resume checkpoint --patch-hidden-dim does not match")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer_state = resume.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            optimizer_status = "restored"
        else:
            optimizer_status = "fresh (checkpoint predates optimizer saving)"
        resume_metrics = resume.get("metrics")
        if not isinstance(resume_metrics, dict):
            raise RuntimeError("resume checkpoint lacks training metrics")
        start_epoch = int(resume_metrics.get("epoch", 0))
        if start_epoch < 1:
            raise RuntimeError("resume checkpoint has an invalid epoch")
        metrics_path = args.output_dir / "tiny11_metrics.json"
        if metrics_path.is_file():
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = [
                item
                for item in prior.get("history", [])
                if isinstance(item, dict) and int(item.get("epoch", 0)) <= start_epoch
            ]
        if not history:
            history = [dict(resume_metrics)]
        best_r2 = max(float(item["r2"]) for item in history)
        if best_path.is_file():
            existing_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            existing_best_metrics = existing_best.get("metrics")
            if isinstance(existing_best_metrics, dict):
                best_r2 = max(best_r2, float(existing_best_metrics["r2"]))
        else:
            torch.save(resume, best_path)
            best_r2 = float(resume_metrics["r2"])
        print(
            f"[train:tiny11] resume={args.resume_checkpoint} epoch={start_epoch} "
            f"optimizer={optimizer_status}",
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
                x = features[indices].to(device)
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
            f"[train:tiny11] epoch={epoch}/{final_epoch} "
            f"loss={metrics['train_loss']:.6f} val_r2={metrics['r2']:.6f} "
            f"val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        checkpoint = {
            "schema_version": "rank10_tiny_unsigned_pointcloud11_v2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_dim": args.hidden_dim,
            "patch_hidden_dim": args.patch_hidden_dim,
            "point_feature_names": FAST_POINT_FEATURE_NAMES,
            "feature_groups": FEATURE_GROUPS,
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "token_up_weight": token_up_weight,
            "token_up_bias": token_up_bias,
            "source_rl_checkpoint": source_rl_checkpoint,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "tiny11_last.pt")
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
            "cosine_drop": (
                importance_baseline["mean_cosine"] - metrics["mean_cosine"]
            ),
        }
        for name, metrics in ablations.items()
    }
    summary = {
        "architecture": (
            f"11->{args.hidden_dim}->max->{args.patch_hidden_dim}"
            f"->{args.hidden_dim}->10"
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "unsigned_distance": True,
        "signed_distance": False,
        "mesh_queries": False,
        "group_lasso_weight": args.group_lasso_weight,
        "best_validation_r2": best_r2,
        "history": history,
        "importance_subset_baseline": importance_baseline,
        "importance": importance,
    }
    (args.output_dir / "tiny11_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[analysis] wrote {args.output_dir / 'tiny11_metrics.json'}", flush=True)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="continue for --epochs additional epochs from tiny11_last.pt",
    )
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
    ):
        parser.error("epochs, hidden size, and batch size must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())

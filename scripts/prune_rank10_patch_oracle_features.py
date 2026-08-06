#!/usr/bin/env python3
"""Ablate and prune the named patch-oracle factors without a PointNet."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.patch_oracle_probe import PATCH_ORACLE_FEATURE_NAMES, PatchOracleToRankToken


DEFAULT_PROBE_DIR = REPO_ROOT / "artifacts/probes/rank10_patch_oracle"


def semantic_feature_group(name: str) -> str:
    """Map individual coordinates/statistics to interpretable factor groups."""

    prefix_groups = (
        ("center_", "patch_center"),
        ("local_mean_", "patch_local_mean"),
        ("local_std_", "patch_local_std"),
        ("local_extent_", "patch_local_extent"),
        ("cov_", "patch_covariance"),
        ("pca_eigenvalue_", "patch_pca_spectrum"),
        ("canonical_normal_", "patch_canonical_normal"),
        ("signed_sdf_q", "signed_sdf_quantiles"),
        ("abs_sdf_q", "absolute_sdf_quantiles"),
        ("log_sdf_q", "log_sdf_quantiles"),
        ("contact_fraction_", "contact_threshold_fractions"),
        ("penetration_", "penetration_geometry"),
        ("soft_contact_", "soft_contact_scales"),
        ("soft_location_", "soft_contact_locations"),
        ("local_sdf_cov_", "local_sdf_covariance"),
        ("local_sdf_gradient_", "local_sdf_gradient"),
        ("closest_mesh_displacement_", "closest_mesh_displacement"),
        ("closest_mesh_direction_", "closest_mesh_direction"),
        ("closest_mesh_normal_", "closest_mesh_normal"),
        ("closest_displacement_pca_", "closest_displacement_pca"),
        ("closest_normal_pca_", "closest_normal_pca"),
        ("soft_mesh_normal_", "soft_mesh_normals"),
        ("quadratic_sdf_", "quadratic_sdf_model"),
        ("contact_centroid_local_", "contact_centroid"),
        ("penetration_centroid_local_", "penetration_geometry"),
        ("penetration_std_local_", "penetration_geometry"),
        ("patch_is_", "patch_body_type"),
    )
    for prefix, group in prefix_groups:
        if name.startswith(prefix):
            return group
    if name in {"rms_radius", "max_radius"}:
        return "patch_radius"
    if name in {"linearity", "planarity", "scattering"}:
        return "patch_shape_ratios"
    if name in {"signed_sdf_min", "signed_sdf_max", "signed_sdf_mean", "signed_sdf_std"}:
        return "signed_sdf_moments"
    if name in {"abs_sdf_min", "abs_sdf_mean", "abs_sdf_std"}:
        return "absolute_sdf_moments"
    if name in {"log_sdf_min", "log_sdf_max", "log_sdf_mean", "log_sdf_std"}:
        return "log_sdf_moments"
    if name in {"signed_sdf_skewness", "signed_sdf_excess_kurtosis", "signed_sdf_trimmed_mean"}:
        return "signed_sdf_higher_moments"
    if name == "patch_closest_normal_alignment":
        return "patch_closest_normal_alignment"
    return name


def build_feature_groups(names: tuple[str, ...]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(semantic_feature_group(name), []).append(index)
    return groups


def _load_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"].float(), payload["targets"].float()


def _paths_from_manifest(probe_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest_path = probe_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_names") != list(PATCH_ORACLE_FEATURE_NAMES):
        raise RuntimeError("probe dataset does not match the current named-feature contract")
    train = [probe_dir / item["path"] for item in manifest["train"]["shards"]]
    validation = [probe_dir / item["path"] for item in manifest["validation"]["shards"]]
    return train, validation


def _sample_validation(
    paths: list[Path],
    *,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    per_shard = max(1, (max_samples + len(paths) - 1) // len(paths))
    feature_parts = []
    target_parts = []
    for path in paths:
        features, targets = _load_shard(path)
        count = min(per_shard, features.shape[0])
        indices = torch.randperm(features.shape[0], generator=generator)[:count]
        feature_parts.append(features[indices])
        target_parts.append(targets[indices])
    features = torch.cat(feature_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    if features.shape[0] > max_samples:
        indices = torch.randperm(features.shape[0], generator=generator)[:max_samples]
        features = features[indices]
        targets = targets[indices]
    return features, targets


def _predict(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            prediction = model(features[start : start + batch_size].to(device))
            predictions.append((prediction * target_std + target_mean).cpu())
    return torch.cat(predictions, dim=0)


def _r2(prediction: torch.Tensor, target: torch.Tensor, target_mean: torch.Tensor) -> float:
    error = (prediction.double() - target.double()).square().sum()
    total = (target.double() - target_mean.cpu().double()).square().sum().clamp_min(1e-12)
    return float(1.0 - error / total)


def rank_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    _, validation_paths = _paths_from_manifest(args.probe_dir)
    normalization = torch.load(
        args.probe_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(
        args.probe_dir / "mlp_best.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    model = PatchOracleToRankToken().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    features, targets = _sample_validation(
        validation_paths,
        max_samples=args.importance_samples,
        seed=args.seed,
    )
    feature_mean = normalization["feature_mean"]
    feature_std = normalization["feature_std"]
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    normalized = ((features - feature_mean) / feature_std).clamp(-12, 12)
    baseline_prediction = _predict(
        model,
        normalized,
        device=device,
        batch_size=args.batch_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    baseline_r2 = _r2(baseline_prediction, targets, target_mean)
    groups = build_feature_groups(PATCH_ORACLE_FEATURE_NAMES)
    ranking = []
    for ordinal, (group, indices) in enumerate(groups.items(), start=1):
        ablated = normalized.clone()
        ablated[:, indices] = 0.0  # training-mean imputation
        prediction = _predict(
            model,
            ablated,
            device=device,
            batch_size=args.batch_size,
            target_mean=target_mean,
            target_std=target_std,
        )
        ablated_r2 = _r2(prediction, targets, target_mean)
        ranking.append(
            {
                "group": group,
                "feature_indices": indices,
                "feature_names": [PATCH_ORACLE_FEATURE_NAMES[index] for index in indices],
                "ablated_r2": ablated_r2,
                "r2_drop": baseline_r2 - ablated_r2,
            }
        )
        print(
            f"[importance] {ordinal}/{len(groups)} group={group} "
            f"r2_drop={baseline_r2 - ablated_r2:.6f}",
            flush=True,
        )
    ranking.sort(key=lambda item: item["r2_drop"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "validation_mean_ablation",
        "baseline_r2": baseline_r2,
        "samples": features.shape[0],
        "groups": ranking,
    }
    (args.output_dir / "group_importance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ranking


def _evaluate_subset(
    model: nn.Module,
    paths: list[Path],
    indices: torch.Tensor,
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
            for start in range(0, targets.shape[0], batch_size):
                x = features[start : start + batch_size, indices].to(device)
                y = targets[start : start + batch_size].to(device)
                x = ((x - feature_mean) / feature_std).clamp(-12, 12)
                prediction = model(x) * target_std + target_mean
                error = (prediction - y).double().cpu()
                centered = (y - target_mean).double().cpu()
                squared_error += error.square().sum(dim=0)
                target_square += centered.square().sum(dim=0)
                cosine_sum += F.cosine_similarity(prediction, y, dim=-1, eps=1e-8).sum().item()
                count += y.shape[0]
    return {
        "patches": count,
        "mse": float(squared_error.sum() / (count * 10)),
        "r2": float(1.0 - squared_error.sum() / target_square.sum().clamp_min(1e-12)),
        "mean_cosine": cosine_sum / count,
        "per_dimension_r2": (1.0 - squared_error / target_square.clamp_min(1e-12)).tolist(),
    }


def train_subsets(args: argparse.Namespace, ranking: list[dict[str, Any]] | None = None) -> None:
    train_paths, validation_paths = _paths_from_manifest(args.probe_dir)
    normalization = torch.load(
        args.probe_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    if ranking is None:
        payload = json.loads((args.output_dir / "group_importance.json").read_text(encoding="utf-8"))
        ranking = payload["groups"]
    requested = sorted(set(args.subset_groups), reverse=True)
    requested = [count for count in requested if 0 < count <= len(ranking)]
    if not requested:
        raise ValueError(f"no --subset-groups value is within 1..{len(ranking)}")
    device = torch.device(args.device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    rng = random.Random(args.seed)
    summary: dict[str, Any] = {}
    for group_count in requested:
        selected_groups = ranking[:group_count]
        selected_indices = sorted(
            index for group in selected_groups for index in group["feature_indices"]
        )
        index_tensor = torch.tensor(selected_indices, dtype=torch.long)
        feature_mean = normalization["feature_mean"][index_tensor].to(device)
        feature_std = normalization["feature_std"][index_tensor].to(device)
        model = PatchOracleToRankToken(input_dim=len(selected_indices)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        best_r2 = float("-inf")
        stale_epochs = 0
        history = []
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
                    x = features[batch_indices][:, index_tensor].to(device)
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
            metrics = _evaluate_subset(
                model,
                validation_paths,
                index_tensor,
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
                f"[train:top{group_count}] epoch={epoch}/{args.epochs} "
                f"features={len(selected_indices)} loss={metrics['train_loss']:.6f} "
                f"val_r2={metrics['r2']:.6f} val_cos={metrics['mean_cosine']:.6f}",
                flush=True,
            )
            checkpoint = {
                "schema_version": "rank10_patch_oracle_pruned_v1",
                "group_count": group_count,
                "selected_groups": [group["group"] for group in selected_groups],
                "selected_feature_indices": selected_indices,
                "selected_feature_names": [PATCH_ORACLE_FEATURE_NAMES[i] for i in selected_indices],
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            torch.save(checkpoint, args.output_dir / f"top{group_count}_last.pt")
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                stale_epochs = 0
                torch.save(checkpoint, args.output_dir / f"top{group_count}_best.pt")
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        summary[str(group_count)] = {
            "feature_count": len(selected_indices),
            "selected_groups": [group["group"] for group in selected_groups],
            "best_validation_r2": best_r2,
            "history": history,
        }
        (args.output_dir / "pruning_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rank", "train", "all"), default="all")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--importance-samples", type=int, default=32768)
    parser.add_argument("--subset-groups", default="24,20,16,12,8")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.probe_dir = args.probe_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.probe_dir / "feature_pruning"
    args.output_dir = args.output_dir.resolve()
    try:
        args.subset_groups = [int(value) for value in args.subset_groups.split(",")]
    except ValueError as exc:
        parser.error(f"--subset-groups must be comma-separated integers: {exc}")
    if args.importance_samples < 1 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("sample count, epochs, patience, and batch size must be positive")
    return args


def main() -> None:
    args = parse_args()
    ranking = rank_groups(args) if args.stage in {"rank", "all"} else None
    if args.stage in {"train", "all"}:
        train_subsets(args, ranking)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare fitted rank-10 and native direct-128 PointNet representations offline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENCODER_PATH = REPO_ROOT / "rsl_rl/modules/oracle_pointcloud_pointnet_encoder.py"
_ENCODER_SPEC = importlib.util.spec_from_file_location(
    "_offline_oracle_pointcloud_pointnet_encoder", _ENCODER_PATH
)
if _ENCODER_SPEC is None or _ENCODER_SPEC.loader is None:
    raise RuntimeError(f"cannot load encoder module: {_ENCODER_PATH}")
_ENCODER_MODULE = importlib.util.module_from_spec(_ENCODER_SPEC)
sys.modules[_ENCODER_SPEC.name] = _ENCODER_MODULE
_ENCODER_SPEC.loader.exec_module(_ENCODER_MODULE)
OraclePointCloudPointNetEncoder = _ENCODER_MODULE.OraclePointCloudPointNetEncoder


DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint lacks model_state_dict: {path}")
    prefix = "encoder."
    selected = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise RuntimeError(f"checkpoint lacks encoder.* parameters: {path}")
    return selected


def _load_encoder(path: Path, *, rank10: bool) -> OraclePointCloudPointNetEncoder:
    encoder = OraclePointCloudPointNetEncoder(
        feature_mode="fast11",
        use_rank10_bottleneck=rank10,
    )
    encoder.load_state_dict(_checkpoint_state(path), strict=True)
    encoder.eval()
    return encoder


def _paths(data_dir: Path, split: str, count: int) -> list[Path]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get(split)
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{manifest_path} lacks non-empty {split!r} entries")
    if count <= 0:
        raise ValueError("shard counts must be positive")
    return [data_dir / item["path"] for item in entries[:count]]


def _load_samples(
    paths: list[Path], max_patches_per_shard: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        point_features = payload.get("point_features")
        target = payload.get("targets")
        if not isinstance(point_features, torch.Tensor) or point_features.ndim != 3:
            raise RuntimeError(f"invalid point_features in {path}")
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise RuntimeError(f"invalid targets in {path}")
        take = min(point_features.shape[0], max_patches_per_shard)
        features.append(point_features[:take].float())
        targets.append(target[:take].float())
    return torch.cat(features), torch.cat(targets)


@torch.inference_mode()
def _encode(
    encoder: OraclePointCloudPointNetEncoder, point_features: torch.Tensor
) -> dict[str, torch.Tensor]:
    normalized = (
        (point_features - encoder.input_mean) / encoder.input_std
    ).clamp(-12.0, 12.0)
    pooled = encoder.point_mlp(normalized).amax(dim=-2)
    latent = encoder.patch_mlp(pooled)
    token = encoder.token_up(latent)
    return {"normalized": normalized, "pooled": pooled, "latent": latent, "token": token}


def _spectrum(values: torch.Tensor) -> dict[str, Any]:
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).flip(0).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    probability = eigenvalues / total
    cumulative = probability.cumsum(0)

    def rank_at(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, fraction).item() + 1)

    nonzero = probability[probability > 0]
    entropy_rank = torch.exp(-(nonzero * nonzero.log()).sum())
    participation_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    return {
        "dimension": values.shape[1],
        "entropy_effective_rank": float(entropy_rank),
        "participation_rank": float(participation_rank),
        "rank_90_percent_variance": rank_at(0.90),
        "rank_95_percent_variance": rank_at(0.95),
        "rank_99_percent_variance": rank_at(0.99),
        "top10_variance_fraction": float(probability[:10].sum()),
    }


def _ridge_fit(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    ridge: float,
) -> dict[str, Any]:
    x_mean = train_x.mean(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    x = train_x - x_mean
    y = train_y - y_mean
    scale = x.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    x = x / scale
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    weight = torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y)
    prediction = ((validation_x - x_mean) / scale) @ weight + y_mean
    error = prediction - validation_y
    centered_target = validation_y - train_y.mean(dim=0, keepdim=True)
    per_dimension_denominator = centered_target.square().sum(dim=0).clamp_min(1e-12)
    per_dimension_r2 = 1.0 - error.square().sum(dim=0) / per_dimension_denominator
    global_r2 = 1.0 - error.square().sum() / per_dimension_denominator.sum()
    cosine = torch.nn.functional.cosine_similarity(
        prediction, validation_y, dim=-1, eps=1e-8
    )
    return {
        "global_r2": float(global_r2),
        "mean_cosine": float(cosine.mean()),
        "per_dimension_r2": per_dimension_r2.tolist(),
    }


def _normalization_report(
    raw: torch.Tensor,
    encoded: dict[str, torch.Tensor],
    encoder: OraclePointCloudPointNetEncoder,
) -> dict[str, Any]:
    raw_flat = raw.flatten(0, 1)
    normalized_flat = encoded["normalized"].flatten(0, 1)
    first_weight = encoder.point_mlp[0].weight.detach()
    raw_std = raw_flat.std(dim=0)
    normalized_std = normalized_flat.std(dim=0)
    contribution = normalized_std * first_weight.square().sum(dim=0).sqrt()
    return {
        "checkpoint_input_mean": encoder.input_mean.tolist(),
        "checkpoint_input_std": encoder.input_std.tolist(),
        "observed_raw_mean": raw_flat.mean(dim=0).tolist(),
        "observed_raw_std": raw_std.tolist(),
        "observed_normalized_std": normalized_std.tolist(),
        "first_layer_input_contribution_scale": contribution.tolist(),
        "largest_to_smallest_nonzero_raw_std_ratio": float(
            raw_std.max() / raw_std[raw_std > 0].min()
        ),
        "largest_to_smallest_nonzero_normalized_std_ratio": float(
            normalized_std.max() / normalized_std[normalized_std > 0].min()
        ),
        "clip_fraction": float((normalized_flat.abs() >= 12.0).float().mean()),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    native = _load_encoder(args.native_checkpoint, rank10=False)
    fitted = _load_encoder(args.fitted_checkpoint, rank10=True)
    train_features, train_targets = _load_samples(
        _paths(args.data_dir, "train", args.train_shards),
        args.max_patches_per_shard,
    )
    validation_features, validation_targets = _load_samples(
        _paths(args.data_dir, "validation", args.validation_shards),
        args.max_patches_per_shard,
    )

    train_native = _encode(native, train_features)
    validation_native = _encode(native, validation_features)
    train_fitted = _encode(fitted, train_features)
    validation_fitted = _encode(fitted, validation_features)

    representations = {
        "native_direct128": (
            train_native["token"],
            validation_native["token"],
        ),
        "native_pooled128": (
            train_native["pooled"],
            validation_native["pooled"],
        ),
        "fitted_rank10": (
            train_fitted["latent"],
            validation_fitted["latent"],
        ),
        "fitted_reconstructed128": (
            train_fitted["token"],
            validation_fitted["token"],
        ),
    }
    probes = {
        name: _ridge_fit(train_x, train_targets, validation_x, validation_targets, args.ridge)
        for name, (train_x, validation_x) in representations.items()
    }
    spectra = {
        name: _spectrum(validation_x)
        for name, (_, validation_x) in representations.items()
    }
    return {
        "native_checkpoint": str(args.native_checkpoint),
        "fitted_checkpoint": str(args.fitted_checkpoint),
        "data_dir": str(args.data_dir),
        "train_patches": train_features.shape[0],
        "validation_patches": validation_features.shape[0],
        "normalization": {
            "native_direct128": _normalization_report(
                train_features, train_native, native
            ),
            "fitted_rank10": _normalization_report(
                train_features, train_fitted, fitted
            ),
        },
        "token_spectrum": spectra,
        "linear_probe_to_successful_teacher_rank10": probes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--fitted-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-shards", type=int, default=2)
    parser.add_argument("--validation-shards", type=int, default=2)
    parser.add_argument("--max-patches-per-shard", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/analysis/pointnet_offline_comparison.json",
    )
    args = parser.parse_args()
    args.native_checkpoint = args.native_checkpoint.resolve()
    args.fitted_checkpoint = args.fitted_checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[offline-pointnet] wrote {args.output}")


if __name__ == "__main__":
    main()

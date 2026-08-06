#!/usr/bin/env python3
"""Compare original and rebuilt fitted-PointNet representations on one dataset.

This is a strict, simulation-free comparison.  Both lineages consume the same
prepared validation patches.  It compares the fitted, DGN, and GG encoders with
held-out linear/orthogonal alignment, linear CKA, activation rank, and GG input
group ablations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_oracle_pointcloud_pointnet_encoder import (  # noqa: E402
    OraclePointCloudPointNetEncoder,
    _activation_analysis,
    _forward_core,
    _load_probe_state,
    _load_rl_encoder_state,
    _validation_paths,
)


OLD_PROBE = REPO_ROOT / (
    "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/"
    "fast_pointcloud11_best.pt"
)
OLD_DGN = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "20260719T092442Z/model_best.pt"
)
OLD_GG = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/"
    "20260719T202622Z/model_best.pt"
)
NEW_DATA = REPO_ROOT / (
    "artifacts/oracle_pointnet_rebuild_new200_d12/fast_pointcloud11/data"
)
NEW_PROBE = REPO_ROOT / (
    "artifacts/oracle_pointnet_rebuild_new200_d12/fast_pointcloud11/"
    "fast_pointcloud11_best.pt"
)
NEW_DGN = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "ce_prl_oracle_rebuild_d12_pointnet_dgn_5k/no-contact/"
    "oracle_pointcloud_pointnet/ce_prl_oracle_rebuild_d12_pointnet_dgn_5k/"
    "20260802T053747Z/model_best.pt"
)
NEW_GG = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "ce_prl_oracle_rebuild_d12_pointnet_gg_15k/no-contact/"
    "oracle_pointcloud_pointnet/ce_prl_oracle_rebuild_d12_pointnet_gg_15k/"
    "20260802T142604Z/model_best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/analysis/oracle_pointnet_first_round/"
    "original_vs_rebuilt_representations.json"
)


def _model(state: dict[str, torch.Tensor], device: torch.device):
    model = OraclePointCloudPointNetEncoder(
        num_points=512,
        num_patches=16,
        patch_size=32,
        feature_dim=128,
        feature_mode="fast11",
        use_rank10_bottleneck=True,
        token_mode="patches",
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _collect(
    state: dict[str, torch.Tensor],
    paths: list[Path],
    *,
    device: torch.device,
    max_patches: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model = _model(state, device)
    bottlenecks: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    count = 0
    with torch.inference_mode():
        for path in paths:
            features = torch.load(
                path, map_location="cpu", weights_only=False
            )["point_features"].float()
            features = features[: max_patches - count]
            for start in range(0, features.shape[0], batch_size):
                raw = features[start : start + batch_size].to(device)
                normalized = ((raw - model.input_mean) / model.input_std).clamp(-12, 12)
                bottleneck, token = _forward_core(model, normalized)
                bottlenecks.append(bottleneck.cpu())
                tokens.append(token.cpu())
                count += raw.shape[0]
                if count >= max_patches:
                    break
            if count >= max_patches:
                break
    if count != max_patches:
        raise RuntimeError(
            f"Prepared validation data has {count} patches; requested {max_patches}"
        )
    return {
        "bottleneck": torch.cat(bottlenecks, dim=0),
        "token": torch.cat(tokens, dim=0),
    }


def _effective_rank(x: torch.Tensor) -> dict[str, float]:
    centered = x.double() - x.double().mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy_rank = torch.exp(
        -(probability * probability.clamp_min(1e-30).log()).sum()
    )
    participation = (
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
    )
    return {
        "entropy_effective_rank": float(entropy_rank),
        "participation_ratio": float(participation),
    }


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    numerator = (x.T @ y).square().sum()
    denominator = torch.sqrt(
        (x.T @ x).square().sum() * (y.T @ y).square().sum()
    ).clamp_min(1e-30)
    return float(numerator / denominator)


def _prediction_metrics(reference: torch.Tensor, predicted: torch.Tensor) -> dict[str, float]:
    reference = reference.double()
    predicted = predicted.double()
    residual = (reference - predicted).square().sum()
    centered = reference - reference.mean(0, keepdim=True)
    r2 = 1.0 - residual / centered.square().sum().clamp_min(1e-30)
    cosine = F.cosine_similarity(reference, predicted, dim=-1, eps=1e-12).mean()
    return {
        "heldout_r2": float(r2),
        "heldout_mean_cosine": float(cosine),
        "heldout_relative_rmse": float(
            torch.sqrt(residual / reference.square().sum().clamp_min(1e-30))
        ),
    }


def _alignment(
    original: torch.Tensor,
    rebuilt: torch.Tensor,
    *,
    seed: int,
    ridge: float,
) -> dict[str, Any]:
    if original.shape != rebuilt.shape:
        raise RuntimeError(
            f"Representation shape mismatch: original={tuple(original.shape)} "
            f"rebuilt={tuple(rebuilt.shape)}"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(original.shape[0], generator=generator)
    split = int(0.7 * original.shape[0])
    if split <= original.shape[1] or split == original.shape[0]:
        raise RuntimeError("Insufficient patches for held-out alignment")
    train, test = order[:split], order[split:]
    x_train = rebuilt[train].double()
    y_train = original[train].double()
    x_test = rebuilt[test].double()
    y_test = original[test].double()
    x_mean = x_train.mean(0, keepdim=True)
    y_mean = y_train.mean(0, keepdim=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean

    u, _, vh = torch.linalg.svd(x_centered.T @ y_centered, full_matrices=False)
    orthogonal = u @ vh
    orthogonal_prediction = (x_test - x_mean) @ orthogonal + y_mean

    identity = torch.eye(x_train.shape[1], dtype=torch.float64)
    linear = torch.linalg.solve(
        x_centered.T @ x_centered + ridge * identity,
        x_centered.T @ y_centered,
    )
    linear_prediction = (x_test - x_mean) @ linear + y_mean
    return {
        "samples": original.shape[0],
        "train_samples": split,
        "test_samples": original.shape[0] - split,
        "linear_cka": _linear_cka(original, rebuilt),
        "direct_mean_cosine": float(
            F.cosine_similarity(original, rebuilt, dim=-1, eps=1e-12).mean()
        ),
        "original_rank": _effective_rank(original),
        "rebuilt_rank": _effective_rank(rebuilt),
        "orthogonal_procrustes": _prediction_metrics(
            y_test, orthogonal_prediction
        ),
        "ridge_linear": _prediction_metrics(y_test, linear_prediction),
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=NEW_DATA)
    parser.add_argument("--old-probe", type=Path, default=OLD_PROBE)
    parser.add_argument("--old-dgn", type=Path, default=OLD_DGN)
    parser.add_argument("--old-gg", type=Path, default=OLD_GG)
    parser.add_argument("--new-probe", type=Path, default=NEW_PROBE)
    parser.add_argument("--new-dgn", type=Path, default=NEW_DGN)
    parser.add_argument("--new-gg", type=Path, default=NEW_GG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patches", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.max_patches <= 128 or args.batch_size <= 0 or args.ridge <= 0:
        parser.error("Need --max-patches > 128, --batch-size > 0, and --ridge > 0")
    required = [
        args.data_dir / "manifest.json",
        args.old_probe,
        args.old_dgn,
        args.old_gg,
        args.new_probe,
        args.new_dgn,
        args.new_gg,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing required inputs:\n" + "\n".join(missing))
    return args


def main() -> None:
    args = _parse()
    device = torch.device(args.device)
    paths, feature_names = _validation_paths(args.data_dir)
    states = {
        "fitted": (
            _load_probe_state(args.old_probe)[0],
            _load_probe_state(args.new_probe)[0],
        ),
        "dgn": (
            _load_rl_encoder_state(args.old_dgn)[0],
            _load_rl_encoder_state(args.new_dgn)[0],
        ),
        "gg": (
            _load_rl_encoder_state(args.old_gg)[0],
            _load_rl_encoder_state(args.new_gg)[0],
        ),
    }
    result: dict[str, Any] = {
        "schema_version": "original_rebuilt_pointnet_representation_v1",
        "data_dir": str(args.data_dir),
        "feature_names": list(feature_names),
        "max_patches": args.max_patches,
        "stages": {},
    }
    for stage, (old_state, new_state) in states.items():
        old = _collect(
            old_state,
            paths,
            device=device,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        )
        new = _collect(
            new_state,
            paths,
            device=device,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        )
        result["stages"][stage] = {
            name: _alignment(
                old[name], new[name], seed=args.seed, ridge=args.ridge
            )
            for name in ("bottleneck", "token")
        }
    result["gg_group_ablation"] = {
        "original": _activation_analysis(
            _model(states["gg"][0], device),
            paths,
            device=device,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        )["group_mean_ablation"],
        "rebuilt": _activation_analysis(
            _model(states["gg"][1], device),
            paths,
            device=device,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        )["group_mean_ablation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {args.max_patches} identical patches at fitted/DGN/GG stages.")
    for stage, values in result["stages"].items():
        token = values["token"]
        print(
            f"{stage}: token CKA={token['linear_cka']:.4f} "
            f"linear held-out R2={token['ridge_linear']['heldout_r2']:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


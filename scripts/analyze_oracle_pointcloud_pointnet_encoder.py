#!/usr/bin/env python3
"""Analyze the fitted/RL-adapted fast11 PointNet encoder without simulation.

The analysis is intrinsic to the encoder.  It uses prepared validation patch
inputs, never evaluates policy success, and reports:

* standardized first-layer input weight norms;
* token/bottleneck changes when one semantic input group is set to its mean;
* activation-weighted importance of each of the ten bottleneck dimensions;
* effective ranks of bottleneck/token activations and every learned matrix;
* parameter drift from fitted probe -> full-YES RL -> GG RL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def _encoder_class():
    path = REPO_ROOT / "rsl_rl/modules/oracle_pointcloud_pointnet_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_pointcloud_pointnet_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load encoder module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointCloudPointNetEncoder


OraclePointCloudPointNetEncoder = _encoder_class()


DEFAULT_DATA = (
    REPO_ROOT
    / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/data"
)
DEFAULT_PROBE = (
    REPO_ROOT
    / "artifacts/probes/rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)
DEFAULT_FULL = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "20260719T092442Z/model_best.pt"
)
DEFAULT_GG = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/"
    "20260719T202622Z/model_best.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/analysis/oracle_pointcloud_pointnet_gg/encoder_importance.json"
)

FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "relative_xyz": (0, 1, 2),
    "patch_center_xyz": (3, 4, 5),
    "unsigned_distance": (6,),
    "direction_xyz": (7, 8, 9),
    "body_identity": (10,),
}


def _load_rl_encoder_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"RL checkpoint lacks model_state_dict: {path}")
    encoder = {
        key.removeprefix("encoder."): value.detach().float()
        for key, value in state.items()
        if key.startswith("encoder.") and isinstance(value, torch.Tensor)
    }
    if not encoder:
        raise RuntimeError(f"RL checkpoint has no encoder state: {path}")
    return encoder, {
        "path": str(path),
        "iteration": payload.get("iter"),
        "infos": payload.get("infos"),
    }


def _load_probe_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"invalid fitted probe checkpoint: {path}")
    state = {
        key: value.detach().float()
        for key, value in payload["model_state_dict"].items()
        if isinstance(value, torch.Tensor)
    }
    state["input_mean"] = payload["normalization"]["input_mean"].float()
    state["input_std"] = payload["normalization"]["input_std"].float()
    state["token_up.weight"] = payload["token_up_weight"].float()
    state["token_up.bias"] = payload["token_up_bias"].float()
    return state, {
        "path": str(path),
        "metrics": payload.get("metrics"),
        "point_feature_names": list(payload.get("point_feature_names", ())),
    }


def _effective_rank_from_singular_values(values: torch.Tensor) -> float:
    energy = values.double().square()
    probability = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def _covariance_effective_rank(sum_x: torch.Tensor, sum_xx: torch.Tensor, count: int) -> dict[str, Any]:
    mean = sum_x / count
    covariance = sum_xx / count - mean[:, None] * mean[None, :]
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    probability = eigenvalues / total
    entropy_rank = float((-(probability * probability.clamp_min(1e-30).log()).sum()).exp())
    participation = float(total.square() / eigenvalues.square().sum().clamp_min(1e-30))
    return {
        "entropy_effective_rank": entropy_rank,
        "participation_ratio": participation,
        "eigenvalues": eigenvalues.tolist(),
    }


def _cosine_and_relative_change(reference: torch.Tensor, changed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = F.cosine_similarity(reference, changed, dim=-1, eps=1e-8)
    relative = torch.linalg.vector_norm(changed - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(1e-8)
    return cosine, relative


def _forward_core(
    model: OraclePointCloudPointNetEncoder, normalized_inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = model.point_mlp(normalized_inputs).amax(dim=-2)
    bottleneck = model.patch_mlp(pooled)
    token = model.token_up(bottleneck)
    return bottleneck, token


def _validation_paths(data_dir: Path) -> tuple[list[Path], tuple[str, ...]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rank10_fast_pointcloud11_dataset_v1":
        raise RuntimeError(f"unexpected prepared data schema in {data_dir}")
    names = tuple(str(item) for item in manifest["point_feature_names"])
    if len(names) != 11:
        raise RuntimeError("fast11 analysis requires exactly eleven point features")
    paths = [data_dir / item["path"] for item in manifest["validation"]]
    return paths, names


def _activation_analysis(
    model: OraclePointCloudPointNetEncoder,
    paths: list[Path],
    *,
    device: torch.device,
    max_patches: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    bottleneck_sum = torch.zeros(10, dtype=torch.float64)
    bottleneck_square = torch.zeros(10, dtype=torch.float64)
    bottleneck_outer = torch.zeros(10, 10, dtype=torch.float64)
    token_sum = torch.zeros(128, dtype=torch.float64)
    token_outer = torch.zeros(128, 128, dtype=torch.float64)
    group_sums = {
        name: {"token_cosine": 0.0, "token_relative_change": 0.0,
               "bottleneck_cosine": 0.0, "bottleneck_relative_change": 0.0}
        for name in FEATURE_GROUPS
    }
    count = 0
    with torch.inference_mode():
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            features = payload["point_features"].float()
            remaining = max_patches - count
            if remaining <= 0:
                break
            features = features[:remaining]
            for start in range(0, features.shape[0], batch_size):
                raw = features[start : start + batch_size].to(device)
                normalized = ((raw - model.input_mean) / model.input_std).clamp(-12, 12)
                bottleneck, token = _forward_core(model, normalized)
                batch_count = token.shape[0]
                z64 = bottleneck.double().cpu()
                t64 = token.double().cpu()
                bottleneck_sum += z64.sum(0)
                bottleneck_square += z64.square().sum(0)
                bottleneck_outer += z64.T @ z64
                token_sum += t64.sum(0)
                token_outer += t64.T @ t64
                for name, indices in FEATURE_GROUPS.items():
                    changed_input = normalized.clone()
                    changed_input[..., list(indices)] = 0.0
                    changed_z, changed_token = _forward_core(model, changed_input)
                    token_cos, token_rel = _cosine_and_relative_change(token, changed_token)
                    z_cos, z_rel = _cosine_and_relative_change(bottleneck, changed_z)
                    group_sums[name]["token_cosine"] += float(token_cos.sum())
                    group_sums[name]["token_relative_change"] += float(token_rel.sum())
                    group_sums[name]["bottleneck_cosine"] += float(z_cos.sum())
                    group_sums[name]["bottleneck_relative_change"] += float(z_rel.sum())
                count += batch_count
                if count >= max_patches:
                    break
    if count == 0:
        raise RuntimeError("no validation patches were analyzed")

    bottleneck_mean = bottleneck_sum / count
    bottleneck_variance = (
        bottleneck_square / count - bottleneck_mean.square()
    ).clamp_min(0)
    bottleneck_rms = (bottleneck_square / count).sqrt()
    token_up = model.token_up.weight.detach().double().cpu()
    column_norm = torch.linalg.vector_norm(token_up, dim=0)
    contribution = bottleneck_rms * column_norm
    contribution_fraction = contribution / contribution.sum().clamp_min(1e-30)
    bottleneck_dimensions = []
    for index in range(10):
        bottleneck_dimensions.append(
            {
                "dimension": index,
                "activation_mean": float(bottleneck_mean[index]),
                "activation_std": float(bottleneck_variance[index].sqrt()),
                "activation_rms": float(bottleneck_rms[index]),
                "token_up_column_norm": float(column_norm[index]),
                "rms_token_contribution": float(contribution[index]),
                "contribution_fraction": float(contribution_fraction[index]),
            }
        )
    bottleneck_dimensions.sort(
        key=lambda item: item["rms_token_contribution"], reverse=True
    )
    group_results = []
    for name, values in group_sums.items():
        result = {"group": name, "channels": list(FEATURE_GROUPS[name])}
        result.update({key: value / count for key, value in values.items()})
        group_results.append(result)
    group_results.sort(key=lambda item: item["token_relative_change"], reverse=True)
    return {
        "patches": count,
        "group_mean_ablation": group_results,
        "bottleneck_dimensions": bottleneck_dimensions,
        "bottleneck_activation_rank": _covariance_effective_rank(
            bottleneck_sum, bottleneck_outer, count
        ),
        "token_activation_rank": _covariance_effective_rank(token_sum, token_outer, count),
    }


def _weight_analysis(state: dict[str, torch.Tensor], names: tuple[str, ...]) -> dict[str, Any]:
    first = state["point_mlp.0.weight"].double()
    column_norms = torch.linalg.vector_norm(first, dim=0)
    channel_rows = [
        {"feature": names[index], "column_norm": float(column_norms[index])}
        for index in range(len(names))
    ]
    channel_rows.sort(key=lambda item: item["column_norm"], reverse=True)
    group_rows = []
    for name, indices in FEATURE_GROUPS.items():
        norm = torch.linalg.vector_norm(first[:, list(indices)])
        group_rows.append(
            {
                "group": name,
                "frobenius_norm": float(norm),
                "norm_per_sqrt_channel": float(norm / math.sqrt(len(indices))),
            }
        )
    group_rows.sort(key=lambda item: item["norm_per_sqrt_channel"], reverse=True)
    matrices = []
    for key, value in state.items():
        if key.endswith("weight") and value.ndim == 2:
            singular = torch.linalg.svdvals(value.double())
            matrices.append(
                {
                    "name": key,
                    "shape": list(value.shape),
                    "spectral_norm": float(singular[0]),
                    "condition_number": float(singular[0] / singular[-1].clamp_min(1e-30)),
                    "effective_rank": _effective_rank_from_singular_values(singular),
                    "singular_values": singular.tolist(),
                }
            )
    return {
        "first_layer_channels": channel_rows,
        "first_layer_groups": group_rows,
        "matrices": matrices,
    }


def _parameter_drift(
    source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    common = sorted(set(source).intersection(target))
    results = []
    for prefix in ("point_mlp.", "patch_mlp.", "token_up."):
        keys = [
            key for key in common
            if key.startswith(prefix) and source[key].is_floating_point()
        ]
        before = torch.cat([source[key].reshape(-1).double() for key in keys])
        after = torch.cat([target[key].reshape(-1).double() for key in keys])
        results.append(
            {
                "module": prefix.rstrip("."),
                "parameters": before.numel(),
                "relative_l2_change": float(
                    torch.linalg.vector_norm(after - before)
                    / torch.linalg.vector_norm(before).clamp_min(1e-30)
                ),
                "cosine": float(F.cosine_similarity(before, after, dim=0)),
            }
        )
    return results


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    validation_paths, feature_names = _validation_paths(args.data_dir)
    probe_state, probe_metadata = _load_probe_state(args.probe_checkpoint)
    full_state, full_metadata = _load_rl_encoder_state(args.full_checkpoint)
    gg_state, gg_metadata = _load_rl_encoder_state(args.gg_checkpoint)
    model = OraclePointCloudPointNetEncoder(
        num_points=512,
        num_patches=16,
        patch_size=32,
        feature_dim=128,
        feature_mode="fast11",
        use_rank10_bottleneck=True,
        token_mode="patches",
    ).to(args.device)
    model.load_state_dict(gg_state, strict=True)
    result = {
        "schema_version": "oracle_pointcloud_encoder_importance_v1",
        "analysis_target": "encoder_tokens_not_policy_success",
        "feature_names": list(feature_names),
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "probe": probe_metadata,
        "full_yes": full_metadata,
        "gg": gg_metadata,
        "weight_analysis": _weight_analysis(gg_state, feature_names),
        "parameter_drift": {
            "probe_to_full": _parameter_drift(probe_state, full_state),
            "full_to_gg": _parameter_drift(full_state, gg_state),
            "probe_to_gg": _parameter_drift(probe_state, gg_state),
        },
        "activation_analysis": _activation_analysis(
            model,
            validation_paths,
            device=torch.device(args.device),
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _print_summary(result: dict[str, Any], output: Path) -> None:
    activation = result["activation_analysis"]
    print(f"patches={activation['patches']}")
    print("\nInput groups by mean-ablation token change:")
    for item in activation["group_mean_ablation"]:
        print(
            f"  {item['group']:<20} relative_change={item['token_relative_change']:.6f} "
            f"cosine={item['token_cosine']:.6f}"
        )
    print("\nBottleneck dimensions by RMS token contribution:")
    for item in activation["bottleneck_dimensions"]:
        print(
            f"  dim={item['dimension']} contribution={item['rms_token_contribution']:.6f} "
            f"fraction={item['contribution_fraction']:.4f} "
            f"activation_std={item['activation_std']:.6f}"
        )
    print(
        "\nBottleneck effective rank: "
        f"{activation['bottleneck_activation_rank']['entropy_effective_rank']:.4f}"
    )
    print(
        "Token effective rank: "
        f"{activation['token_activation_rank']['entropy_effective_rank']:.4f}"
    )
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probe-checkpoint", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--gg-checkpoint", type=Path, default=DEFAULT_GG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-patches", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    for name in ("data_dir", "probe_checkpoint", "full_checkpoint", "gg_checkpoint", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.max_patches <= 0 or args.batch_size <= 0:
        parser.error("--max-patches and --batch-size must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    analysis = analyze(parsed)
    _print_summary(analysis, parsed.output)

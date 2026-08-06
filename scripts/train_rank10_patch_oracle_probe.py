#!/usr/bin/env python3
"""Extract patch-local oracle factors and fit them to the trained rank-10 token.

Extraction uses true signed point-to-opposite-mesh SDF from the contact dataset.
The probe receives one patch at a time; no feature can inspect another patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.dataset import NewPretrainDataset, collect_pt_files
from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from pretrain.patch_oracle_probe import (
    DeepPatchOracleToRankToken,
    PATCH_ORACLE_FEATURE_NAMES,
    PatchOracleToRankToken,
    build_patch_oracle_features,
)
from pretrain.rank10_pointnet_contract import (
    PATCH_METADATA_NAMES,
    POINT_FEATURE_NAMES,
    build_rank10_pointnet_source,
)
from pretrain.train import collate_fn
from utils.geometry.sdf import mutual_signed_sdf_geometry_env_frame


DEFAULT_RL_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_unicorn_ours_encoder_bottleneck_rank10_full_yes_5k/"
    "no-contact/TCE/"
    "panda_general_unicorn_ours_encoder_bottleneck_rank10_full_yes_5k/"
    "20260718T131611Z/model_best.pt"
)
DEFAULT_DATA_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_generated_gripper/"
    "fdc5885d5d2a55727c19a6d984557275d2a7f5e48e70f6ef32e01a5bbc03daa3"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/probes/rank10_patch_oracle"
AGGREGATE_CONTRACT = "aggregate_v1"
POINTNET_CONTRACT = "pointnet_v2"


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.square_sum = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1, values.shape[-1]).to(device="cpu", dtype=torch.float64)
        self.count += flat.shape[0]
        self.sum += flat.sum(dim=0)
        self.square_sum += flat.square().sum(dim=0)

    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("normalization requires at least two patches")
        mean = self.sum / self.count
        variance = (self.square_sum / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float().clamp_min(1e-6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rl_encoder_and_bottleneck(
    checkpoint_path: Path,
    *,
    device: torch.device,
    vit_attention_mode: str,
) -> tuple[TCEPointCloudEncoder, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"RL checkpoint must contain a mapping: {checkpoint_path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"RL checkpoint lacks model_state_dict: {checkpoint_path}")
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    down_weight = state.get("encoder_token_bottleneck_down.weight")
    down_bias = state.get("encoder_token_bottleneck_down.bias")
    if not isinstance(down_weight, torch.Tensor) or not isinstance(down_bias, torch.Tensor):
        raise RuntimeError("RL checkpoint lacks the trainable encoder-token bottleneck")
    if tuple(down_weight.shape) != (10, 128) or tuple(down_bias.shape) != (10,):
        raise RuntimeError(
            "expected the rank-10 bottleneck, got "
            f"weight={tuple(down_weight.shape)} bias={tuple(down_bias.shape)}"
        )
    encoder = TCEPointCloudEncoder(
        TCEPointCloudEncoderCfg(
            num_pts=512,
            patch_size=32,
            encoder_channel=128,
            vit_depth=12,
            vit_heads=4,
            freeze=True,
            vit_attention_mode=vit_attention_mode,
        )
    )
    encoder.load_state_dict(dict(encoder_state), strict=True)
    encoder.to(device).eval()
    return (
        encoder,
        down_weight.to(device=device, dtype=torch.float32),
        down_bias.to(device=device, dtype=torch.float32),
        checkpoint,
    )


def _file_identity(path: str) -> tuple[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"contact file is not a mapping: {path}")
    return str(payload.get("object_id")), str(payload.get("tool_id"))


def _grouped_file_split(
    data_dir: Path,
    *,
    max_files: int,
    validation_ratio: float,
    seed: int,
    use_geometry_candidates: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    files = collect_pt_files(
        data_dir,
        use_geometry_candidates=use_geometry_candidates,
    )
    if not files:
        raise RuntimeError(f"no contact .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files > 0:
        files = files[:max_files]
    records = [(path, *_file_identity(path)) for path in files]
    objects = sorted({object_id for _, object_id, _ in records})
    tools = sorted({tool_id for _, _, tool_id in records})
    rng.shuffle(objects)
    rng.shuffle(tools)
    validation_objects = set(objects[: max(1, int(len(objects) * validation_ratio))])
    validation_tools = set(tools[: max(1, int(len(tools) * validation_ratio))])
    validation_files = [
        path
        for path, object_id, tool_id in records
        if object_id in validation_objects or tool_id in validation_tools
    ]
    train_files = [
        path
        for path, object_id, tool_id in records
        if object_id not in validation_objects and tool_id not in validation_tools
    ]
    if not train_files or not validation_files:
        raise RuntimeError(
            "grouped object/tool split is empty; increase --max-files or change --validation-ratio"
        )
    metadata = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "train_files": len(train_files),
        "validation_files": len(validation_files),
        "validation_object_ids": sorted(validation_objects),
        "validation_tool_ids": sorted(validation_tools),
    }
    return train_files, validation_files, metadata


def _make_dataset(
    files: Iterable[str],
    *,
    num_precontact_steps: int,
    translation_range_m: float,
    max_contacts_per_file: int,
) -> NewPretrainDataset:
    return NewPretrainDataset(
        files,
        augment=False,
        require_movement=False,
        num_points=512,
        num_precontact_steps=num_precontact_steps,
        allow_mock_physics=False,
        noise_max_trans=translation_range_m,
        noise_max_rot_deg=0.0,
        noise_max_retries=10,
        floor_eps=0.0,
        validation_seed=12345,
        denoise_target_mode="one_step",
        tool_mesh_contract="adjusted_decomposed_mesh",
        include_meshes=True,
        max_contacts_per_file=max_contacts_per_file,
    )


def _gather_patches(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1)
    return values[batch, indices]


def _extract_batch(
    batch: Mapping[str, Any],
    *,
    encoder: TCEPointCloudEncoder,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    device: torch.device,
    sdf_chunk_size: int,
    output_contract: str,
) -> dict[str, torch.Tensor]:
    tool_rotated = batch["tool_points_E_k"].to(device, non_blocking=True)
    object_rotated = batch["object_points_E_k"].to(device, non_blocking=True)
    relative_translation = batch["rel_tool_object_t_k"].to(device, non_blocking=True)
    tool_translation = batch["tool_translation_E_k"].to(device, non_blocking=True)
    object_translation = batch["object_bbox_center_E"].to(device, non_blocking=True)

    batch_size, timesteps, num_points, point_dim = tool_rotated.shape
    tool_encoder_points = tool_rotated + relative_translation.unsqueeze(-2)
    object_encoder_points = object_rotated
    flat_tool = tool_encoder_points.reshape(batch_size * timesteps, num_points, point_dim)
    flat_object = object_encoder_points.reshape(batch_size * timesteps, num_points, point_dim)

    with torch.inference_mode():
        encoded = encoder.encode(flat_tool, flat_object)
        latent = F.linear(encoded.fused_tokens, down_weight, down_bias)
        tool_query_env = tool_rotated + tool_translation.unsqueeze(-2)
        object_query_env = object_rotated + object_translation[:, None, None, :]
        (
            tool_sdf,
            tool_displacement,
            tool_normal,
            object_sdf,
            object_displacement,
            object_normal,
        ) = mutual_signed_sdf_geometry_env_frame(
            tool_query_points_E=tool_query_env,
            object_query_points_E=object_query_env,
            object_mesh_vertices=batch["object_mesh_vertices"],
            object_mesh_faces=batch["object_mesh_faces"],
            tool_mesh_vertices=batch["tool_mesh_vertices"],
            tool_mesh_faces=batch["tool_mesh_faces"],
            object_rotation_E=batch["object_rotation_E"].to(device, non_blocking=True),
            object_bbox_center_E=object_translation,
            tool_rotation_E_k=batch["tool_rotation_E_k"].to(device, non_blocking=True),
            tool_translation_E_k=tool_translation,
            chunk_size=sdf_chunk_size,
            backend="kaolin",
        )

        flat_tool_sdf = tool_sdf.reshape(batch_size * timesteps, num_points)
        flat_object_sdf = object_sdf.reshape(batch_size * timesteps, num_points)
        tool_patches = _gather_patches(flat_tool, encoded.tool_patch_idx)
        object_patches = _gather_patches(flat_object, encoded.obj_patch_idx)
        tool_patch_sdf = _gather_patches(flat_tool_sdf, encoded.tool_patch_idx)
        object_patch_sdf = _gather_patches(flat_object_sdf, encoded.obj_patch_idx)
        flat_tool_displacement = tool_displacement.reshape(
            batch_size * timesteps, num_points, 3
        )
        flat_object_displacement = object_displacement.reshape(
            batch_size * timesteps, num_points, 3
        )
        flat_tool_normal = tool_normal.reshape(batch_size * timesteps, num_points, 3)
        flat_object_normal = object_normal.reshape(batch_size * timesteps, num_points, 3)
        tool_patch_displacement = _gather_patches(
            flat_tool_displacement, encoded.tool_patch_idx
        )
        object_patch_displacement = _gather_patches(
            flat_object_displacement, encoded.obj_patch_idx
        )
        tool_patch_normal = _gather_patches(flat_tool_normal, encoded.tool_patch_idx)
        object_patch_normal = _gather_patches(flat_object_normal, encoded.obj_patch_idx)
        patch_points = torch.cat((tool_patches, object_patches), dim=1)
        patch_centers = torch.cat(
            (encoded.tool_patch_centers, encoded.obj_patch_centers), dim=1
        )
        patch_sdf = torch.cat((tool_patch_sdf, object_patch_sdf), dim=1)
        patch_displacement = torch.cat(
            (tool_patch_displacement, object_patch_displacement), dim=1
        )
        patch_normal = torch.cat((tool_patch_normal, object_patch_normal), dim=1)
        patch_is_tool = torch.cat(
            (
                torch.ones_like(tool_patch_sdf[..., 0], dtype=torch.bool),
                torch.zeros_like(object_patch_sdf[..., 0], dtype=torch.bool),
            ),
            dim=1,
        )
        features = build_patch_oracle_features(
            patch_points=patch_points,
            patch_centers=patch_centers,
            signed_sdf=patch_sdf,
            closest_displacement=patch_displacement,
            closest_normal=patch_normal,
            patch_is_tool=patch_is_tool,
        )
        targets = latent.reshape(-1, latent.shape[-1]).cpu()
        if output_contract == POINTNET_CONTRACT:
            point_features, patch_metadata = build_rank10_pointnet_source(
                patch_points=patch_points,
                patch_centers=patch_centers,
                signed_sdf=patch_sdf,
                closest_displacement=patch_displacement,
                closest_normal=patch_normal,
                patch_is_tool=patch_is_tool,
            )
            return {
                "point_features": point_features.reshape(
                    -1, point_features.shape[-2], point_features.shape[-1]
                ).cpu(),
                "patch_metadata": patch_metadata.reshape(
                    -1, patch_metadata.shape[-1]
                ).cpu(),
                "targets": targets,
            }
        if output_contract != AGGREGATE_CONTRACT:
            raise ValueError(f"Unsupported extraction contract: {output_contract}")
        return {
            "features": features.reshape(-1, features.shape[-1]).cpu(),
            "targets": targets,
            "patch_is_tool": patch_is_tool.reshape(-1).cpu(),
        }


def _flush_shard(
    directory: Path,
    shard_index: int,
    payload_parts: dict[str, list[torch.Tensor]],
) -> tuple[Path, int]:
    payload = {
        key: torch.cat(parts, dim=0)
        for key, parts in payload_parts.items()
    }
    if "targets" not in payload:
        raise RuntimeError("rank-10 extraction shard lacks targets")
    path = directory / f"shard_{shard_index:06d}.pt"
    torch.save(payload, path)
    return path, payload["targets"].shape[0]


def _extract_split(
    name: str,
    dataset: NewPretrainDataset,
    *,
    output_dir: Path,
    encoder: TCEPointCloudEncoder,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sdf_chunk_size: int,
    shard_patches: int,
    feature_moments: RunningMoments | None,
    target_moments: RunningMoments | None,
    output_contract: str,
) -> dict[str, Any]:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    payload_parts: dict[str, list[torch.Tensor]] = {}
    buffered = 0
    total = 0
    shards = []
    for batch_index, batch in enumerate(loader, start=1):
        payload = _extract_batch(
            batch,
            encoder=encoder,
            down_weight=down_weight,
            down_bias=down_bias,
            device=device,
            sdf_chunk_size=sdf_chunk_size,
            output_contract=output_contract,
        )
        if feature_moments is not None:
            if "features" not in payload:
                raise RuntimeError(
                    "aggregate feature moments require aggregate extraction"
                )
            feature_moments.update(payload["features"])
        if target_moments is not None:
            target_moments.update(payload["targets"])
        for key, value in payload.items():
            payload_parts.setdefault(key, []).append(value)
        buffered += payload["targets"].shape[0]
        if buffered >= shard_patches or batch_index == len(loader):
            path, count = _flush_shard(
                split_dir,
                len(shards),
                payload_parts,
            )
            shards.append({"path": str(path.relative_to(output_dir)), "patches": count})
            total += count
            payload_parts.clear()
            buffered = 0
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"[extract:{name}] samples={min(batch_index * batch_size, len(dataset))}/"
                f"{len(dataset)} patches={total + buffered}",
                flush=True,
            )
    return {"samples": len(dataset), "patches": total, "shards": shards}


def extract(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output exists: {args.output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    encoder, down_weight, down_bias, checkpoint = _load_rl_encoder_and_bottleneck(
        args.rl_checkpoint,
        device=device,
        vit_attention_mode=args.vit_attention_mode,
    )
    train_files, validation_files, split_metadata = _grouped_file_split(
        args.data_dir,
        max_files=args.max_files,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        use_geometry_candidates=args.use_geometry_candidates,
    )
    (args.output_dir / "split.json").write_text(
        json.dumps(split_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_dataset = _make_dataset(
        train_files,
        num_precontact_steps=args.num_precontact_steps,
        translation_range_m=args.translation_range_m,
        max_contacts_per_file=args.max_contacts_per_file,
    )
    validation_dataset = _make_dataset(
        validation_files,
        num_precontact_steps=args.num_precontact_steps,
        translation_range_m=args.translation_range_m,
        max_contacts_per_file=args.max_contacts_per_file,
    )
    feature_moments = (
        RunningMoments(len(PATCH_ORACLE_FEATURE_NAMES))
        if args.extraction_contract == AGGREGATE_CONTRACT
        else None
    )
    target_moments = RunningMoments(10)
    train_info = _extract_split(
        "train",
        train_dataset,
        output_dir=args.output_dir,
        encoder=encoder,
        down_weight=down_weight,
        down_bias=down_bias,
        device=device,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        sdf_chunk_size=args.sdf_chunk_size,
        shard_patches=args.shard_patches,
        feature_moments=feature_moments,
        target_moments=target_moments,
        output_contract=args.extraction_contract,
    )
    validation_info = _extract_split(
        "validation",
        validation_dataset,
        output_dir=args.output_dir,
        encoder=encoder,
        down_weight=down_weight,
        down_bias=down_bias,
        device=device,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        sdf_chunk_size=args.sdf_chunk_size,
        shard_patches=args.shard_patches,
        feature_moments=None,
        target_moments=None,
        output_contract=args.extraction_contract,
    )
    target_mean, target_std = target_moments.mean_std()
    normalization = {
        "target_mean": target_mean,
        "target_std": target_std,
    }
    if feature_moments is not None:
        feature_mean, feature_std = feature_moments.mean_std()
        normalization.update(
            {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
            }
        )
    torch.save(normalization, args.output_dir / "normalization.pt")
    manifest = {
        "schema_version": (
            "rank10_patch_oracle_probe_dataset_v1"
            if args.extraction_contract == AGGREGATE_CONTRACT
            else "rank10_patch_oracle_probe_dataset_v2"
        ),
        "rl_checkpoint": str(args.rl_checkpoint.resolve()),
        "rl_checkpoint_sha256": _sha256(args.rl_checkpoint),
        "rl_checkpoint_iteration": checkpoint.get("iter"),
        "data_dir": str(args.data_dir.resolve()),
        **(
            {
                "feature_names": list(PATCH_ORACLE_FEATURE_NAMES),
                "feature_dim": len(PATCH_ORACLE_FEATURE_NAMES),
            }
            if args.extraction_contract == AGGREGATE_CONTRACT
            else {
                "point_feature_names": list(POINT_FEATURE_NAMES),
                "patch_metadata_names": list(PATCH_METADATA_NAMES),
            }
        ),
        "target_dim": 10,
        "strict_patch_local": True,
        "patch_equivariant": True,
        "cross_patch_features": False,
        "num_precontact_steps": args.num_precontact_steps,
        "translation_range_m": args.translation_range_m,
        "use_geometry_candidates": args.use_geometry_candidates,
        "max_contacts_per_file": args.max_contacts_per_file,
        "split": split_metadata,
        "train": train_info,
        "validation": validation_info,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[extract] saved {args.output_dir.resolve()}", flush=True)


def _load_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"].float(), payload["targets"].float()


def _evaluate(
    model: nn.Module,
    shard_paths: list[Path],
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
        for path in shard_paths:
            features, targets = _load_shard(path)
            for start in range(0, features.shape[0], batch_size):
                x = features[start : start + batch_size].to(device)
                y = targets[start : start + batch_size].to(device)
                x = ((x - feature_mean) / feature_std).clamp(-12, 12)
                prediction_normalized = model(x)
                prediction = prediction_normalized * target_std + target_mean
                error = (prediction - y).double().cpu()
                centered_target = (y - target_mean).double().cpu()
                squared_error += error.square().sum(dim=0)
                target_square += centered_target.square().sum(dim=0)
                cosine_sum += F.cosine_similarity(prediction, y, dim=-1, eps=1e-8).sum().item()
                count += y.shape[0]
    per_dimension_r2 = 1.0 - squared_error / target_square.clamp_min(1e-12)
    return {
        "patches": count,
        "mse": float(squared_error.sum() / (count * 10)),
        "r2": float(1.0 - squared_error.sum() / target_square.sum().clamp_min(1e-12)),
        "mean_cosine": cosine_sum / count,
        "per_dimension_r2": per_dimension_r2.tolist(),
    }


def _train_model(
    name: str,
    model: nn.Module,
    *,
    args: argparse.Namespace,
    train_paths: list[Path],
    validation_paths: list[Path],
    normalization: Mapping[str, torch.Tensor],
    feature_names: tuple[str, ...] = PATCH_ORACLE_FEATURE_NAMES,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[train:{name}] model={type(model).__name__} parameters={parameter_count:,}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    feature_mean = normalization["feature_mean"].to(device)
    feature_std = normalization["feature_std"].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    best_r2 = float("-inf")
    history = []
    rng = random.Random(args.seed)
    start_epoch = 0
    resumed_optimizer = False
    if args.resume_checkpoint is not None:
        payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"resume checkpoint is not a mapping: {args.resume_checkpoint}")
        if payload.get("model_name") != name:
            raise RuntimeError(
                f"resume checkpoint model_name={payload.get('model_name')!r} "
                f"does not match requested probe {name!r}"
            )
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError(f"resume checkpoint lacks model_state_dict: {args.resume_checkpoint}")
        model.load_state_dict(dict(state), strict=True)
        checkpoint_metrics = payload.get("metrics")
        if not isinstance(checkpoint_metrics, Mapping):
            raise RuntimeError(f"resume checkpoint lacks metrics: {args.resume_checkpoint}")
        start_epoch = int(checkpoint_metrics.get("epoch", 0))
        if start_epoch < 1:
            raise RuntimeError(f"resume checkpoint has invalid epoch: {args.resume_checkpoint}")
        optimizer_state = payload.get("optimizer_state_dict")
        if isinstance(optimizer_state, Mapping):
            optimizer.load_state_dict(dict(optimizer_state))
            resumed_optimizer = True
        python_rng_state = payload.get("python_rng_state")
        torch_rng_state = payload.get("torch_rng_state")
        if python_rng_state is not None:
            rng.setstate(python_rng_state)
        if isinstance(torch_rng_state, torch.Tensor):
            torch.set_rng_state(torch_rng_state)

        metrics_path = args.output_dir / f"{name}_metrics.json"
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            previous_history = previous.get("history", [])
            if isinstance(previous_history, list):
                history = [
                    item
                    for item in previous_history
                    if isinstance(item, dict) and int(item.get("epoch", 0)) <= start_epoch
                ]
        if not history:
            history = [dict(checkpoint_metrics)]
        best_r2 = max(float(item["r2"]) for item in history)
        print(
            f"[train:{name}] resume={args.resume_checkpoint} epoch={start_epoch} "
            f"optimizer_state={'restored' if resumed_optimizer else 'restarted'}",
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
            features, targets = _load_shard(path)
            order = torch.randperm(features.shape[0])
            for start in range(0, features.shape[0], args.train_batch_size):
                index = order[start : start + args.train_batch_size]
                x = features[index].to(device)
                y = targets[index].to(device)
                x = ((x - feature_mean) / feature_std).clamp(-12, 12)
                y = (y - target_mean) / target_std
                prediction = model(x)
                mse = F.mse_loss(prediction, y)
                cosine = 1.0 - F.cosine_similarity(prediction, y, dim=-1, eps=1e-8).mean()
                loss = mse + args.cosine_weight * cosine
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * index.numel()
                total_count += index.numel()
        metrics = _evaluate(
            model,
            validation_paths,
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.train_batch_size,
        )
        metrics.update({"epoch": epoch, "train_loss": total_loss / total_count})
        history.append(metrics)
        print(
            f"[train:{name}] epoch={epoch}/{final_epoch} loss={metrics['train_loss']:.6f} "
            f"val_r2={metrics['r2']:.6f} val_cos={metrics['mean_cosine']:.6f}",
            flush=True,
        )
        payload = {
            "schema_version": "rank10_patch_oracle_probe_v1",
            "model_name": name,
            "model_class": type(model).__name__,
            "parameter_count": parameter_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "continued_from": (
                str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
            ),
            "feature_names": feature_names,
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "metrics": metrics,
        }
        torch.save(payload, args.output_dir / f"{name}_last.pt")
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            torch.save(payload, args.output_dir / f"{name}_best.pt")
    result = {"best_validation_r2": best_r2, "history": history}
    (args.output_dir / f"{name}_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def train(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"extract probe dataset first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rank10_patch_oracle_probe_dataset_v1":
        raise RuntimeError(
            "aggregate probe training requires "
            "rank10_patch_oracle_probe_dataset_v1"
        )
    if manifest.get("strict_patch_local") is not True or manifest.get("cross_patch_features") is not False:
        raise RuntimeError("refusing a probe dataset without the strict patch-local contract")
    train_paths = [args.output_dir / item["path"] for item in manifest["train"]["shards"]]
    validation_paths = [
        args.output_dir / item["path"] for item in manifest["validation"]["shards"]
    ]
    normalization = torch.load(
        args.output_dir / "normalization.pt", map_location="cpu", weights_only=False
    )
    results = {}
    if args.probe in {"linear", "both", "all"}:
        results["linear"] = _train_model(
            "linear",
            nn.Linear(len(PATCH_ORACLE_FEATURE_NAMES), 10),
            args=args,
            train_paths=train_paths,
            validation_paths=validation_paths,
            normalization=normalization,
        )
    if args.probe in {"mlp", "both", "all"}:
        results["mlp"] = _train_model(
            "mlp",
            PatchOracleToRankToken(),
            args=args,
            train_paths=train_paths,
            validation_paths=validation_paths,
            normalization=normalization,
        )
    if args.probe in {"deep", "all"}:
        results["deep_mlp"] = _train_model(
            "deep_mlp",
            DeepPatchOracleToRankToken(),
            args=args,
            train_paths=train_paths,
            validation_paths=validation_paths,
            normalization=normalization,
        )
    summary_path = args.output_dir / "probe_summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            existing.update(results)
            results = existing
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "train", "all"), default="all")
    parser.add_argument(
        "--extraction-contract",
        choices=(AGGREGATE_CONTRACT, POINTNET_CONTRACT),
        default=AGGREGATE_CONTRACT,
    )
    parser.add_argument("--rl-checkpoint", type=Path, default=DEFAULT_RL_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vit-attention-mode",
        choices=("joint_self", "cross_only"),
        required=True,
        help="Attention semantics used by the encoder in --rl-checkpoint.",
    )
    parser.add_argument("--max-files", type=int, default=1024)
    parser.add_argument("--max-contacts-per-file", type=int, default=0)
    parser.add_argument("--use-geometry-candidates", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-precontact-steps", type=int, default=4)
    parser.add_argument("--translation-range-m", type=float, default=0.20)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sdf-chunk-size", type=int, default=8192)
    parser.add_argument("--shard-patches", type=int, default=65536)
    parser.add_argument(
        "--probe",
        choices=("linear", "mlp", "deep", "both", "all"),
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="continue for --epochs additional epochs from a matching probe checkpoint",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_files < 0 or args.max_files == 1 or not 0 < args.validation_ratio < 1:
        parser.error("--max-files must be 0 (all) or >=2; --validation-ratio must be in (0,1)")
    if args.max_contacts_per_file < 0:
        parser.error("--max-contacts-per-file must be non-negative")
    if args.num_precontact_steps < 0 or args.translation_range_m < 0:
        parser.error("precontact steps/range must be non-negative")
    if args.extract_batch_size < 1 or args.train_batch_size < 1 or args.epochs < 1:
        parser.error("batch sizes and epochs must be positive")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
        if not args.resume_checkpoint.is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume_checkpoint}")
        if args.stage == "extract":
            parser.error("--resume-checkpoint requires --stage train or all")
        if args.probe not in {"linear", "mlp", "deep"}:
            parser.error("--resume-checkpoint requires one specific --probe")
    if args.stage in {"train", "all"} and args.extraction_contract != AGGREGATE_CONTRACT:
        parser.error(
            "pointnet_v2 is extraction-only; fit it with "
            "scripts/train_rank10_minimal_pointnet.py"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"extract", "all"}:
        if not args.rl_checkpoint.is_file():
            raise FileNotFoundError(args.rl_checkpoint)
        if not args.data_dir.is_dir():
            raise FileNotFoundError(args.data_dir)
        extract(args)
    if args.stage in {"train", "all"}:
        train(args)


if __name__ == "__main__":
    main()

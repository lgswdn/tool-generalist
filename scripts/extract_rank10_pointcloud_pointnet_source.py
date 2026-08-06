#!/usr/bin/env python3
"""Extract rank-10 targets with the exact RL point-cloud distance features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.dataset import NewPretrainDataset
from pretrain.train import collate_fn
from rsl_rl.modules.oracle_pointcloud_pointnet_encoder import (
    OraclePointCloudPointNetEncoder,
)
from scripts.train_rank10_minimal_pointnet import (
    FAST_POINT_FEATURE_NAMES,
    RunningMoments,
)
from scripts.train_rank10_patch_oracle_probe import (
    _grouped_file_split,
    _load_rl_encoder_and_bottleneck,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(files: list[str], max_contacts_per_file: int) -> NewPretrainDataset:
    return NewPretrainDataset(
        files,
        augment=False,
        require_movement=False,
        num_points=512,
        num_precontact_steps=0,
        allow_mock_physics=False,
        noise_max_trans=0.0,
        noise_max_rot_deg=0.0,
        noise_max_retries=1,
        floor_eps=0.0,
        validation_seed=12345,
        denoise_target_mode="one_step",
        tool_mesh_contract="adjusted_decomposed_mesh",
        include_meshes=False,
        max_contacts_per_file=max_contacts_per_file,
    )


def _extract_split(
    name: str,
    dataset: NewPretrainDataset,
    *,
    output_dir: Path,
    teacher: torch.nn.Module,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    oracle: OraclePointCloudPointNetEncoder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    shard_patches: int,
    input_moments: RunningMoments | None,
    target_moments: RunningMoments | None,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    buffered = 0
    total = 0
    shards: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffered, total
        features = torch.cat(feature_parts)
        targets = torch.cat(target_parts)
        path = split_dir / f"shard_{len(shards):06d}.pt"
        torch.save(
            {"point_features": features, "targets": targets},
            path,
        )
        count = int(targets.shape[0])
        shards.append(
            {"path": str(path.relative_to(output_dir)), "patches": count}
        )
        total += count
        buffered = 0
        feature_parts.clear()
        target_parts.clear()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            tool = batch["tool_points_E_k"].to(device, non_blocking=True)
            obj = batch["object_points_E_k"].to(device, non_blocking=True)
            relative = batch["rel_tool_object_t_k"].to(
                device, non_blocking=True
            )
            batch_count, steps, points, dims = tool.shape
            flat_tool = (tool + relative.unsqueeze(-2)).reshape(
                batch_count * steps, points, dims
            )
            flat_object = obj.reshape(batch_count * steps, points, dims)
            encoded = teacher.encode(flat_tool, flat_object)
            targets = F.linear(
                encoded.fused_tokens, down_weight, down_bias
            ).reshape(-1, 10)
            features = oracle.raw_point_features(
                flat_tool, flat_object
            ).reshape(-1, 32, len(FAST_POINT_FEATURE_NAMES))
            if features.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    "Point-cloud feature/teacher-token count mismatch: "
                    f"{features.shape[0]} != {targets.shape[0]}"
                )
            if not (
                bool(features.isfinite().all())
                and bool(targets.isfinite().all())
            ):
                raise RuntimeError("Point-cloud distillation tensors are non-finite")
            if input_moments is not None:
                input_moments.update(features)
            if target_moments is not None:
                target_moments.update(targets)
            feature_parts.append(features.half().cpu())
            target_parts.append(targets.float().cpu())
            buffered += int(targets.shape[0])
            if buffered >= shard_patches or batch_index == len(loader):
                flush()
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"[pointcloud-extract:{name}] "
                    f"samples={min(batch_index * batch_size, len(dataset))}/"
                    f"{len(dataset)} patches={total + buffered}",
                    flush=True,
                )
    return shards


def extract(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    device = torch.device(args.device)
    teacher, down_weight, down_bias, _ = _load_rl_encoder_and_bottleneck(
        args.rl_checkpoint,
        device=device,
        vit_attention_mode=args.vit_attention_mode,
    )
    oracle = OraclePointCloudPointNetEncoder(
        num_points=512,
        num_patches=16,
        patch_size=32,
        feature_dim=128,
        nearest_frame_batch_size=args.batch_size,
        feature_mode="fast11",
        use_rank10_bottleneck=True,
        token_mode="patches",
    ).to(device).eval()
    train_files, validation_files, split = _grouped_file_split(
        args.data_dir,
        max_files=args.max_files,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        use_geometry_candidates=True,
    )
    input_moments = RunningMoments(len(FAST_POINT_FEATURE_NAMES))
    target_moments = RunningMoments(10)
    train = _extract_split(
        "train",
        _dataset(train_files, args.max_contacts_per_file),
        output_dir=args.output_dir,
        teacher=teacher,
        down_weight=down_weight,
        down_bias=down_bias,
        oracle=oracle,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_patches=args.shard_patches,
        input_moments=input_moments,
        target_moments=target_moments,
    )
    validation = _extract_split(
        "validation",
        _dataset(validation_files, args.max_contacts_per_file),
        output_dir=args.output_dir,
        teacher=teacher,
        down_weight=down_weight,
        down_bias=down_bias,
        oracle=oracle,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_patches=args.shard_patches,
        input_moments=None,
        target_moments=None,
    )
    input_mean, input_std = input_moments.mean_std()
    target_mean, target_std = target_moments.mean_std()
    torch.save(
        {
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        args.output_dir / "normalization.pt",
    )
    manifest = {
        "schema_version": "rank10_fast_pointcloud11_dataset_v1",
        "source_rl_checkpoint": str(args.rl_checkpoint.resolve()),
        "source_rl_checkpoint_sha256": _sha256(args.rl_checkpoint),
        "data_dir": str(args.data_dir.resolve()),
        "point_feature_names": FAST_POINT_FEATURE_NAMES,
        "distance": "unsigned_nearest_opposite_pointcloud_point",
        "direction": "normalized_vector_to_nearest_opposite_pointcloud_point",
        "mesh_queries": False,
        "saved_contact_labels_used": False,
        "max_contacts_per_file": args.max_contacts_per_file,
        "split": split,
        "train": train,
        "validation": validation,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vit-attention-mode",
        choices=("joint_self", "cross_only"),
        required=True,
    )
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--max-contacts-per-file", type=int, default=56)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-patches", type=int, default=65536)
    args = parser.parse_args()
    if not args.rl_checkpoint.is_file():
        parser.error(f"RL checkpoint does not exist: {args.rl_checkpoint}")
    if not args.data_dir.is_dir():
        parser.error(f"Contact directory does not exist: {args.data_dir}")
    if args.max_files < 2 or args.max_contacts_per_file < 1:
        parser.error("max files must be >=2 and contacts per file must be >=1")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation ratio must be in (0, 1)")
    if min(args.batch_size, args.shard_patches) < 1:
        parser.error("batch and shard sizes must be positive")
    return args


if __name__ == "__main__":
    extract(parse_args())

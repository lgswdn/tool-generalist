#!/usr/bin/env python3
"""Standalone XYZ-only patch-distance pretraining from geometry manifests.

This intentionally does not read contact ``.pt`` files and does not participate
in the experiment artifact resolver.  It samples object and generated-gripper
surface point clouds directly from their OBJ assets and writes one explicit
checkpoint for the RL experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretrain.patch_distance_pointnet_model import PatchDistancePointNetPretrainModel
from utils.geometry.mesh_io import load_scaled_sampled_surface_points


DEFAULT_OBJECT_MANIFEST = Path(
    "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"
)
DEFAULT_OBJECT_ROOT = Path(
    "/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized"
)
DEFAULT_TOOL_MANIFEST = REPO_ROOT / "configs/generated_gripper_contact_assets/tools_selected.json"
DEFAULT_TOOL_ROOT = REPO_ROOT / "configs/generated_gripper_contact_assets/meshdata_adjusted"
DEFAULT_OUTPUT = REPO_ROOT / ".pretrained_checkpoints/patch_distance_pointnet/best.pt"


@dataclass(frozen=True)
class GeometryAsset:
    name: str
    mesh_path: Path
    listed_scale: float | None = None


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _object_assets(manifest: Path, mesh_root: Path) -> list[GeometryAsset]:
    entries = _load_json(manifest)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"object manifest must be a non-empty list: {manifest}")
    assets: list[GeometryAsset] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            name = entry
            try:
                mesh_name, scale_text = name.rsplit("-", 1)
                listed_scale = float(scale_text)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"invalid '<object>-<scale>' entry at {manifest}[{index}]: {entry!r}"
                ) from exc
        elif isinstance(entry, dict):
            name = str(entry.get("name", entry.get("object", entry.get("object_id", ""))))
            listed_scale = float(entry["scale"]) if entry.get("scale") is not None else None
            mesh_name = name.rsplit("-", 1)[0] if listed_scale is None else name
        else:
            raise ValueError(f"invalid object entry at {manifest}[{index}]")
        mesh_path = mesh_root / f"{mesh_name}.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"object mesh is missing: {mesh_path}")
        assets.append(GeometryAsset(name=name, mesh_path=mesh_path, listed_scale=listed_scale))
    return assets


def _tool_assets(manifest: Path, mesh_root: Path) -> list[GeometryAsset]:
    entries = _load_json(manifest)
    if isinstance(entries, dict):
        entries = entries.get("tools", entries.get("selected"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"tool manifest must be a non-empty list: {manifest}")
    assets: list[GeometryAsset] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("name", entry.get("tool_id", entry.get("id", ""))))
        else:
            raise ValueError(f"invalid tool entry at {manifest}[{index}]")
        if not name:
            raise ValueError(f"tool entry lacks an id at {manifest}[{index}]")
        mesh_path = mesh_root / name / "coacd" / "decomposed.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"generated-gripper mesh is missing: {mesh_path}")
        assets.append(GeometryAsset(name=name, mesh_path=mesh_path))
    return assets


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q.astype(np.float32)


class GeometryPointCloudPairDataset(Dataset):
    """Random independent object/gripper clouds; no pose or contact data."""

    def __init__(
        self,
        *,
        objects: list[GeometryAsset],
        tools: list[GeometryAsset],
        length: int,
        num_points: int,
        cached_surface_points: int,
        object_scale_range: tuple[float, float],
        use_listed_object_scales: bool,
        jitter_std_m: float,
        seed: int,
        training: bool,
    ) -> None:
        if not objects or not tools:
            raise ValueError("geometry dataset requires both objects and grippers")
        if cached_surface_points < num_points:
            raise ValueError("cached_surface_points must be >= num_points")
        self.objects = objects
        self.tools = tools
        self.length = int(length)
        self.num_points = int(num_points)
        self.cached_surface_points = int(cached_surface_points)
        self.object_scale_range = tuple(float(v) for v in object_scale_range)
        self.use_listed_object_scales = bool(use_listed_object_scales)
        self.jitter_std_m = float(jitter_std_m)
        self.seed = int(seed)
        self.training = bool(training)
        self._surface_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> np.random.Generator:
        if not self.training:
            return np.random.default_rng(self.seed + index)
        worker_seed = int(torch.initial_seed() % (2**32))
        return np.random.default_rng(worker_seed + index)

    def _base_surface(self, asset: GeometryAsset) -> np.ndarray:
        cached = self._surface_cache.get(asset.mesh_path)
        if cached is None:
            stable_seed = sum(asset.name.encode("utf-8")) + self.seed
            points = load_scaled_sampled_surface_points(
                asset.mesh_path,
                scale=1.0,
                num_points=self.cached_surface_points,
                seed=stable_seed,
                process=False,
            ).astype(np.float32)
            bbox_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
            cached = np.ascontiguousarray(points - bbox_center)
            self._surface_cache[asset.mesh_path] = cached
        return cached

    def _cloud(
        self,
        asset: GeometryAsset,
        rng: np.random.Generator,
        *,
        object_cloud: bool,
    ) -> torch.Tensor:
        base = self._base_surface(asset)
        selected = rng.choice(base.shape[0], size=self.num_points, replace=False)
        cloud = base[selected].copy()
        if object_cloud:
            if self.use_listed_object_scales:
                if asset.listed_scale is None:
                    raise ValueError(f"object lacks a listed scale: {asset.name}")
                scale = asset.listed_scale
            else:
                scale = rng.uniform(*self.object_scale_range)
            cloud *= float(scale)
        cloud = cloud @ _random_rotation(rng).T
        if self.jitter_std_m > 0.0:
            cloud += rng.normal(0.0, self.jitter_std_m, size=cloud.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(cloud, dtype=np.float32))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        obj = self.objects[int(rng.integers(len(self.objects)))]
        tool = self.tools[int(rng.integers(len(self.tools)))]
        return (
            self._cloud(tool, rng, object_cloud=False),
            self._cloud(obj, rng, object_cloud=True),
        )


def _split_assets(
    assets: list[GeometryAsset], val_ratio: float, seed: int
) -> tuple[list[GeometryAsset], list[GeometryAsset]]:
    shuffled = list(assets)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_count:] or shuffled, shuffled[:val_count]


def _average_metrics(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for tool_points, object_points in loader:
            tool_points = tool_points.to(device, non_blocking=True)
            object_points = object_points.to(device, non_blocking=True)
            loss, metrics = model(tool_points, object_points)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            batches += 1
    if dist.is_initialized():
        keys = sorted(sums)
        reduced = torch.tensor(
            [*(sums[key] for key in keys), float(batches)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        sums = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        batches = int(reduced[-1].item())
    return _average_metrics(sums, batches)


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> dict:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    dims = {
        "num_pts": args.num_points,
        "num_patches": args.num_patches,
        "patch_size": args.patch_size,
        "encoder_channel": args.encoder_channel,
        "feature_dim": args.encoder_channel,
    }
    return {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "schema_version": "pretrain_checkpoint_v1",
            "model": {"family": "patch_distance_pointnet", "dims": dims},
            "model_dims": dims,
            "objective": "min_{p in 32-point patch} ||query-p||_2",
            "uses_contact_pt": False,
            "uses_mesh_distance": False,
            "config": vars(args),
        },
    }


def train(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed patch-distance pretraining requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    objects = _object_assets(args.object_manifest, args.object_mesh_root)
    tools = _tool_assets(args.tool_manifest, args.tool_mesh_root)
    train_objects, val_objects = _split_assets(objects, args.val_ratio, args.seed)
    train_tools, val_tools = _split_assets(tools, args.val_ratio, args.seed + 1)
    dataset_kwargs = dict(
        num_points=args.num_points,
        cached_surface_points=args.cached_surface_points,
        object_scale_range=(args.object_scale_min, args.object_scale_max),
        use_listed_object_scales=args.use_listed_object_scales,
        jitter_std_m=args.jitter_std_m,
    )
    train_dataset = GeometryPointCloudPairDataset(
        objects=train_objects,
        tools=train_tools,
        length=args.samples_per_epoch,
        seed=args.seed,
        training=True,
        **dataset_kwargs,
    )
    val_dataset = GeometryPointCloudPairDataset(
        objects=val_objects,
        tools=val_tools,
        length=args.val_samples,
        seed=args.seed + 100_000,
        training=False,
        **dataset_kwargs,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = PatchDistancePointNetPretrainModel(
        num_points=args.num_points,
        num_patches=args.num_patches,
        patch_size=args.patch_size,
        encoder_channel=args.encoder_channel,
        point_scale_m=args.point_scale_m,
        query_count=args.query_count,
        supervised_patches_per_cloud=args.supervised_patches_per_cloud,
        query_min_offset_m=args.query_min_offset_m,
        query_max_offset_m=args.query_max_offset_m,
        distance_scale_m=args.distance_scale_m,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_learning_rate
    )
    start_epoch = 0
    best_val = math.inf
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        resume_model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("best_val", math.inf))

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"device={device} world_size={world_size} "
            f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * world_size} "
            f"objects={len(objects)} grippers={len(tools)} "
            f"contact_pt=False mesh_distance=False"
        )
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, None)
        val_loss = val_metrics["patch_distance_loss"]
        if rank == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_metrics['patch_distance_loss']:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae_m={val_metrics['patch_distance_mae_m']:.6g} "
                f"val_near_mae_m={val_metrics['patch_distance_near_mae_m']:.6g}"
            )
        if val_loss < best_val:
            best_val = val_loss
            if rank == 0:
                torch.save(
                    _checkpoint(
                        model, optimizer, epoch=epoch + 1, best_val=best_val, args=args
                    ),
                    args.output,
                )
        scheduler.step()
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-manifest", type=Path, default=DEFAULT_OBJECT_MANIFEST)
    parser.add_argument("--object-mesh-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--tool-manifest", type=Path, default=DEFAULT_TOOL_MANIFEST)
    parser.add_argument("--tool-mesh-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs; the best validation checkpoint is retained.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=20000,
        help="Global samples per epoch, divided across distributed ranks.",
    )
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU/process."
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--cached-surface-points", type=int, default=2048)
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--encoder-channel", type=int, default=128)
    parser.add_argument("--point-scale-m", type=float, default=0.05)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--supervised-patches-per-cloud", type=int, default=8)
    parser.add_argument("--query-min-offset-m", type=float, default=0.0005)
    parser.add_argument("--query-max-offset-m", type=float, default=0.03)
    parser.add_argument("--distance-scale-m", type=float, default=0.03)
    parser.add_argument("--object-scale-min", type=float, default=0.1)
    parser.add_argument("--object-scale-max", type=float, default=0.3)
    parser.add_argument("--use-listed-object-scales", action="store_true")
    parser.add_argument("--jitter-std-m", type=float, default=0.0005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

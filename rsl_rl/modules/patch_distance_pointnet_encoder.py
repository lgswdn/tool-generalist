"""Patchwise PointNet whose only input is local XYZ geometry."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class PatchDistancePointNetEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class PatchDistancePointNetEncoder(nn.Module):
    """Shared PointNet over FPS/KNN patches, with no privileged inputs.

    Each point is represented only by ``(point - patch_center) / point_scale_m``.
    Patch position and body identity are deliberately absent here; the RL policy
    adds those after loading and freezing this geometry-pretrained encoder.
    """

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        point_scale_m: float = 0.05,
    ) -> None:
        super().__init__()
        if min(num_points, num_patches, patch_size, feature_dim) <= 0:
            raise ValueError("patch-distance PointNet dimensions must be positive")
        if point_scale_m <= 0.0:
            raise ValueError("point_scale_m must be > 0")
        self.num_points = int(num_points)
        self._num_patches = int(num_patches)
        self.patch_size = int(patch_size)
        self._feature_dim = int(feature_dim)
        self.point_scale_m = float(point_scale_m)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.patch_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @staticmethod
    def _fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        count = min(int(count), num_points)
        centers = torch.zeros(
            batch_size, count, dtype=torch.long, device=points.device
        )
        distance = torch.full(
            (batch_size, num_points),
            float("inf"),
            dtype=points.dtype,
            device=points.device,
        )
        farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
        batch = torch.arange(batch_size, device=points.device)
        for index in range(count):
            centers[:, index] = farthest
            center = points[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, (points - center).square().sum(dim=-1))
            farthest = distance.max(dim=1).indices
        return centers

    def _patchify(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-2:] != (self.num_points, 3):
            raise ValueError(
                f"expected point clouds shaped (B, {self.num_points}, 3), "
                f"got {tuple(points.shape)}"
            )
        batch_size, num_points, _ = points.shape
        center_idx = self._fps_indices(points, self.num_patches)
        batch = torch.arange(batch_size, device=points.device).view(batch_size, 1)
        centers = points[batch, center_idx]
        k = min(self.patch_size, num_points)
        patch_idx = torch.cdist(centers, points).topk(
            k=k, dim=-1, largest=False
        ).indices
        if k < self.patch_size:
            patch_idx = torch.cat(
                (
                    patch_idx,
                    patch_idx[..., -1:].expand(
                        batch_size,
                        patch_idx.shape[1],
                        self.patch_size - k,
                    ),
                ),
                dim=-1,
            )
        batch = torch.arange(batch_size, device=points.device).view(-1, 1, 1)
        patches = points[batch, patch_idx]
        return patches, patch_idx, centers

    def encode_cloud(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, patch_idx, centers = self._patchify(points)
        local_xyz = (patches - centers.unsqueeze(2)) / self.point_scale_m
        point_features = self.point_mlp(local_xyz)
        pooled = torch.cat(
            (point_features.max(dim=2).values, point_features.mean(dim=2)),
            dim=-1,
        )
        return self.patch_mlp(pooled), patch_idx, centers

    def encode(
        self, tool_pc: torch.Tensor, obj_pc: torch.Tensor
    ) -> PatchDistancePointNetEncodeResult:
        tool_tokens, tool_idx, tool_centers = self.encode_cloud(tool_pc)
        obj_tokens, obj_idx, obj_centers = self.encode_cloud(obj_pc)
        return PatchDistancePointNetEncodeResult(
            fused_tokens=torch.cat((tool_tokens, obj_tokens), dim=1),
            tool_patch_idx=tool_idx,
            obj_patch_idx=obj_idx,
            tool_patch_centers=tool_centers,
            obj_patch_centers=obj_centers,
        )

    def forward(
        self, tool_pc: torch.Tensor, obj_pc: torch.Tensor
    ) -> PatchDistancePointNetEncodeResult:
        return self.encode(tool_pc, obj_pc)

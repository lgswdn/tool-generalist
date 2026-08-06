"""Patchwise PointNet over privileged unsigned point-to-mesh distances."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

try:
    from pytorch3d.ops import knn_points, sample_farthest_points
except ImportError:  # Lightweight CPU/test environments.
    knn_points = None
    sample_farthest_points = None


class OraclePointMeshEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class OraclePointMeshPointNetEncoder(nn.Module):
    """Encode each FPS/KNN patch independently from per-point ``(xyz, d)``.

    ``d`` is exact unsigned distance to the opposite triangle mesh. There is no
    cross-patch PointNet, transformer, global token, or binary-contact input.
    """

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        coordinate_scale_m: float = 0.30,
        distance_scale_m: float = 0.10,
        normalization_clip: float = 5.0,
    ) -> None:
        super().__init__()
        if min(num_points, num_patches, patch_size, feature_dim) <= 0:
            raise ValueError("oracle pointmesh PointNet dimensions must be positive")
        if min(coordinate_scale_m, distance_scale_m, normalization_clip) <= 0:
            raise ValueError("oracle pointmesh normalization scales must be positive")
        self.num_points = int(num_points)
        self._P = int(num_patches)
        self.patch_size = int(patch_size)
        self._D = int(feature_dim)
        self.coordinate_scale_m = float(coordinate_scale_m)
        self.distance_scale_m = float(distance_scale_m)
        self.normalization_clip = float(normalization_clip)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.patch_projection = nn.Sequential(
            nn.Linear(256, self._D),
            nn.LayerNorm(self._D),
        )

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def num_patches(self) -> int:
        return self._P

    @staticmethod
    def _fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        count = min(int(count), num_points)
        if sample_farthest_points is not None and points.is_cuda:
            _, indices = sample_farthest_points(
                points.contiguous(), K=count, random_start_point=False
            )
            return indices
        centroids = torch.zeros(batch_size, count, dtype=torch.long, device=points.device)
        distance = torch.full(
            (batch_size, num_points), float("inf"), dtype=points.dtype, device=points.device
        )
        farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
        batch = torch.arange(batch_size, device=points.device)
        for index in range(count):
            centroids[:, index] = farthest
            center = points[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, ((points - center) ** 2).sum(dim=-1))
            farthest = distance.max(dim=1).indices
        return centroids

    def _patch_indices_and_centers(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = points.shape
        center_idx = self._fps_indices(points, self._P)
        batch = torch.arange(batch_size, device=points.device).view(batch_size, 1)
        centers = points[batch, center_idx]
        k = min(self.patch_size, num_points)
        if knn_points is not None and points.is_cuda:
            _, indices, _ = knn_points(
                centers.contiguous(),
                points.contiguous(),
                K=k,
                return_nn=False,
                return_sorted=True,
            )
        else:
            indices = torch.cdist(centers, points).topk(
                k=k, dim=-1, largest=False
            ).indices
        if k < self.patch_size:
            indices = torch.cat(
                (
                    indices,
                    indices[..., -1:].expand(
                        batch_size, indices.shape[1], self.patch_size - k
                    ),
                ),
                dim=-1,
            )
        if indices.shape[1] < self._P:
            missing = self._P - indices.shape[1]
            indices = torch.cat(
                (indices, indices[:, -1:].expand(batch_size, missing, self.patch_size)),
                dim=1,
            )
            centers = torch.cat(
                (centers, centers[:, -1:].expand(batch_size, missing, 3)), dim=1
            )
        return indices, centers

    def _encode_cloud(
        self,
        points: torch.Tensor,
        unsigned_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_points, xyz_dim = points.shape
        if xyz_dim != 3 or num_points != self.num_points:
            raise RuntimeError(
                f"oracle pointmesh cloud must have shape (B, {self.num_points}, 3), "
                f"got {tuple(points.shape)}"
            )
        if unsigned_distance.shape != (batch_size, num_points):
            raise RuntimeError(
                "oracle unsigned mesh distance shape mismatch: expected "
                f"{(batch_size, num_points)}, got {tuple(unsigned_distance.shape)}"
            )
        if not bool(torch.isfinite(unsigned_distance).all()) or bool((unsigned_distance < 0).any()):
            raise RuntimeError("oracle pointmesh distances must be finite and non-negative")
        patch_idx, centers = self._patch_indices_and_centers(points)
        batch = torch.arange(batch_size, device=points.device).view(batch_size, 1, 1)
        patch_xyz = points[batch, patch_idx]
        patch_distance = unsigned_distance.gather(
            1, patch_idx.reshape(batch_size, -1)
        ).reshape(batch_size, self._P, self.patch_size, 1)
        xyz_feature = torch.clamp(
            patch_xyz / self.coordinate_scale_m,
            -self.normalization_clip,
            self.normalization_clip,
        )
        distance_feature = torch.clamp(
            patch_distance / self.distance_scale_m,
            0.0,
            self.normalization_clip,
        )
        point_feature = self.point_mlp(torch.cat((xyz_feature, distance_feature), dim=-1))
        pooled = torch.cat(
            (point_feature.max(dim=2).values, point_feature.mean(dim=2)), dim=-1
        )
        return self.patch_projection(pooled), patch_idx, centers

    def encode(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        *,
        tool_unsigned_distance: torch.Tensor | None = None,
        obj_unsigned_distance: torch.Tensor | None = None,
    ) -> OraclePointMeshEncodeResult:
        if tool_unsigned_distance is None or obj_unsigned_distance is None:
            raise RuntimeError(
                "oracle pointmesh PointNet requires exact per-point unsigned mesh distance"
            )
        tool_tokens, tool_idx, tool_centers = self._encode_cloud(
            tool_pc, tool_unsigned_distance
        )
        obj_tokens, obj_idx, obj_centers = self._encode_cloud(obj_pc, obj_unsigned_distance)
        return OraclePointMeshEncodeResult(
            fused_tokens=torch.cat((tool_tokens, obj_tokens), dim=1),
            tool_patch_idx=tool_idx,
            obj_patch_idx=obj_idx,
            tool_patch_centers=tool_centers,
            obj_patch_centers=obj_centers,
        )

    def forward(self, tool_pc: torch.Tensor, obj_pc: torch.Tensor, **kwargs):
        return self.encode(tool_pc, obj_pc, **kwargs)

"""Exact privileged mesh-SDF patch representation for controlled RL ablations.

The environment supplies real per-point signed distances to the opposite mesh.
No point-cloud-distance fallback, learned point backbone, or patch transformer
is used here.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

try:
    from pytorch3d.ops import knn_points, sample_farthest_points
except ImportError:  # Lightweight CPU/test environments.
    knn_points = None
    sample_farthest_points = None


class OraclePatchEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class _OraclePatchBase(nn.Module):
    """Patch construction shared by the exact privileged mesh-SDF oracle."""

    raw_feature_dim = 8

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        contact_eps: float = 0.002,
        center_scale_m: float = 0.30,
        distance_scale_m: float = 0.10,
        normalization_clip: float = 5.0,
    ) -> None:
        super().__init__()
        if min(num_points, num_patches, patch_size, feature_dim) <= 0:
            raise ValueError("oracle patch dimensions must be positive")
        if contact_eps < 0 or center_scale_m <= 0 or distance_scale_m <= 0:
            raise ValueError("oracle metric scales must be positive (contact_eps may be zero)")
        if normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        self.num_points = int(num_points)
        self._P = int(num_patches)
        self.patch_size = int(patch_size)
        self._D = int(feature_dim)
        self.contact_eps = float(contact_eps)
        self.center_scale_m = float(center_scale_m)
        self.distance_scale_m = float(distance_scale_m)
        self.normalization_clip = float(normalization_clip)
        self.embedding = nn.Sequential(
            nn.Linear(self.raw_feature_dim, self._D),
            nn.LayerNorm(self._D),
            nn.GELU(),
            nn.Linear(self._D, self._D),
            nn.LayerNorm(self._D),
        )

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def num_patches(self) -> int:
        return self._P

    @staticmethod
    def _fps_indices(pc: torch.Tensor, num_centers: int) -> torch.Tensor:
        batch_size, num_points, _ = pc.shape
        count = min(int(num_centers), num_points)
        if sample_farthest_points is not None and pc.is_cuda:
            _, indices = sample_farthest_points(
                pc.contiguous(),
                K=count,
                random_start_point=False,
            )
            return indices
        centroids = torch.zeros(batch_size, count, dtype=torch.long, device=pc.device)
        distance = torch.full((batch_size, num_points), float("inf"), dtype=pc.dtype, device=pc.device)
        farthest = torch.zeros(batch_size, dtype=torch.long, device=pc.device)
        batch = torch.arange(batch_size, device=pc.device)
        for index in range(count):
            centroids[:, index] = farthest
            center = pc[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, ((pc - center) ** 2).sum(dim=-1))
            farthest = distance.max(dim=1).indices
        return centroids

    def _patches(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = pc.shape
        center_idx = self._fps_indices(pc, self._P)
        batch = torch.arange(batch_size, device=pc.device).view(batch_size, 1)
        centers = pc[batch, center_idx]
        k = min(self.patch_size, num_points)
        if knn_points is not None and pc.is_cuda:
            _, idx, _ = knn_points(
                centers.contiguous(),
                pc.contiguous(),
                K=k,
                return_nn=False,
                return_sorted=True,
            )
        else:
            idx = torch.cdist(centers, pc).topk(k=k, dim=-1, largest=False).indices
        if k < self.patch_size:
            idx = torch.cat(
                (idx, idx[..., -1:].expand(batch_size, idx.shape[1], self.patch_size - k)),
                dim=-1,
            )
        if idx.shape[1] < self._P:
            missing = self._P - idx.shape[1]
            idx = torch.cat((idx, idx[:, -1:].expand(batch_size, missing, self.patch_size)), dim=1)
            centers = torch.cat((centers, centers[:, -1:].expand(batch_size, missing, 3)), dim=1)
        batch3 = torch.arange(batch_size, device=pc.device).view(batch_size, 1, 1)
        return pc[batch3, idx], idx, centers

class OraclePatchDistanceEncoder(_OraclePatchBase):
    """Transparent patch summary with minimum-SDF location and log SDF.

    The default explicit 12D descriptor is:
    ``center_xyz, min_sdf, mean_sdf, contact, body_one_hot,
    argmin_sdf_relative_xyz, signed_log_min_sdf``.
    Setting ``include_contact_feature=False`` removes the contact coordinate,
    producing an 11D descriptor whose contact target cannot enter the input.
    There is no pointwise learned encoder and no joint patch transformer.
    """

    raw_feature_dim = 12

    def __init__(
        self,
        *,
        include_contact_feature: bool = True,
        patch_relative_scale_m: float = 0.05,
        log_distance_resolution_m: float = 0.005,
        log_distance_cap_m: float = 0.05,
        **kwargs,
    ) -> None:
        self.include_contact_feature = bool(include_contact_feature)
        self.raw_feature_dim = 12 if self.include_contact_feature else 11
        super().__init__(**kwargs)
        if min(
            patch_relative_scale_m,
            log_distance_resolution_m,
            log_distance_cap_m,
        ) <= 0:
            raise ValueError("oracle argmax/log metric scales must be positive")
        self.patch_relative_scale_m = float(patch_relative_scale_m)
        self.log_distance_resolution_m = float(log_distance_resolution_m)
        self.log_distance_cap_m = float(log_distance_cap_m)
        # Replace the parent's 8D embedding with the explicit descriptor.
        self.embedding = nn.Sequential(
            nn.Linear(self.raw_feature_dim, self._D),
            nn.LayerNorm(self._D),
            nn.GELU(),
            nn.Linear(self._D, self._D),
            nn.LayerNorm(self._D),
        )

    def _argmax_raw_features(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        patch_signed_sdf: torch.Tensor,
        *,
        type_id: int,
    ) -> torch.Tensor:
        batch_size, patch_count, patch_size, _ = patches.shape
        if patch_signed_sdf.shape != (batch_size, patch_count, patch_size):
            raise RuntimeError(
                "exact patch mesh SDF shape mismatch: expected "
                f"{(batch_size, patch_count, patch_size)}, got {tuple(patch_signed_sdf.shape)}"
            )
        if not bool(torch.isfinite(patch_signed_sdf).all()):
            raise RuntimeError("exact patch mesh SDF contains non-finite values")
        signed = patch_signed_sdf
        signed_closest = signed.min(dim=-1).values
        signed_mean = signed.mean(dim=-1)

        # Explicit argmax of contact score (-SDF), i.e. argmin SDF. A relative coordinate
        # is used instead of the arbitrary KNN-array index.
        closest_idx = (-signed).argmax(dim=-1, keepdim=True)
        relative_xyz = patches - centers.unsqueeze(2)
        closest_xyz = relative_xyz.gather(
            2,
            closest_idx.unsqueeze(-1).expand(batch_size, patch_count, 1, 3),
        ).squeeze(2)
        closest_xyz = torch.clamp(
            closest_xyz / self.patch_relative_scale_m,
            -self.normalization_clip,
            self.normalization_clip,
        )

        center_feature = torch.clamp(
            centers / self.center_scale_m,
            -self.normalization_clip,
            self.normalization_clip,
        )
        # Preserve the real signed-SDF global approach signal. With the default
        # 0.10 m scale, 5 cm -> 0.5 and 10 cm -> 1.0; clipping happens only
        # at the broad normalization_clip limit (50 cm by default).
        distance_feature = torch.clamp(
            torch.stack((signed_closest, signed_mean), dim=-1) / self.distance_scale_m,
            -self.normalization_clip,
            self.normalization_clip,
        )
        # Signed log1p of the real minimum SDF emphasizes the 0-vs-5 mm regime. Magnitudes beyond
        # 5 cm are deliberately identical, so 10-vs-11 cm cannot matter.
        log_denom = torch.log1p(
            signed_closest.new_tensor(self.log_distance_cap_m / self.log_distance_resolution_m)
        )
        log_magnitude = signed_closest.abs().clamp_max(self.log_distance_cap_m)
        signed_log_min = (
            signed_closest.sign()
            * torch.log1p(log_magnitude / self.log_distance_resolution_m)
            / log_denom
        ).clamp(-1.0, 1.0)
        body_type = centers.new_zeros(batch_size, patch_count, 2)
        body_type[..., int(type_id)] = 1.0
        features = [center_feature, distance_feature]
        if self.include_contact_feature:
            contact = (signed <= self.contact_eps).any(dim=-1, keepdim=True).to(centers.dtype)
            features.append(contact)
        features.extend((body_type, closest_xyz, signed_log_min.unsqueeze(-1)))
        return torch.cat(features, dim=-1)

    @staticmethod
    def _gather_patch_sdf(point_sdf: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, patch_size = patch_idx.shape
        if point_sdf.ndim != 2 or point_sdf.shape[0] != batch_size:
            raise RuntimeError(
                "exact point mesh SDF must have shape (B, N), got "
                f"{tuple(point_sdf.shape)}"
            )
        return point_sdf.gather(1, patch_idx.reshape(batch_size, -1)).reshape(
            batch_size, patch_count, patch_size
        )

    def encode(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        *,
        tool_signed_sdf: torch.Tensor | None = None,
        obj_signed_sdf: torch.Tensor | None = None,
    ) -> OraclePatchEncodeResult:
        if tool_signed_sdf is None or obj_signed_sdf is None:
            raise RuntimeError(
                "oracle_patch requires real per-point signed mesh SDF values; "
                "point-cloud distance fallback is forbidden"
            )
        tool_patch, tool_idx, tool_centers = self._patches(tool_pc)
        obj_patch, obj_idx, obj_centers = self._patches(obj_pc)
        tool_patch_sdf = self._gather_patch_sdf(tool_signed_sdf, tool_idx)
        obj_patch_sdf = self._gather_patch_sdf(obj_signed_sdf, obj_idx)
        tool_raw = self._argmax_raw_features(tool_patch, tool_centers, tool_patch_sdf, type_id=0)
        obj_raw = self._argmax_raw_features(obj_patch, obj_centers, obj_patch_sdf, type_id=1)
        return OraclePatchEncodeResult(
            self.embedding(torch.cat((tool_raw, obj_raw), dim=1)),
            tool_idx,
            obj_idx,
            tool_centers,
            obj_centers,
        )

    def forward(self, tool_pc: torch.Tensor, obj_pc: torch.Tensor, **kwargs) -> OraclePatchEncodeResult:
        return self.encode(tool_pc, obj_pc, **kwargs)

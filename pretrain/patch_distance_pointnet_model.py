"""Point-set distance-field pretraining for an XYZ-only patch PointNet."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/patch_distance_pointnet_encoder.py"
    spec = importlib.util.spec_from_file_location("patch_distance_pointnet_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch-distance PointNet encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PatchDistancePointNetEncoder


class PatchDistancePointNetPretrainModel(nn.Module):
    """Make each patch token decode distance to its own 32 input points.

    The target for query ``q`` and patch point set ``P`` is exactly
    ``min(p in P) ||q - p||_2``.  No mesh or opposite-body information is used.
    """

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        encoder_channel: int = 128,
        point_scale_m: float = 0.05,
        query_count: int = 24,
        supervised_patches_per_cloud: int = 8,
        query_min_offset_m: float = 0.0005,
        query_max_offset_m: float = 0.03,
        distance_scale_m: float = 0.03,
    ) -> None:
        super().__init__()
        if not 1 <= supervised_patches_per_cloud <= num_patches:
            raise ValueError("supervised_patches_per_cloud must be in [1, num_patches]")
        if query_count <= 0:
            raise ValueError("query_count must be > 0")
        if not 0.0 < query_min_offset_m < query_max_offset_m:
            raise ValueError("query offsets must satisfy 0 < min < max")
        if distance_scale_m <= 0.0:
            raise ValueError("distance_scale_m must be > 0")
        self.model_family = "patch_distance_pointnet"
        self.num_patches = int(num_patches)
        self.query_count = int(query_count)
        self.supervised_patches_per_cloud = int(supervised_patches_per_cloud)
        self.query_min_offset_m = float(query_min_offset_m)
        self.query_max_offset_m = float(query_max_offset_m)
        self.distance_scale_m = float(distance_scale_m)
        self.encoder = _encoder_class()(
            num_points=num_points,
            num_patches=num_patches,
            patch_size=patch_size,
            feature_dim=encoder_channel,
            point_scale_m=point_scale_m,
        )
        self.query_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.distance_decoder = nn.Sequential(
            nn.Linear(encoder_channel + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Fixed validation offsets cover logarithmically spaced metric scales
        # and approximately uniform directions without consuming RNG state.
        index = torch.arange(query_count, dtype=torch.float32)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        z = 1.0 - 2.0 * (index + 0.5) / query_count
        radius_xy = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        direction = torch.stack(
            (radius_xy * torch.cos(index * golden_angle),
             radius_xy * torch.sin(index * golden_angle), z),
            dim=-1,
        )
        radius = torch.exp(
            torch.linspace(
                math.log(query_min_offset_m),
                math.log(query_max_offset_m),
                query_count,
            )
        )
        self.register_buffer(
            "validation_offsets", direction * radius.unsqueeze(-1), persistent=False
        )

    @staticmethod
    def _gather_patches(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
        return points[batch, indices]

    def _selected_patch_ids(self, device: torch.device) -> torch.Tensor:
        count = self.supervised_patches_per_cloud
        if self.training:
            return torch.randperm(self.num_patches, device=device)[:count]
        return torch.linspace(
            0, self.num_patches - 1, count, device=device
        ).round().to(dtype=torch.long)

    def _sample_queries(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, patch_count, patch_size, _ = patches.shape
        query_count = self.query_count
        if self.training:
            anchor_idx = torch.randint(
                patch_size,
                (batch_size, patch_count, query_count),
                device=patches.device,
            )
            direction = torch.randn(
                batch_size,
                patch_count,
                query_count,
                3,
                device=patches.device,
                dtype=patches.dtype,
            )
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            log_min = math.log(self.query_min_offset_m)
            log_max = math.log(self.query_max_offset_m)
            radius = torch.exp(
                torch.empty(
                    batch_size,
                    patch_count,
                    query_count,
                    1,
                    device=patches.device,
                    dtype=patches.dtype,
                ).uniform_(log_min, log_max)
            )
            offsets = direction * radius
        else:
            anchor_idx = (
                torch.arange(query_count, device=patches.device) % patch_size
            ).view(1, 1, query_count).expand(batch_size, patch_count, -1)
            offsets = self.validation_offsets.to(
                device=patches.device, dtype=patches.dtype
            ).view(1, 1, query_count, 3).expand(batch_size, patch_count, -1, -1)
        anchor_idx_expanded = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        anchors = torch.gather(patches, dim=2, index=anchor_idx_expanded)
        queries = anchors + offsets
        query_relative = (queries - centers.unsqueeze(2)) / self.encoder.point_scale_m
        return queries, query_relative

    def _decode_loss(
        self,
        tokens: torch.Tensor,
        query_relative: torch.Tensor,
        point_patch_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_features = self.query_embed(query_relative)
        expanded_tokens = tokens.unsqueeze(2).expand(
            -1, -1, query_relative.shape[2], -1
        )
        raw_prediction = self.distance_decoder(
            torch.cat((expanded_tokens, query_features), dim=-1)
        ).squeeze(-1)
        prediction = torch.sigmoid(raw_prediction)
        target = (point_patch_distance / self.distance_scale_m).clamp(0.0, 1.0)
        near_weight = 1.0 + 2.0 * torch.exp(-point_patch_distance / 0.005)
        loss = (
            F.smooth_l1_loss(prediction, target, reduction="none", beta=0.05)
            * near_weight
        ).mean()
        metric_prediction_m = prediction * self.distance_scale_m
        near = point_patch_distance <= 0.005
        near_error = (metric_prediction_m - point_patch_distance).abs()[near]
        return loss, {
            "mae_m": (metric_prediction_m - point_patch_distance).abs().mean(),
            "near_mae_m": near_error.mean() if near_error.numel() else loss.new_zeros(()),
        }

    def forward(
        self,
        tool_points: torch.Tensor,
        object_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.loss_from_clouds(tool_points, object_points)

    def loss_from_clouds(
        self,
        tool_points: torch.Tensor,
        object_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train directly from two point-cloud batches with no contact record."""

        encoded = self.encoder.encode(tool_points, object_points)
        patch_ids = self._selected_patch_ids(tool_points.device)

        tool_patches = self._gather_patches(tool_points, encoded.tool_patch_idx)[:, patch_ids]
        object_patches = self._gather_patches(object_points, encoded.obj_patch_idx)[:, patch_ids]
        tool_centers = encoded.tool_patch_centers[:, patch_ids]
        object_centers = encoded.obj_patch_centers[:, patch_ids]
        tool_queries, tool_query_relative = self._sample_queries(tool_patches, tool_centers)
        object_queries, object_query_relative = self._sample_queries(
            object_patches, object_centers
        )
        with torch.no_grad():
            tool_distance = torch.linalg.vector_norm(
                tool_queries.unsqueeze(-2) - tool_patches.unsqueeze(-3), dim=-1
            ).min(dim=-1).values
            object_distance = torch.linalg.vector_norm(
                object_queries.unsqueeze(-2) - object_patches.unsqueeze(-3), dim=-1
            ).min(dim=-1).values
        tool_loss, tool_metrics = self._decode_loss(
            encoded.fused_tokens[:, : self.num_patches][:, patch_ids],
            tool_query_relative,
            tool_distance,
        )
        object_loss, object_metrics = self._decode_loss(
            encoded.fused_tokens[:, self.num_patches :][:, patch_ids],
            object_query_relative,
            object_distance,
        )
        loss = tool_loss + object_loss
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "patch_distance_loss": float(loss.detach().cpu()),
            "tool_patch_distance_loss": float(tool_loss.detach().cpu()),
            "object_patch_distance_loss": float(object_loss.detach().cpu()),
            "patch_distance_mae_m": float(
                (0.5 * (tool_metrics["mae_m"] + object_metrics["mae_m"])).detach().cpu()
            ),
            "patch_distance_near_mae_m": float(
                (0.5 * (tool_metrics["near_mae_m"] + object_metrics["near_mae_m"])).detach().cpu()
            ),
        }

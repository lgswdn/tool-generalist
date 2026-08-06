"""Raw patch contract used to distill rank-10 TCE tokens into PointNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pretrain.patch_oracle_probe import (
    CONTACT_THRESHOLDS_M,
    _signed_log_sdf,
)


POINT_FEATURE_NAMES = (
    "local_point_x",
    "local_point_y",
    "local_point_z",
    "signed_sdf",
    "signed_log_sdf",
    "closest_mesh_displacement_x",
    "closest_mesh_displacement_y",
    "closest_mesh_displacement_z",
    "closest_mesh_direction_x",
    "closest_mesh_direction_y",
    "closest_mesh_direction_z",
    "closest_mesh_normal_x",
    "closest_mesh_normal_y",
    "closest_mesh_normal_z",
    "contact_within_0.5mm",
    "contact_within_1mm",
    "contact_within_2mm",
    "contact_within_5mm",
    "contact_within_10mm",
    "is_penetrating",
)
PATCH_METADATA_NAMES = (
    "center_x",
    "center_y",
    "center_z",
    "patch_is_tool",
    "patch_is_object",
)


def build_rank10_pointnet_source(
    *,
    patch_points: torch.Tensor,
    patch_centers: torch.Tensor,
    signed_sdf: torch.Tensor,
    closest_displacement: torch.Tensor,
    closest_normal: torch.Tensor,
    patch_is_tool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strict per-point inputs and per-patch metadata."""

    if patch_points.shape[:-2] != patch_centers.shape[:-1]:
        raise ValueError("patch point and center axes do not match")
    if patch_points.shape[-2:] != (32, 3):
        raise ValueError(
            f"rank-10 PointNet requires 32x3 patches, got {patch_points.shape[-2:]}"
        )
    if signed_sdf.shape != patch_points.shape[:-1]:
        raise ValueError("signed SDF shape does not match patch points")
    if closest_displacement.shape != patch_points.shape:
        raise ValueError("closest displacement shape does not match patch points")
    if closest_normal.shape != patch_points.shape:
        raise ValueError("closest normal shape does not match patch points")
    if patch_is_tool.shape != patch_points.shape[:-2]:
        raise ValueError("patch type shape does not match patch points")

    dtype = patch_points.dtype
    local = patch_points - patch_centers.unsqueeze(-2)
    distance_direction = F.normalize(
        closest_displacement, dim=-1, eps=1e-8
    )
    contact_flags = torch.stack(
        [
            (signed_sdf.abs() <= threshold).to(dtype)
            for threshold in CONTACT_THRESHOLDS_M
        ],
        dim=-1,
    )
    point_features = torch.cat(
        (
            local,
            signed_sdf.unsqueeze(-1),
            _signed_log_sdf(signed_sdf).unsqueeze(-1),
            closest_displacement,
            distance_direction,
            closest_normal,
            contact_flags,
            (signed_sdf < 0).to(dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    tool = patch_is_tool.to(dtype).unsqueeze(-1)
    patch_metadata = torch.cat(
        (patch_centers, tool, 1.0 - tool),
        dim=-1,
    )
    if point_features.shape[-1] != len(POINT_FEATURE_NAMES):
        raise RuntimeError("rank-10 PointNet point-feature contract mismatch")
    if patch_metadata.shape[-1] != len(PATCH_METADATA_NAMES):
        raise RuntimeError("rank-10 PointNet patch-metadata contract mismatch")
    if not (
        torch.isfinite(point_features).all()
        and torch.isfinite(patch_metadata).all()
    ):
        raise ValueError("rank-10 PointNet source features must be finite")
    return point_features, patch_metadata

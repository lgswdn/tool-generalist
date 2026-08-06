"""Cheap analytic patch features from point-cloud nearest-neighbor geometry."""

from __future__ import annotations

import math

import torch


LOG_RESOLUTION_M = 0.005
LOG_CAP_M = 0.05
SOFT_DISTANCE_SCALES_M = (0.0005, 0.001, 0.002, 0.005)
DISTANCE_QUANTILES = (0.10, 0.50, 0.90)


FAST_POINTCLOUD_PATCH_FEATURE_NAMES: tuple[str, ...] = (
    "center_x",
    "center_y",
    "center_z",
    "local_mean_x",
    "local_mean_y",
    "local_mean_z",
    "local_extent_x",
    "local_extent_y",
    "local_extent_z",
    "rms_radius",
    "pointcloud_distance_min",
    "pointcloud_distance_max",
    "pointcloud_distance_mean",
    "pointcloud_distance_std",
    "pointcloud_distance_q10",
    "pointcloud_distance_q50",
    "pointcloud_distance_q90",
    "log_pointcloud_distance_min",
    "log_pointcloud_distance_max",
    "log_pointcloud_distance_mean",
    "log_pointcloud_distance_std",
    "soft_pointcloud_contact_0.5mm",
    "soft_pointcloud_contact_1mm",
    "soft_pointcloud_contact_2mm",
    "soft_pointcloud_contact_5mm",
    "nearest_pointcloud_direction_x",
    "nearest_pointcloud_direction_y",
    "nearest_pointcloud_direction_z",
    "mean_pointcloud_direction_x",
    "mean_pointcloud_direction_y",
    "mean_pointcloud_direction_z",
    "soft_pointcloud_direction_2mm_x",
    "soft_pointcloud_direction_2mm_y",
    "soft_pointcloud_direction_2mm_z",
    "patch_is_tool",
)


def _log_distance(distance: torch.Tensor) -> torch.Tensor:
    denominator = math.log1p(LOG_CAP_M / LOG_RESOLUTION_M)
    return torch.log1p(distance.clamp(max=LOG_CAP_M) / LOG_RESOLUTION_M) / denominator


def build_fast_pointcloud_patch_features(point_features: torch.Tensor) -> torch.Tensor:
    """Reduce cheap point-cloud proximity inputs independently within each patch.

    ``point_features`` has shape ``(..., K, 11)`` and follows the existing
    ``FAST_POINT_FEATURE_NAMES`` contract:

    ``(local_xyz, center_xyz, unsigned_nn_distance, nn_direction_xyz, is_tool)``.

    The only cross-body operation needed upstream is one nearest-neighbor query
    between the two 512-point clouds.  This function performs only reductions
    over the K points of the current patch; it has no PointNet, mesh query,
    eigendecomposition, least-squares fit, or cross-patch network.
    """

    if point_features.ndim < 2 or point_features.shape[-1] != 11:
        raise ValueError("point_features must have shape (..., K, 11)")
    if not torch.isfinite(point_features).all():
        raise ValueError("point-cloud patch inputs must be finite")

    local = point_features[..., 0:3]
    center_samples = point_features[..., 3:6]
    distance = point_features[..., 6].clamp_min(0)
    direction = point_features[..., 7:10]
    body_samples = point_features[..., 10]

    # Center and body type are repeated for all points by the source contract.
    center = center_samples.mean(dim=-2)
    patch_is_tool = body_samples.mean(dim=-1, keepdim=True)
    local_mean = local.mean(dim=-2)
    local_extent = local.amax(dim=-2) - local.amin(dim=-2)
    rms_radius = local.square().sum(dim=-1).mean(dim=-1, keepdim=True).sqrt()

    distance_quantiles = torch.quantile(
        distance,
        torch.tensor(
            DISTANCE_QUANTILES,
            dtype=distance.dtype,
            device=distance.device,
        ),
        dim=-1,
    ).movedim(0, -1)
    distance_moments = torch.stack(
        (
            distance.amin(dim=-1),
            distance.amax(dim=-1),
            distance.mean(dim=-1),
            distance.std(dim=-1, unbiased=False),
        ),
        dim=-1,
    )

    log_distance = _log_distance(distance)
    log_moments = torch.stack(
        (
            log_distance.amin(dim=-1),
            log_distance.amax(dim=-1),
            log_distance.mean(dim=-1),
            log_distance.std(dim=-1, unbiased=False),
        ),
        dim=-1,
    )
    soft_contact = torch.stack(
        [torch.exp(-distance / scale).mean(dim=-1) for scale in SOFT_DISTANCE_SCALES_M],
        dim=-1,
    )

    nearest_index = distance.argmin(dim=-1, keepdim=True)
    nearest_direction = direction.gather(
        -2, nearest_index.unsqueeze(-1).expand(*nearest_index.shape, 3)
    ).squeeze(-2)
    mean_direction = direction.mean(dim=-2)
    soft_weight = torch.exp(-distance / 0.002)
    soft_direction = (
        (direction * soft_weight.unsqueeze(-1)).sum(dim=-2)
        / soft_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    )

    result = torch.cat(
        (
            center,
            local_mean,
            local_extent,
            rms_radius,
            distance_moments,
            distance_quantiles,
            log_moments,
            soft_contact,
            nearest_direction,
            mean_direction,
            soft_direction,
            patch_is_tool,
        ),
        dim=-1,
    )
    if result.shape[-1] != len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES):
        raise RuntimeError(
            "fast point-cloud patch feature contract mismatch: "
            f"{result.shape[-1]} != {len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES)}"
        )
    if not torch.isfinite(result).all():
        raise RuntimeError("fast point-cloud patch features contain non-finite values")
    return result

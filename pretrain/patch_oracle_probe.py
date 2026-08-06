"""Strictly patch-local oracle features and a small rank-token probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CENTER_SCALE_M = 0.30
PATCH_SCALE_M = 0.05
SDF_SCALE_M = 0.30
LOG_RESOLUTION_M = 0.005
LOG_CAP_M = 0.05
CONTACT_THRESHOLDS_M = (0.0005, 0.001, 0.002, 0.005, 0.010)
SOFT_CONTACT_SCALES_M = (0.0005, 0.001, 0.002, 0.005, 0.010)
SOFT_LOCATION_SCALES_M = (0.001, 0.002, 0.005)
SDF_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ABS_SDF_QUANTILES = (0.10, 0.50, 0.90)
LOG_SDF_QUANTILES = (0.10, 0.50, 0.90)


def _axis_names(prefix: str) -> list[str]:
    return [f"{prefix}_{axis}" for axis in "xyz"]


PATCH_ORACLE_FEATURE_NAMES: tuple[str, ...] = tuple(
    _axis_names("center")
    + _axis_names("local_mean")
    + _axis_names("local_std")
    + _axis_names("local_extent")
    + ["rms_radius", "max_radius"]
    + ["cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz", "cov_zz"]
    + ["pca_eigenvalue_0", "pca_eigenvalue_1", "pca_eigenvalue_2"]
    + _axis_names("canonical_normal")
    + ["linearity", "planarity", "scattering"]
    + ["signed_sdf_min", "signed_sdf_max", "signed_sdf_mean", "signed_sdf_std"]
    + [f"signed_sdf_q{int(q * 100):02d}" for q in SDF_QUANTILES]
    + ["abs_sdf_min", "abs_sdf_mean", "abs_sdf_std"]
    + [f"abs_sdf_q{int(q * 100):02d}" for q in ABS_SDF_QUANTILES]
    + ["log_sdf_min", "log_sdf_max", "log_sdf_mean", "log_sdf_std"]
    + [f"log_sdf_q{int(q * 100):02d}" for q in LOG_SDF_QUANTILES]
    + [f"contact_fraction_{threshold * 1000:g}mm" for threshold in CONTACT_THRESHOLDS_M]
    + ["penetration_fraction", "penetration_mean_depth", "penetration_max_depth"]
    + [f"soft_contact_{scale * 1000:g}mm" for scale in SOFT_CONTACT_SCALES_M]
    + [
        f"soft_location_{scale * 1000:g}mm_{axis}"
        for scale in SOFT_LOCATION_SCALES_M
        for axis in "xyz"
    ]
    + _axis_names("local_sdf_cov")
    + _axis_names("local_sdf_gradient_direction")
    + ["local_sdf_gradient_magnitude"]
    + _axis_names("closest_mesh_displacement")
    + _axis_names("closest_mesh_direction")
    + _axis_names("closest_mesh_normal")
    + ["patch_closest_normal_alignment"]
    + [f"closest_displacement_pca_{axis}" for axis in ("normal", "tangent1", "tangent2")]
    + [f"closest_normal_pca_{axis}" for axis in ("normal", "tangent1", "tangent2")]
    + [
        f"soft_mesh_normal_{scale * 1000:g}mm_{axis}"
        for scale in SOFT_LOCATION_SCALES_M
        for axis in "xyz"
    ]
    + [f"soft_mesh_normal_concentration_{scale * 1000:g}mm" for scale in SOFT_LOCATION_SCALES_M]
    + [
        "quadratic_sdf_intercept",
        "quadratic_sdf_linear_normal",
        "quadratic_sdf_linear_tangent1",
        "quadratic_sdf_linear_tangent2",
        "quadratic_sdf_square_normal",
        "quadratic_sdf_normal_tangent1",
        "quadratic_sdf_normal_tangent2",
        "quadratic_sdf_square_tangent1",
        "quadratic_sdf_tangent1_tangent2",
        "quadratic_sdf_square_tangent2",
        "quadratic_sdf_residual_rms",
        "quadratic_sdf_residual_max",
        "quadratic_sdf_fit_r2",
        "signed_sdf_skewness",
        "signed_sdf_excess_kurtosis",
        "signed_sdf_trimmed_mean",
    ]
    + _axis_names("contact_centroid_local")
    + _axis_names("penetration_centroid_local")
    + _axis_names("penetration_std_local")
    + ["patch_is_tool", "patch_is_object"]
)


def _canonicalize_vector_sign(vector: torch.Tensor) -> torch.Tensor:
    axis = vector.abs().argmax(dim=-1, keepdim=True)
    pivot = vector.gather(-1, axis)
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    return vector * sign


def _signed_log_sdf(sdf: torch.Tensor) -> torch.Tensor:
    magnitude = sdf.abs().clamp(max=LOG_CAP_M)
    denominator = math.log1p(LOG_CAP_M / LOG_RESOLUTION_M)
    return sdf.sign() * torch.log1p(magnitude / LOG_RESOLUTION_M) / denominator


def build_patch_oracle_features(
    *,
    patch_points: torch.Tensor,
    patch_centers: torch.Tensor,
    signed_sdf: torch.Tensor,
    closest_displacement: torch.Tensor,
    closest_normal: torch.Tensor,
    patch_is_tool: torch.Tensor,
) -> torch.Tensor:
    """Build independent features for each patch.

    Every reduction is over the points within the same patch.  There is no
    reduction, attention, sorting, ranking, or distance query across patches,
    so permuting the patch axis permutes the output in exactly the same way.

    Args:
        patch_points: ``(..., P, K, 3)`` object-centered patch points.
        patch_centers: ``(..., P, 3)`` FPS centers for those patches.
        signed_sdf: ``(..., P, K)`` true point-to-opposite-mesh signed SDF.
        closest_displacement: ``(..., P, K, 3)`` query-to-mesh vector.
        closest_normal: ``(..., P, K, 3)`` selected opposite-triangle normal.
        patch_is_tool: ``(..., P)`` Boolean/0-1 patch type.
    """

    if patch_points.ndim < 4 or patch_points.shape[-1] != 3:
        raise ValueError("patch_points must have shape (..., P, K, 3)")
    if patch_centers.shape != patch_points.shape[:-2] + (3,):
        raise ValueError("patch_centers shape must match patch_points (..., P, 3)")
    if signed_sdf.shape != patch_points.shape[:-1]:
        raise ValueError("signed_sdf shape must match patch_points (..., P, K)")
    if closest_displacement.shape != patch_points.shape:
        raise ValueError("closest_displacement shape must match patch_points")
    if closest_normal.shape != patch_points.shape:
        raise ValueError("closest_normal shape must match patch_points")
    if patch_is_tool.shape != patch_points.shape[:-2]:
        raise ValueError("patch_is_tool shape must match patch_points (..., P)")
    if not (
        torch.isfinite(patch_points).all()
        and torch.isfinite(patch_centers).all()
        and torch.isfinite(signed_sdf).all()
        and torch.isfinite(closest_displacement).all()
        and torch.isfinite(closest_normal).all()
    ):
        raise ValueError("patch oracle inputs must be finite")

    dtype = patch_points.dtype
    local = patch_points - patch_centers.unsqueeze(-2)
    local_mean = local.mean(dim=-2)
    centered = local - local_mean.unsqueeze(-2)
    point_count = patch_points.shape[-2]
    variance_denominator = max(point_count - 1, 1)
    covariance = torch.matmul(centered.transpose(-1, -2), centered) / variance_denominator
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    normal = _canonicalize_vector_sign(eigenvectors[..., :, 0])
    largest = eigenvalues[..., 2].clamp_min(1e-12)
    linearity = (eigenvalues[..., 2] - eigenvalues[..., 1]) / largest
    planarity = (eigenvalues[..., 1] - eigenvalues[..., 0]) / largest
    scattering = eigenvalues[..., 0] / largest

    radius = torch.linalg.vector_norm(local, dim=-1)
    local_extent = local.amax(dim=-2) - local.amin(dim=-2)
    local_std = local.std(dim=-2, unbiased=False)
    covariance_terms = torch.stack(
        (
            covariance[..., 0, 0],
            covariance[..., 0, 1],
            covariance[..., 0, 2],
            covariance[..., 1, 1],
            covariance[..., 1, 2],
            covariance[..., 2, 2],
        ),
        dim=-1,
    )

    sdf_std = signed_sdf.std(dim=-1, unbiased=False)
    sdf_quantiles = torch.quantile(
        signed_sdf,
        torch.tensor(SDF_QUANTILES, device=signed_sdf.device, dtype=dtype),
        dim=-1,
    ).movedim(0, -1)
    abs_sdf = signed_sdf.abs()
    abs_quantiles = torch.quantile(
        abs_sdf,
        torch.tensor(ABS_SDF_QUANTILES, device=signed_sdf.device, dtype=dtype),
        dim=-1,
    ).movedim(0, -1)
    log_sdf = _signed_log_sdf(signed_sdf)
    log_quantiles = torch.quantile(
        log_sdf,
        torch.tensor(LOG_SDF_QUANTILES, device=signed_sdf.device, dtype=dtype),
        dim=-1,
    ).movedim(0, -1)

    contact_fractions = torch.stack(
        [(abs_sdf <= threshold).to(dtype).mean(dim=-1) for threshold in CONTACT_THRESHOLDS_M],
        dim=-1,
    )
    penetration_depth = (-signed_sdf).clamp_min(0)
    penetration_mask = signed_sdf < 0
    penetration_fraction = penetration_mask.to(dtype).mean(dim=-1)
    penetration_mean = penetration_depth.sum(dim=-1) / penetration_mask.sum(dim=-1).clamp_min(1)
    penetration_max = penetration_depth.amax(dim=-1)
    soft_contact = torch.stack(
        [torch.exp(-abs_sdf / scale).mean(dim=-1) for scale in SOFT_CONTACT_SCALES_M],
        dim=-1,
    )
    soft_locations = []
    for scale in SOFT_LOCATION_SCALES_M:
        weight = torch.exp(-abs_sdf / scale)
        soft_locations.append(
            (weight.unsqueeze(-1) * local).sum(dim=-2)
            / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )

    sdf_centered = signed_sdf - signed_sdf.mean(dim=-1, keepdim=True)
    local_sdf_cov = (centered * sdf_centered.unsqueeze(-1)).mean(dim=-2)
    regularizer = torch.eye(3, device=patch_points.device, dtype=dtype) * 1e-8
    gradient = torch.linalg.solve(covariance + regularizer, local_sdf_cov.unsqueeze(-1)).squeeze(-1)
    gradient_magnitude = torch.linalg.vector_norm(gradient, dim=-1)
    gradient_direction = F.normalize(gradient, dim=-1, eps=1e-8)

    nearest_index = abs_sdf.argmin(dim=-1, keepdim=True)
    vector_index = nearest_index.unsqueeze(-1).expand(*nearest_index.shape, 3)
    nearest_displacement = closest_displacement.gather(-2, vector_index).squeeze(-2)
    nearest_direction = F.normalize(nearest_displacement, dim=-1, eps=1e-8)
    nearest_normal = F.normalize(
        closest_normal.gather(-2, vector_index).squeeze(-2), dim=-1, eps=1e-8
    )
    normal_alignment = (normal * nearest_normal).sum(dim=-1)
    nearest_displacement_pca = torch.matmul(
        nearest_displacement.unsqueeze(-2), eigenvectors
    ).squeeze(-2)
    nearest_normal_pca = torch.matmul(nearest_normal.unsqueeze(-2), eigenvectors).squeeze(-2)

    soft_normal_directions = []
    soft_normal_concentrations = []
    for scale in SOFT_LOCATION_SCALES_M:
        weight = torch.exp(-abs_sdf / scale)
        mean_normal = (
            weight.unsqueeze(-1) * closest_normal
        ).sum(dim=-2) / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        concentration = torch.linalg.vector_norm(mean_normal, dim=-1)
        soft_normal_directions.append(F.normalize(mean_normal, dim=-1, eps=1e-8))
        soft_normal_concentrations.append(concentration)

    # Explicit local quadratic SDF model in the patch PCA frame.  Coordinates
    # and distances are physically normalized before fitting, so every
    # coefficient has a stable, interpretable scale.
    local_pca = torch.matmul(local, eigenvectors) / PATCH_SCALE_M
    pca_normal, pca_tangent1, pca_tangent2 = local_pca.unbind(dim=-1)
    quadratic_design = torch.stack(
        (
            torch.ones_like(pca_normal),
            pca_normal,
            pca_tangent1,
            pca_tangent2,
            pca_normal.square(),
            pca_normal * pca_tangent1,
            pca_normal * pca_tangent2,
            pca_tangent1.square(),
            pca_tangent1 * pca_tangent2,
            pca_tangent2.square(),
        ),
        dim=-1,
    )
    normalized_sdf = signed_sdf / SDF_SCALE_M
    quadratic_gram = torch.matmul(quadratic_design.transpose(-1, -2), quadratic_design)
    quadratic_rhs = torch.matmul(
        quadratic_design.transpose(-1, -2), normalized_sdf.unsqueeze(-1)
    )
    quadratic_regularizer = torch.eye(10, device=patch_points.device, dtype=dtype) * 1e-4
    quadratic_regularizer[0, 0] = 0.0
    quadratic_coefficients = torch.linalg.solve(
        quadratic_gram + quadratic_regularizer,
        quadratic_rhs,
    ).squeeze(-1)
    quadratic_prediction = (
        quadratic_design * quadratic_coefficients.unsqueeze(-2)
    ).sum(dim=-1)
    quadratic_residual = normalized_sdf - quadratic_prediction
    residual_square_sum = quadratic_residual.square().sum(dim=-1)
    target_square_sum = (
        normalized_sdf - normalized_sdf.mean(dim=-1, keepdim=True)
    ).square().sum(dim=-1)
    quadratic_fit_r2 = 1.0 - residual_square_sum / target_square_sum.clamp_min(1e-12)

    standardized_sdf = sdf_centered / sdf_std.unsqueeze(-1).clamp_min(1e-8)
    sdf_skewness = standardized_sdf.pow(3).mean(dim=-1)
    sdf_excess_kurtosis = standardized_sdf.pow(4).mean(dim=-1) - 3.0
    sorted_sdf = signed_sdf.sort(dim=-1).values
    trim_count = max(1, int(point_count * 0.1))
    sdf_trimmed_mean = sorted_sdf[..., trim_count:-trim_count].mean(dim=-1)

    contact_weight = (abs_sdf <= 0.002).to(dtype)
    contact_centroid = (
        contact_weight.unsqueeze(-1) * local
    ).sum(dim=-2) / contact_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    penetration_weight = penetration_mask.to(dtype)
    penetration_centroid = (
        penetration_weight.unsqueeze(-1) * local
    ).sum(dim=-2) / penetration_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    penetration_variance = (
        penetration_weight.unsqueeze(-1)
        * (local - penetration_centroid.unsqueeze(-2)).square()
    ).sum(dim=-2) / penetration_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    penetration_std_local = penetration_variance.sqrt()

    tool_type = patch_is_tool.to(dtype)
    object_type = 1.0 - tool_type
    groups = [
        patch_centers / CENTER_SCALE_M,
        local_mean / PATCH_SCALE_M,
        local_std / PATCH_SCALE_M,
        local_extent / PATCH_SCALE_M,
        radius.square().mean(dim=-1, keepdim=True).sqrt() / PATCH_SCALE_M,
        radius.amax(dim=-1, keepdim=True) / PATCH_SCALE_M,
        covariance_terms / (PATCH_SCALE_M**2),
        eigenvalues / (PATCH_SCALE_M**2),
        normal,
        torch.stack((linearity, planarity, scattering), dim=-1),
        torch.stack(
            (
                signed_sdf.amin(dim=-1),
                signed_sdf.amax(dim=-1),
                signed_sdf.mean(dim=-1),
                sdf_std,
            ),
            dim=-1,
        )
        / SDF_SCALE_M,
        sdf_quantiles / SDF_SCALE_M,
        torch.stack(
            (abs_sdf.amin(dim=-1), abs_sdf.mean(dim=-1), abs_sdf.std(dim=-1, unbiased=False)),
            dim=-1,
        )
        / SDF_SCALE_M,
        abs_quantiles / SDF_SCALE_M,
        torch.stack(
            (
                log_sdf.amin(dim=-1),
                log_sdf.amax(dim=-1),
                log_sdf.mean(dim=-1),
                log_sdf.std(dim=-1, unbiased=False),
            ),
            dim=-1,
        ),
        log_quantiles,
        contact_fractions,
        torch.stack(
            (
                penetration_fraction,
                penetration_mean / LOG_CAP_M,
                penetration_max / LOG_CAP_M,
            ),
            dim=-1,
        ),
        soft_contact,
        torch.cat(soft_locations, dim=-1) / PATCH_SCALE_M,
        local_sdf_cov / (PATCH_SCALE_M * SDF_SCALE_M),
        gradient_direction,
        torch.tanh(gradient_magnitude).unsqueeze(-1),
        nearest_displacement / PATCH_SCALE_M,
        nearest_direction,
        nearest_normal,
        normal_alignment.unsqueeze(-1),
        nearest_displacement_pca / PATCH_SCALE_M,
        nearest_normal_pca,
        torch.cat(soft_normal_directions, dim=-1),
        torch.stack(soft_normal_concentrations, dim=-1),
        quadratic_coefficients,
        torch.stack(
            (
                quadratic_residual.square().mean(dim=-1).sqrt(),
                quadratic_residual.abs().amax(dim=-1),
                quadratic_fit_r2.clamp(-10.0, 1.0),
                sdf_skewness.clamp(-10.0, 10.0),
                sdf_excess_kurtosis.clamp(-10.0, 20.0),
                sdf_trimmed_mean / SDF_SCALE_M,
            ),
            dim=-1,
        ),
        contact_centroid / PATCH_SCALE_M,
        penetration_centroid / PATCH_SCALE_M,
        penetration_std_local / PATCH_SCALE_M,
        tool_type.unsqueeze(-1),
        object_type.unsqueeze(-1),
    ]
    features = torch.cat(groups, dim=-1)
    if features.shape[-1] != len(PATCH_ORACLE_FEATURE_NAMES):
        raise RuntimeError(
            f"patch oracle feature contract mismatch: {features.shape[-1]} != "
            f"{len(PATCH_ORACLE_FEATURE_NAMES)}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("patch oracle feature extraction produced non-finite values")
    return features


class PatchOracleToRankToken(nn.Module):
    """Shared patchwise MLP; never mixes information across the patch axis."""

    def __init__(
        self,
        *,
        input_dim: int = len(PATCH_ORACLE_FEATURE_NAMES),
        hidden_dims: tuple[int, ...] = (128, 64),
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        dims = (int(input_dim),) + tuple(int(dim) for dim in hidden_dims) + (int(output_dim),)
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.net(patch_features)


class PatchOracleResidualBlock(nn.Module):
    """Pre-normalized residual block that remains independently patchwise."""

    def __init__(self, width: int = 256, hidden_width: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden_width)
        self.fc2 = nn.Linear(hidden_width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.fc2(F.gelu(self.fc1(self.norm(values))))


class DeepPatchOracleToRankToken(nn.Module):
    """Deep residual patch-oracle probe; never mixes different patches."""

    def __init__(
        self,
        *,
        input_dim: int = len(PATCH_ORACLE_FEATURE_NAMES),
        width: int = 256,
        residual_hidden_width: int = 512,
        num_residual_blocks: int = 4,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(int(input_dim), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            PatchOracleResidualBlock(int(width), int(residual_hidden_width))
            for _ in range(int(num_residual_blocks))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), 128),
            nn.GELU(),
            nn.Linear(128, int(output_dim)),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        latent = self.input(patch_features)
        for block in self.blocks:
            latent = block(latent)
        return self.output(latent)

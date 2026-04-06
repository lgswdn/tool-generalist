# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Momentum-Aware Point Cloud Encoder (next-frame point prediction)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from rsl_rl.modules.util.config import ConfigBase
from rsl_rl.modules.models.cloud.point_mae import get_pos_enc_module

from flash_attn import flash_attn_func

# ============================================================================
# Point Cloud Preprocessing
# ============================================================================

from pytorch3d.ops import sample_farthest_points, knn_points


# ============================================================================
# 6D Rotation Representation Utilities
# ============================================================================


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    https://arxiv.org/abs/1812.07035
    
    Args:
        d6: [..., 6] 6D rotation representation
    Returns:
        rot_mat: [..., 3, 3] rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D rotation representation.
    
    Args:
        matrix: [..., 3, 3] rotation matrix
    Returns:
        d6: [..., 6] 6D rotation representation
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


# ============================================================================
# ViT-style blocks (pre-norm MHSA + MLP)
# ============================================================================


class ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            # For flash_attn, we need to create Q, K, V projections manually
            self.head_dim = dim // num_heads
            assert (
                dim % num_heads == 0
            ), f"dim {dim} must be divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out_proj = nn.Linear(dim, dim)
            self.attn_dropout = attn_dropout
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )

        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        y = self.norm1(x)

        if self.use_flash_attn:
            # Flash attention path
            B, L, D = y.shape
            # Compute Q, K, V
            qkv = self.qkv(y)  # [B, L, 3*D]
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, L, head_dim]

            # Flash attention requires [B, L, num_heads, head_dim] format
            q = q.transpose(1, 2)  # [B, L, num_heads, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Flash attention only supports fp16/bf16, convert if needed
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            # Use flash_attn (causal=False for bidirectional attention)
            y = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                causal=False,
            )
            # y: [B, L, num_heads, head_dim]

            # Reshape and project output
            y = y.reshape(B, L, D).to(torch.float32)
            y = self.out_proj(y)
        else:
            # Standard attention path
            y, _ = self.attn(y, y, y, need_weights=False).to(torch.float32)

        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x


def morton_code_3d(p: torch.Tensor) -> torch.Tensor:
    """
    Compute Morton code (Z-order curve) for 3D points.

    Args:
        p: [..., 3] 3D points (x, y, z) in integer coordinates

    Returns:
        morton: [...] Morton codes
    """
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    # Interleave bits: x, y, z -> ...z2y2x2z1y1x1z0y0x0
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249

    y = (y | (y << 16)) & 0x030000FF
    y = (y | (y << 8)) & 0x0300F00F
    y = (y | (y << 4)) & 0x030C30C3
    y = (y | (y << 2)) & 0x09249249

    z = (z | (z << 16)) & 0x030000FF
    z = (z | (z << 8)) & 0x0300F00F
    z = (z | (z << 4)) & 0x030C30C3
    z = (z | (z << 2)) & 0x09249249

    return x | (y << 1) | (z << 2)


def spatial_sort_by_curve(positions: torch.Tensor) -> torch.Tensor:
    """
    Sort points by spatial curve (Morton or Hilbert).

    Args:
        positions: [B, N, 3] 3D point positions

    Returns:
        sorted_indices: [B, N] indices to sort points along the spatial curve
    """
    B, N, _ = positions.shape
    device = positions.device

    # Normalize positions to integer grid [0, HMAX]
    HMAX = (1 << 10) - 1  # 10 bits per dimension = 30 bits total < 32 bits
    bmin = positions.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
    bmax = positions.max(dim=1, keepdim=True)[0]  # [B, 1, 3]

    # Avoid division by zero
    brange = bmax - bmin
    brange = torch.clamp(brange, min=1e-8)

    # Convert to fixed-point integer representation
    p_int = torch.floor(HMAX * (positions - bmin) / brange).to(
        dtype=torch.int32
    )  # [B, N, 3]

    codes = morton_code_3d(p_int)  # [B, N]

    # Sort by codes
    sorted_indices = torch.argsort(codes, dim=1)  # [B, N]

    return sorted_indices


class PointCloudPreprocessor(nn.Module):
    """
    Preprocess point clouds: patch grouping + normalization.

    Features:
    - Patch grouping using FPS+KNN or spatial curve (Hilbert/Morton)
    - Mass and velocity normalization using running statistics

    Patch Grouping Methods:
    1. "fps_knn" (FPS + KNN):
       - Uses Farthest Point Sampling to select patch centers
       - Groups points using K-nearest neighbors
       - Advantages:
         * Ensures uniform distribution of patch centers
         * Good coverage of the entire point cloud
       - Disadvantages:
         * Slower: O(n^2) for FPS + O(n*k) for KNN
         * May not preserve spatial locality within patches

    2. "spatial_curve" (Hilbert/Morton curve):
       - Sorts points along a space-filling curve (Hilbert or Morton)
       - Groups consecutive points along the curve into patches
       - Advantages:
         * Faster: O(n log n) for sorting
         * Better spatial locality: nearby points in space are grouped together
         * Deterministic (no random initialization)
         * Better for dense point clouds
       - Disadvantages:
         * Patch centers may be less uniformly distributed
         * May not cover sparse regions as well as FPS
    """

    def __init__(
        self,
        patch_size: int = 32,
        num_points: int = 512,
        patch_grouping_method: str = "fps_knn",  # "fps_knn" or "spatial_curve"
        normalize_mass: bool = True,
        normalize_velocity: bool = True,
        momentum: float = 0.1,  # For running statistics
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_points = num_points
        self.num_patches = num_points // patch_size
        self.patch_grouping_method = patch_grouping_method
        self.normalize_mass = normalize_mass
        self.normalize_velocity = normalize_velocity
        self.momentum = momentum

        # Register running statistics for mass and velocity
        self.register_buffer("mass_mean", torch.zeros(1))
        self.register_buffer("mass_std", torch.ones(1))
        self.register_buffer("velocity_mean", torch.zeros(3))
        self.register_buffer("velocity_std", torch.ones(3))
        self.register_buffer("_mass_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "_velocity_initialized", torch.tensor(False, dtype=torch.bool)
        )

    def update_statistics(self, mass: torch.Tensor, velocity: torch.Tensor):
        if not self.training:
            return

        B, N, _ = velocity.shape

        valid_mask_mass = mass.abs() > 1e-6
        valid_mask_vel = velocity.norm(dim=-1) > 1e-6

        if valid_mask_mass.sum() > 0:
            valid_mass = mass[valid_mask_mass]
            batch_mean = valid_mass.mean()
            batch_std = valid_mass.std()

            if not bool(self._mass_initialized):
                self.mass_mean.fill_(batch_mean.item())
                self.mass_std.fill_(batch_std.item())
                self._mass_initialized.fill_(True)
            else:
                self.mass_mean = (
                    1 - self.momentum
                ) * self.mass_mean + self.momentum * batch_mean
                self.mass_std = (
                    1 - self.momentum
                ) * self.mass_std + self.momentum * batch_std

        if valid_mask_vel.sum() > 0:
            valid_velocity = velocity[valid_mask_vel]
            batch_mean = valid_velocity.mean(dim=0)
            batch_std = valid_velocity.std(dim=0)

            if not bool(self._velocity_initialized):
                self.velocity_mean.copy_(batch_mean)
                self.velocity_std.copy_(batch_std)
                self._velocity_initialized.fill_(True)
            else:
                self.velocity_mean = (
                    1 - self.momentum
                ) * self.velocity_mean + self.momentum * batch_mean
                self.velocity_std = (
                    1 - self.momentum
                ) * self.velocity_std + self.momentum * batch_std

    def normalize_features(self, pointcloud: torch.Tensor) -> torch.Tensor:
        normalized = pointcloud.clone()
        if self.normalize_mass and bool(self._mass_initialized):
            normalized[..., 3] = (normalized[..., 3] - self.mass_mean) / (
                self.mass_std + 1e-8
            )

        if self.normalize_velocity and bool(self._velocity_initialized):
            normalized[..., 4:7] = (normalized[..., 4:7] - self.velocity_mean) / (
                self.velocity_std + 1e-8
            )

        return normalized

    def denormalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        if not self.normalize_velocity or not self._velocity_initialized:
            return velocity

        return velocity * self.velocity_std + self.velocity_mean

    def group_into_patches(
        self,
        pointcloud: torch.Tensor,
        patch_size: Optional[int] = None,
        grouping_method: Optional[str] = None,
        group_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Group point cloud into patches using FPS+KNN or spatial curve.

        Args:
            pointcloud: [B, N, 7] point cloud (x, y, z, mass, vx, vy, vz)
            patch_size: Optional patch size. If None, uses self.patch_size
            grouping_method: Optional grouping method. If None, uses self.patch_grouping_method
            group_counts: Optional [num_groups] tensor of point counts per group.
                          If provided, FPS is performed within each group separately.

        Returns:
            patches: [B, num_patches, patch_size, 7] grouped and centered patches
            patch_centers: [B, num_patches, 3] patch center positions
            patch_indices: [B, num_patches, patch_size] indices mapping points to patches
        """
        B, N, D = pointcloud.shape
        device = pointcloud.device

        # Use provided patch_size or default
        if patch_size is None:
            patch_size = self.patch_size

        # Use provided grouping_method or default
        if grouping_method is None:
            grouping_method = self.patch_grouping_method

        # Extract positions (must be contiguous for pytorch3d FPS)
        positions = pointcloud[..., :3].contiguous()  # [B, N, 3]

        if grouping_method == "spatial_curve":
            num_patches = N // patch_size

            # Use spatial curve (Hilbert/Morton) for patch grouping
            # Sort points along the spatial curve
            sorted_indices = spatial_sort_by_curve(positions)  # [B, N]

            # Gather sorted points
            batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, N)
            sorted_points = pointcloud[batch_indices, sorted_indices, :]  # [B, N, 7]
            sorted_positions = sorted_points[..., :3]  # [B, N, 3]

            # Group consecutive points into patches
            patches_list = []
            patch_centers_list = []
            patch_indices_list = []

            for patch_idx in range(num_patches):
                start_idx = patch_idx * patch_size
                end_idx = min(start_idx + patch_size, N)
                actual_patch_size = end_idx - start_idx

                # Get patch points
                patch_points = sorted_points[
                    :, start_idx:end_idx, :
                ]  # [B, actual_patch_size, 7]
                patch_positions = sorted_positions[
                    :, start_idx:end_idx, :
                ]  # [B, actual_patch_size, 3]

                # Pad if necessary (shouldn't happen if N is divisible by patch_size)
                if actual_patch_size < patch_size:
                    padding = patch_size - actual_patch_size
                    patch_points = F.pad(
                        patch_points, (0, 0, 0, padding), mode="constant", value=0
                    )
                    patch_positions = F.pad(
                        patch_positions, (0, 0, 0, padding), mode="constant", value=0
                    )

                # Compute patch center (mean of patch positions)
                patch_center = patch_positions.mean(dim=1, keepdim=True)  # [B, 1, 3]

                # Center patch points
                patch_points_centered = patch_points.clone()
                patch_points_centered[..., :3] = patch_points[..., :3] - patch_center

                patches_list.append(
                    patch_points_centered.unsqueeze(1)
                )  # [B, 1, patch_size, 7]
                patch_centers_list.append(patch_center)  # [B, 1, 3]

                # Store original indices (mapped back from sorted indices)
                original_indices = sorted_indices[
                    :, start_idx:end_idx
                ]  # [B, actual_patch_size]
                if actual_patch_size < patch_size:
                    # Pad with last index
                    padding_indices = original_indices[:, -1:].expand(-1, padding)
                    original_indices = torch.cat(
                        [original_indices, padding_indices], dim=1
                    )
                patch_indices_list.append(
                    original_indices.unsqueeze(1)
                )  # [B, 1, patch_size]

            patches = torch.cat(patches_list, dim=1)  # [B, num_patches, patch_size, 7]
            patch_centers = torch.cat(patch_centers_list, dim=1)  # [B, num_patches, 3]
            patch_indices = torch.cat(
                patch_indices_list, dim=1
            )  # [B, num_patches, patch_size]

        else:
            # Default: FPS + KNN method
            if group_counts is not None:
                patches_list = []
                patch_centers_list = []
                patch_indices_list = []

                start_offset = 0
                for group_pts in group_counts:
                    if group_pts == 0:
                        continue

                    group_num_patches = int(group_pts.item() // patch_size)
                    if group_num_patches == 0:
                        start_offset += int(group_pts.item())
                        continue

                    group_positions = positions[
                        :, start_offset : start_offset + group_pts, :
                    ].contiguous()

                    # Sample patch centers using FPS within group
                    # Convert to float32 for FPS and KNN operations (distance calculations need precision)
                    group_positions_f32 = group_positions.to(dtype=torch.float32).contiguous()
                    group_patch_centers, _ = sample_farthest_points(
                        group_positions_f32, K=group_num_patches, random_start_point=False
                    )  # [B, group_num_patches, 3]

                    # Find K nearest neighbors within group
                    # Ensure both inputs are float32 for knn_points
                    _, group_nn_idx, _ = knn_points(
                        group_patch_centers,
                        group_positions_f32,
                        K=patch_size,
                        return_nn=False,
                        return_sorted=True,
                    )  # group_nn_idx: [B, group_num_patches, patch_size]

                    # Map indices back to global pointcloud
                    group_nn_idx = group_nn_idx + start_offset

                    # Gather points for each patch
                    batch_indices = (
                        torch.arange(B, device=device)
                        .view(B, 1, 1)
                        .expand(B, group_num_patches, patch_size)
                    )
                    group_patches = pointcloud[
                        batch_indices, group_nn_idx, :
                    ]  # [B, group_num_patches, patch_size, 7]

                    # Center patches
                    group_patch_centers_expanded = group_patch_centers.unsqueeze(
                        2
                    )  # [B, group_num_patches, 1, 3]
                    group_patches[..., :3] = (
                        group_patches[..., :3] - group_patch_centers_expanded
                    )

                    patches_list.append(group_patches)
                    patch_centers_list.append(group_patch_centers)
                    patch_indices_list.append(group_nn_idx)

                    start_offset += group_pts

                patches = torch.cat(patches_list, dim=1)
                patch_centers = torch.cat(patch_centers_list, dim=1)
                patch_indices = torch.cat(patch_indices_list, dim=1)
            else:
                num_patches = N // patch_size

                # Sample patch centers using FPS
                # Convert to float32 for FPS and KNN operations (distance calculations need precision)
                positions_f32 = positions.to(dtype=torch.float32)
                patch_centers, _ = sample_farthest_points(
                    positions_f32, K=num_patches, random_start_point=False
                )  # [B, num_patches, 3]

                # Find K nearest neighbors for each patch center
                # Ensure both inputs are float32 for knn_points
                _, nn_idx, _ = knn_points(
                    patch_centers,
                    positions_f32,
                    K=patch_size,
                    return_nn=False,
                    return_sorted=True,
                )  # nn_idx: [B, num_patches, patch_size]

                # Gather points for each patch
                batch_indices = (
                    torch.arange(B, device=device)
                    .view(B, 1, 1)
                    .expand(B, num_patches, patch_size)
                )
                patches = pointcloud[
                    batch_indices, nn_idx, :
                ]  # [B, num_patches, patch_size, 7]
                patch_indices = nn_idx

                # Center patches (subtract patch center from points)
                patch_centers_expanded = patch_centers.unsqueeze(
                    2
                )  # [B, num_patches, 1, 3]
                patches[..., :3] = patches[..., :3] - patch_centers_expanded

        return patches, patch_centers, patch_indices

    def forward(
        self,
        pointcloud: torch.Tensor,
        update_stats: bool = True,
        group_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess point cloud: normalize and group into patches.

        Args:
            pointcloud: [B, N, 7] (x, y, z, mass, vx, vy, vz)
            update_stats: Whether to update running statistics
            group_counts: Optional [num_groups] tensor of point counts per group.

        Returns:
            patches: [B, num_patches, patch_size, 7] grouped and normalized patches
            patch_centers: [B, num_patches, 3] patch center positions
            patch_indices: [B, num_patches, patch_size] indices mapping
        """
        # Update statistics if training
        if update_stats and self.training:
            mass = pointcloud[..., 3:4]  # [B, N, 1]
            velocity = pointcloud[..., 4:7]  # [B, N, 3]
            self.update_statistics(mass, velocity)

        normalized_pc = self.normalize_features(pointcloud)
        patches, patch_centers, patch_indices = self.group_into_patches(
            normalized_pc, group_counts=group_counts
        )

        return patches, patch_centers, patch_indices


class PointNetPatchEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dims=(64, 128),
        out_dim: int = 128,
    ):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
        )

        # 聚合后再投影（关键）
        self.proj = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """
        x: [B, P, N, C]
        return: patch_tokens [B, P, out_dim]
        """
        B, P, N, C = x.shape

        x = self.mlp1(x)  # [B, P, N, H1]
        x = self.mlp2(x)  # [B, P, N, H2]

        x_max = x.max(dim=2).values  # [B, P, H2]
        x_mean = x.mean(dim=2)  # [B, P, H2]

        x = torch.cat([x_max, x_mean], dim=-1)  # [B, P, 2H2]

        x = self.proj(x)  # [B, P, out_dim]
        return x


class MLPDecoder(nn.Module):
    """MLP decoder: fused features -> latent features (same last dim)."""

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, int(input_dim)))
        self.decoder = nn.Sequential(*layers)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.dim() not in (3, 4):
            raise ValueError(
                f"MLPDecoder expects fused_features dim in (3,4), got shape={tuple(fused_features.shape)}"
            )
        return self.decoder(fused_features)


# ============================================================================
# Main Momentum-Aware Point Cloud Encoder
# ============================================================================


@dataclass
class MomentumAwarePointEncoderConfig(ConfigBase):
    """Configuration for Momentum-Aware Point Cloud Encoder."""

    # Point cloud settings
    num_points_per_object: int = (
        512  # Total number of points in the point cloud for each object
    )
    num_obstacles: int = 1
    num_ee_points: int = 256

    # Preprocessing settings
    patch_size: int = 32  # Points per patch for object
    patch_grouping_method: str = (
        "fps_knn"  # "fps_knn" or "spatial_curve" (Morton curve)
    )
    normalize_position: bool = False  # Whether to normalize position
    normalize_mass: bool = True  # Whether to normalize mass
    normalize_velocity: bool = True  # Whether to normalize velocity
    stats_momentum: float = 0.01  # Momentum for running statistics

    # Patch encoder settings (PointNet style)
    patch_encoder_hidden_dims: tuple[int, ...] = (
        32,
        64,
    )  # Hidden dimensions for PointNet patch encoder
    patch_encoder_point_dim: int = 7  # Point dimension (x, y, z, mass, vx, vy, vz)
    encoder_feature_dim: int = 128  # Patch embedding dimension

    # Position embedding settings
    pos_embed_type: Optional[str] = "sine"  # "sine", "linear", "mlp", "nerf", None

    # ViT-style block stacking over patch tokens (point-cloud-only).
    vit_depth: int = 12
    vit_num_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    vit_attn_dropout: float = 0.0
    vit_use_flash_attn: bool = True
    vit_use_cls_token: bool = True
    # Optional type embedding: 0=object, 1=obstacle, 2=end-effector (if present)
    use_type_embedding: bool = True
    num_cloud_types: int = 3

    # EE flow conditioning settings (for pretraining with end-effector motion)
    use_ee_flow_conditioning: bool = False  # Whether to condition decoder on EE flow
    ee_flow_num_patches: int = 8  # Number of patches for EE flow (after mean pooling)
    ee_flow_mlp_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])  # Hidden dims for EE flow MLP encoder
    ee_flow_cross_attn_heads: int = 8  # Number of attention heads for cross attention
    ee_flow_cross_attn_dropout: float = 0.0  # Dropout for cross attention

    # Decoder settings (predict next frame)
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [64])
    decoder_predict_delta: bool = (
        False  # If True, predict delta (relative change), else predict absolute positions
    )
    enable_decoder: bool = (
        True  # Toggle decoder head; when False, forward returns encoded features only
    )
    
    # Delta pose prediction mode (alternative to point-wise prediction)
    predict_delta_pose: bool = False  # If True, predict 9D delta pose (3D translation + 6D rotation)
    delta_pose_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])  # Hidden dims for delta pose head

    # Unpatchify settings
    unpatchify_point_dim: int = 32
    unpatchify_hidden_dims: list[int] = field(default_factory=lambda: [])
    unpatchify_method: str = "scatter"  # "mlp" or "scatter"

    # Training settings
    freeze_encoder: bool = False
    encoder_weights_path: Optional[str] = None


class MomentumAwarePointEncoder(nn.Module):
    """Point-cloud encoder using ViT-style block stacking (object/obstacle/ee clouds)."""

    def __init__(self, cfg: MomentumAwarePointEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.enable_decoder = cfg.enable_decoder

        self.total_num_points = (
            cfg.num_points_per_object * (1 + cfg.num_obstacles) + cfg.num_ee_points
        )

        # Create point cloud preprocessor (patch grouping + normalization)
        self.preprocessor = PointCloudPreprocessor(
            patch_size=cfg.patch_size,
            num_points=self.total_num_points,
            patch_grouping_method=cfg.patch_grouping_method,
            normalize_mass=cfg.normalize_mass,
            normalize_velocity=cfg.normalize_velocity,
            momentum=cfg.stats_momentum,
        )

        # Calculate number of patches (for backward compatibility)
        self.num_patches = self.total_num_points // cfg.patch_size

        self.patch_encoder = PointNetPatchEncoder(
            in_dim=cfg.patch_encoder_point_dim,
            hidden_dims=cfg.patch_encoder_hidden_dims,
            out_dim=cfg.encoder_feature_dim,
        )

        # Create position embedding for patch centers
        if cfg.pos_embed_type is not None:
            self.pos_embed = get_pos_enc_module(
                cfg.pos_embed_type,
                cfg.encoder_feature_dim,
                in_channels=3,  # Patch center is 3D position
            )
        else:
            self.pos_embed = None

        if cfg.use_type_embedding:
            embed_dim = cfg.encoder_feature_dim  # Embedding dimension after stem
            self.type_embeddings = nn.Parameter(
                torch.zeros(cfg.num_cloud_types, embed_dim)
            )
            # Initialize with small random values
            nn.init.normal_(self.type_embeddings, std=0.02)
        else:
            self.type_embeddings = None

        # Load pretrained weights if specified (only patch encoder / pos embed)
        if cfg.encoder_weights_path is not None:
            self._load_encoder_weights(cfg.encoder_weights_path)

        # Freeze encoder if needed
        if cfg.freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            print(
                f"[MomentumAwarePointEncoder] Momentum encoder fully frozen (cfg.freeze_encoder=True)"
            )

        # ViT-style blocks over patch tokens (no robot_state/hand_state tokens).
        token_dim_final = cfg.encoder_feature_dim

        if cfg.vit_use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim_final))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.vit_blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=token_dim_final,
                    num_heads=cfg.vit_num_heads,
                    mlp_ratio=cfg.vit_mlp_ratio,
                    dropout=cfg.vit_dropout,
                    attn_dropout=cfg.vit_attn_dropout,
                    use_flash_attn=cfg.vit_use_flash_attn,
                )
                for _ in range(cfg.vit_depth)
            ]
        )
        self.vit_norm = nn.LayerNorm(token_dim_final)

        # Method: Pool EE flow into patches, encode with MLP, then apply cross attention with point cloud tokens
        if cfg.use_ee_flow_conditioning:
            # MLP encoder: [B, num_patches, 3] -> [B, num_patches, token_dim]
            mlp_layers = []
            input_dim = 3
            for hidden_dim in cfg.ee_flow_mlp_hidden_dims:
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                mlp_layers.append(nn.LayerNorm(hidden_dim))
                mlp_layers.append(nn.GELU())
                input_dim = hidden_dim
            # Final layer: map to token_dim
            mlp_layers.append(nn.Linear(input_dim, token_dim_final))
            mlp_layers.append(nn.LayerNorm(token_dim_final))
            self.ee_flow_mlp_encoder = nn.Sequential(*mlp_layers)
            
            # Cross attention: patch_tokens (Q) attend to ee_flow_tokens (K, V)
            self.ee_flow_cross_attn = nn.MultiheadAttention(
                embed_dim=token_dim_final,
                num_heads=cfg.ee_flow_cross_attn_heads,
                dropout=cfg.ee_flow_cross_attn_dropout,
                batch_first=True,
            )
            self.ee_flow_cross_attn_norm = nn.LayerNorm(token_dim_final)
        else:
            self.ee_flow_mlp_encoder = None
            self.ee_flow_cross_attn = None
            self.ee_flow_cross_attn_norm = None

        # Decoder: predict next frame point cloud positions OR delta pose
        if self.enable_decoder:
            if cfg.predict_delta_pose:
                # Delta pose prediction mode: 9D output (3D translation + 6D rotation)
                # Only uses object tokens, ignores obstacle and EE tokens
                num_object_patches = cfg.num_points_per_object // cfg.patch_size
                delta_pose_input_dim = token_dim_final * num_object_patches
                
                # Build delta pose prediction head
                layers = []
                prev_dim = delta_pose_input_dim
                for hidden_dim in cfg.delta_pose_hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 9))  # 9D output: 3D translation + 6D rotation
                
                self.delta_pose_head = nn.Sequential(*layers)
                
                # Set unpatchify-related attributes to None (not used in delta pose mode)
                self.decoder = None
                self.position_head = None
                self.velocity_head = None
                self.unpatchify_method = None
                self.unpatchify_mlp = None
                self.unpatchify_proj = None
                self.relative_pos_encoder = None
            else:
                # Standard point-wise prediction mode
                # Use unpatchify_point_dim as the point feature dimension
                point_feature_dim = cfg.unpatchify_point_dim
                decoder_input_dim = point_feature_dim
                # This project uses MLP decoder only.
                self.decoder = MLPDecoder(
                    input_dim=decoder_input_dim, hidden_dims=cfg.decoder_hidden_dims
                )

                self.position_head = nn.Linear(point_feature_dim, 3)
                self.velocity_head = nn.Linear(point_feature_dim, 3)

                # Choose unpatchify method
                self.unpatchify_method = cfg.unpatchify_method
                # Unpatchify input dimension is just token_dim_final
                # (EE flow is fused via cross attention, not concatenation)
                unpatchify_input_dim = token_dim_final
                
                if cfg.unpatchify_method == "mlp":
                    self.unpatchify_mlp = self._build_unpatchify_mlp(
                        input_dim=unpatchify_input_dim, patch_size=cfg.patch_size
                    )
                    self.unpatchify_proj = None
                elif cfg.unpatchify_method == "scatter":
                    # For scatter method, project token_dim to point_dim
                    self.unpatchify_proj = nn.Linear(unpatchify_input_dim, point_feature_dim)
                    # Add relative position encoder to distinguish points within a patch
                    self.relative_pos_encoder = nn.Sequential(
                        nn.Linear(
                            3, point_feature_dim // 4
                        ),  # 3D relative position -> feature_dim/4
                        nn.LayerNorm(point_feature_dim // 4),
                        nn.GELU(),
                        nn.Linear(
                            point_feature_dim // 4, point_feature_dim
                        ),  # -> full feature_dim
                    )
                    self.unpatchify_mlp = None
                else:
                    raise ValueError(
                        f"Unknown unpatchify_method: {cfg.unpatchify_method}. Must be 'mlp' or 'scatter'"
                    )
                
                # Set delta_pose_head to None (not used in standard mode)
                self.delta_pose_head = None

        print(f"[MomentumAwarePointEncoder] Architecture:")
        print(
            f"  - Preprocessing: {self.total_num_points} points -> {self.num_patches} patches (patch_size={cfg.patch_size})"
        )
        print(f"    - Patch grouping method: {cfg.patch_grouping_method}")
        if cfg.patch_grouping_method == "spatial_curve":
            print(f"    - Spatial curve type: Morton (Z-order)")
        print(
            f"    - Normalize position: {cfg.normalize_position}, "
            f"Normalize mass: {cfg.normalize_mass}, Normalize velocity: {cfg.normalize_velocity}"
        )
        print(
            f"  - Patch Encoder: PointNet style (per-point MLP + intermediate max-pool)"
        )
        print(
            f"    - Input: patches [patch_size={cfg.patch_size}, {cfg.patch_encoder_point_dim}] (shared for all clouds)"
        )
        print(f"    - Hidden: {cfg.patch_encoder_hidden_dims}")
        print(f"    - Output: {cfg.encoder_feature_dim}")
        print(f"    - Position Embedding: {cfg.pos_embed_type}")
        print(
            f"  - ViT Fusion: depth={cfg.vit_depth}, heads={cfg.vit_num_heads}, dim={token_dim_final}"
        )
        flash_attn_status = "enabled" if cfg.vit_use_flash_attn else "disabled"
        print(
            f"    - CLS token: {bool(cfg.vit_use_cls_token)}; Type embedding: {bool(cfg.use_type_embedding)}; Flash Attention: {flash_attn_status}"
        )
        if cfg.use_ee_flow_conditioning:
            print(f"  - EE Flow Conditioning: enabled (Mean Pooling + MLP + Cross Attention)")
            print(f"    - Mean pooling: -> {cfg.ee_flow_num_patches} patches")
            print(f"    - MLP encoder: 3 -> {cfg.ee_flow_mlp_hidden_dims} -> {token_dim_final}")
            print(f"    - Cross attention heads: {cfg.ee_flow_cross_attn_heads}")
            print(f"    - Fusion: Point cloud tokens (Q) attend to EE flow patch tokens (K, V)")
        
        if self.enable_decoder:
            if cfg.predict_delta_pose:
                print("  - Prediction Mode: Delta Pose (9D: 3D translation + 6D rotation)")
                num_object_patches = cfg.num_points_per_object // cfg.patch_size
                print(f"    - Input: {num_object_patches} object patch tokens (concatenated)")
                print(f"    - MLP: {token_dim_final * num_object_patches} -> {cfg.delta_pose_hidden_dims} -> 9")
                print("    - Note: Obstacle and EE tokens are NOT used for prediction")
            else:
                if cfg.unpatchify_method == "mlp":
                    print(
                        f"  - Unpatchify: MLP -> point_dim {cfg.unpatchify_point_dim} (hidden_dims: {cfg.unpatchify_hidden_dims})"
                    )
                elif cfg.unpatchify_method == "scatter":
                    print(
                        f"  - Unpatchify: Scatter -> point_dim {cfg.unpatchify_point_dim} (Linear projection + Relative position encoding)"
                    )
                if self.decoder is not None:
                    print("  - Decoder: MLP")
                    decoder_input_dim = cfg.unpatchify_point_dim
                    print(
                        f"    - MLP: {decoder_input_dim} -> {cfg.decoder_hidden_dims or '[]'} -> {decoder_input_dim}"
                    )
        else:
            print("  - Decoder disabled (encoder-only mode)")

    def _load_encoder_weights(self, weights_path: str):
        """Load pretrained encoder weights."""
        print(
            f"[MomentumAwarePointEncoder] Loading encoder weights from: {weights_path}"
        )
        state_dict = torch.load(weights_path, map_location="cpu")

        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Secondary encoder removed; skip loading pc_encoder weights
        print("  [Warning] pc_encoder removed; skip loading secondary encoder weights.")

    @staticmethod
    def create_point_types_from_structure(
        batch_size: int,
        num_object_points: int = 512,
        num_obstacle_points: int = 512,
        num_robot_points: int = 256,
        device: torch.device = None,
    ) -> torch.Tensor:
        total_points = num_object_points + num_obstacle_points + num_robot_points
        point_types = torch.zeros(
            batch_size, total_points, dtype=torch.long, device=device
        )

        # Set obstacle points (type 1)
        if num_obstacle_points > 0:
            point_types[
                :, num_object_points : num_object_points + num_obstacle_points
            ] = 1

        # Set robot points (type 2)
        if num_robot_points > 0:
            point_types[:, num_object_points + num_obstacle_points :] = 2

        return point_types

    def _build_unpatchify_mlp(self, input_dim: int, patch_size: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.cfg.unpatchify_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        output_dim = patch_size * self.cfg.unpatchify_point_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def encode_tokens(
        self,
        pointclouds: torch.Tensor,
        point_types: Optional[torch.Tensor] = None,  # [B, num_points] - type indices
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Encode unified point clouds into fused tokens (for RL use).

        Args:
            pointclouds: [B, num_points, 7] unified point clouds (x, y, z, mass, vx, vy, vz)
            point_types: [B, num_points] - type indices (0: target, 1: obstacle, 2: robot)
            return_features: Whether to return intermediate features dictionary

        Returns:
            fused_tokens: [B, num_patches, token_dim] - all tokens after transformer fusion
            features_dict: Optional dict containing:
                - tokens: [B, num_patches, token_dim] fused tokens
                - patch_centers: [B, num_patches, 3] patch center positions
                - patch_indices: [B, num_patches, patch_size] indices mapping points to patches
        """
        B, num_points, _ = pointclouds.shape

        # 1. 预处理：normalize + group into patches
        group_counts = torch.tensor(
            [
                self.cfg.num_points_per_object,
                self.cfg.num_obstacles * self.cfg.num_points_per_object,
                self.cfg.num_ee_points,
            ],
            device=pointclouds.device,
        )
        patches, patch_centers, patch_indices = self.preprocessor(
            pointclouds, update_stats=self.training, group_counts=group_counts
        )  # patches: [B, num_patches, patch_size, 7], patch_centers: [B, num_patches, 3]

        # 2. Center patches (subtract patch center from point positions)
        patches = self._center_patches(patches, patch_centers)

        # 3. Patch encoding: encode each patch to patch embedding
        patch_emb = self.patch_encoder(patches)  # [B, num_patches, encoder_feature_dim]

        # 4. Add position embedding
        if self.pos_embed is not None:
            patch_emb = patch_emb + self.pos_embed(patch_centers)

        # 5. Add type embedding (if point_types provided)
        x = patch_emb
        if self.type_embeddings is not None and point_types is not None:
            B, num_patches, patch_size = patch_indices.shape
            center_point_indices = patch_indices[
                :, :, 0
            ]  # [B, num_patches] - indices of center points
            batch_indices = (
                torch.arange(B, device=patch_indices.device)
                .view(B, 1)
                .expand(B, num_patches)
            )
            patch_type_ids = point_types[
                batch_indices, center_point_indices
            ]  # [B, num_patches]
            x = x + self.type_embeddings[patch_type_ids]

        # 6. Add CLS token if enabled
        if self.cls_token is not None:
            x = torch.cat(
                [self.cls_token.expand(B, -1, -1), x], dim=1
            )  # [B, 1+num_patches, token_dim_final]

        # 7. Pass through ViT blocks
        for blk in self.vit_blocks:
            x = blk(x)
        x = self.vit_norm(x)

        # 8. Strip CLS token if present
        fused_tokens = (
            x[:, 1:, :] if self.cls_token is not None else x
        )  # [B, num_patches, token_dim_final]

        features_dict = None
        if return_features:
            features_dict = {
                "tokens": fused_tokens,
                "patch_centers": patch_centers,
                "patch_indices": patch_indices,
            }

        return fused_tokens, features_dict

    def _center_patches(
        self, patches: torch.Tensor, patch_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Center patches by subtracting patch center from point positions.

        Args:
            patches: [B, num_patches, patch_size, 7] patches
            patch_centers: [B, num_patches, 3] patch center positions

        Returns:
            patches_centered: [B, num_patches, patch_size, 7] centered patches
        """
        patch_positions = patches[..., :3]  # [B, num_patches, patch_size, 3]
        patch_features = patches[..., 3:]  # [B, num_patches, patch_size, 4]

        # Center positions relative to patch center
        patch_positions_centered = (
            patch_positions - patch_centers[..., None, :]
        )  # [B, num_patches, patch_size, 3]

        # Combine centered positions with original features
        patches_centered = torch.cat(
            [patch_positions_centered, patch_features], dim=-1
        )  # [B, num_patches, patch_size, 7]

        return patches_centered

    def _extract_patch_level_state_unified(
        self,
        pointclouds: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract patch-level state (average position and velocity) from a unified point cloud.

        Args:
            pointclouds:   [B, N_total, 7] unified point clouds (object + obstacles + ee)
            patch_indices: [B, num_objects, num_patches, patch_size] unified indices into pointclouds

        Returns:
            patch_state:   [B, num_objects, num_patches, 6] patch-level state (pos + vel)
        """
        B, num_objects, num_patches, patch_size = patch_indices.shape
        device = pointclouds.device

        # Build batch index for advanced indexing
        batch_indices = (
            torch.arange(B, device=device)
            .view(B, 1, 1, 1)
            .expand(B, num_objects, num_patches, patch_size)
        )

        # Gather points for each patch from unified cloud
        # pointclouds: [B, N_total, 7], indices: [B, num_objects, num_patches, patch_size]
        patch_points = pointclouds[
            batch_indices, patch_indices, :
        ]  # [B, num_objects, num_patches, patch_size, 7]

        # Average over patch points
        patch_positions = patch_points[..., :3].mean(
            dim=3
        )  # [B, num_objects, num_patches, 3]
        patch_velocities = patch_points[..., 4:7].mean(
            dim=3
        )  # [B, num_objects, num_patches, 3]
        patch_state = torch.cat(
            [patch_positions, patch_velocities], dim=-1
        )  # [B, num_objects, num_patches, 6]
        return patch_state

    def forward(
        self,
        pointclouds: torch.Tensor,  # [B, num_points, 7] (Unified)
        point_types: Optional[torch.Tensor] = None,  # [B, num_points] - type indices
        ee_flow: Optional[torch.Tensor] = None,  # [B, num_ee_points, 3] - optional EE flow input
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next frame point clouds or delta pose if decoder is enabled.

        Args:
            pointclouds: [B, num_points, 7] unified point cloud (x, y, z, mass, vx, vy, vz)
            point_types: [B, num_points] - point type indices (0: target, 1: obstacle, 2: robot)
            ee_flow: [B, num_ee_points, 3] - optional end-effector flow (next_ee_pc - current_ee_pc)

        Returns:
            If predict_delta_pose is True:
                prediction: [B, 9] predicted delta pose (3D translation + 6D rotation)
            Else:
                prediction: [B, num_unpatchified_points, 6] predicted next point cloud (pos + vel)
                          where num_unpatchified_points = num_patches * patch_size
                          Note: num_unpatchified_points may differ from num_points if num_points is not divisible by patch_size
        """
        B, num_points, _ = pointclouds.shape

        # Encode point clouds
        fused_tokens, features_dict = self.encode_tokens(
            pointclouds, point_types, return_features=True
        )
        
        # Encode EE flow if provided and EE flow conditioning is enabled
        # Method: Pool EE flow into patches, encode with MLP, then apply cross attention
        if self.cfg.use_ee_flow_conditioning and ee_flow is not None:
            if self.ee_flow_mlp_encoder is None or self.ee_flow_cross_attn is None:
                raise RuntimeError("EE flow cross attention is enabled but modules are not initialized")
            
            B_ee, N_ee, _ = ee_flow.shape  # [B, num_ee_points, 3]
            
            # Mean pooling: [B, num_ee_points, 3] -> [B, num_patches, 3]
            # Use AdaptiveAvgPool1d for mean pooling: transpose to [B, 3, num_ee_points], pool to [B, 3, num_patches], transpose back
            ee_flow_pooled = F.adaptive_avg_pool1d(
                ee_flow.transpose(1, 2),  # [B, 3, num_ee_points]
                self.cfg.ee_flow_num_patches
            ).transpose(1, 2)  # [B, num_patches, 3]
            
            # Encode pooled EE flow with MLP: [B, num_patches, 3] -> [B, num_patches, token_dim]
            ee_patch_tokens = self.ee_flow_mlp_encoder(ee_flow_pooled)  # [B, num_patches, token_dim]
            
            # Apply cross attention: fused_tokens (Q) attend to ee_patch_tokens (K, V)
            # Note: MultiheadAttention expects (batch, seq_len, embed_dim)
            fused_tokens_attn, _ = self.ee_flow_cross_attn(
                query=fused_tokens,          # [B, num_patches, token_dim]
                key=ee_patch_tokens,         # [B, num_patches, token_dim]
                value=ee_patch_tokens,       # [B, num_patches, token_dim]
            )  # [B, num_patches, token_dim]
            
            # Residual connection + layer norm
            fused_tokens = fused_tokens + fused_tokens_attn
            fused_tokens = self.ee_flow_cross_attn_norm(fused_tokens)
        
        # Delta pose prediction mode
        if self.cfg.predict_delta_pose:
            # Extract object tokens only (first num_object_patches tokens)
            num_object_patches = self.cfg.num_points_per_object // self.cfg.patch_size
            object_tokens = fused_tokens[:, :num_object_patches, :]  # [B, num_obj_patches, token_dim]
            
            # Concatenate all object tokens
            object_features = object_tokens.reshape(B, -1)  # [B, num_obj_patches * token_dim]
            
            # Predict delta pose (9D: 3D translation + 6D rotation)
            delta_pose = self.delta_pose_head(object_features)  # [B, 9]
            
            return {"prediction": delta_pose}


        # Unpatchify: convert patch tokens to point features
        if self.unpatchify_method == "mlp":
            point_features_patched = self.unpatchify_mlp(
                fused_tokens
            )  # [B, num_patches, patch_size * unpatchify_point_dim]

            # Reshape to [B, num_patches * patch_size, unpatchify_point_dim]
            _, num_patches, _ = point_features_patched.shape
            patch_size = self.cfg.patch_size
            num_unpatchified_points = num_patches * patch_size
            point_features_flat = point_features_patched.view(
                B, num_unpatchified_points, self.cfg.unpatchify_point_dim
            )
        elif self.unpatchify_method == "scatter":
            # Project tokens to point feature dimension
            patch_features = self.unpatchify_proj(
                fused_tokens
            )  # [B, num_patches, unpatchify_point_dim]

            # Get patch_indices and patch_centers from features_dict
            patch_indices = features_dict[
                "patch_indices"
            ]  # [B, num_patches, patch_size]
            patch_centers = features_dict["patch_centers"]  # [B, num_patches, 3]

            # Scatter patch features to points
            # patch_indices: [B, num_patches, patch_size] - each patch has patch_size point indices
            B, num_patches, patch_size = patch_indices.shape
            num_points = pointclouds.shape[1]

            # Get point positions from input pointcloud
            point_positions = pointclouds[..., :3]  # [B, num_points, 3]

            # Compute relative positions for each point in each patch
            batch_indices = (
                torch.arange(B, device=patch_indices.device)
                .view(B, 1, 1)
                .expand(B, num_patches, patch_size)
            )  # [B, num_patches, patch_size]

            # Get positions of points in each patch: [B, num_patches, patch_size, 3]
            patch_point_positions = point_positions[batch_indices, patch_indices, :]

            # Compute relative positions: point_pos - patch_center
            patch_centers_expanded = patch_centers.unsqueeze(
                2
            )  # [B, num_patches, 1, 3]
            relative_positions = (
                patch_point_positions - patch_centers_expanded
            )  # [B, num_patches, patch_size, 3]

            # Encode relative positions
            relative_pos_encoded = self.relative_pos_encoder(
                relative_positions
            )  # [B, num_patches, patch_size, unpatchify_point_dim]

            # Combine patch features with relative position encoding
            # Expand patch_features to match patch_size dimension
            patch_features_expanded = patch_features.unsqueeze(2).expand(
                B, num_patches, patch_size, self.cfg.unpatchify_point_dim
            )  # [B, num_patches, patch_size, unpatchify_point_dim]

            # Add relative position encoding to patch features
            feature_with_pos = (
                patch_features_expanded + relative_pos_encoded
            )  # [B, num_patches, patch_size, unpatchify_point_dim]

            # Flatten for scatter
            batch_flat = batch_indices.flatten()
            point_flat = patch_indices.flatten()  # [B * num_patches * patch_size]
            feature_flat = feature_with_pos.flatten(
                0, 2
            )  # [B * num_patches * patch_size, unpatchify_point_dim]

            # Scatter: aggregate features for each point (use mean for points that appear in multiple patches)
            point_features_flat = torch.zeros(
                B * num_points,
                self.cfg.unpatchify_point_dim,
                device=pointclouds.device,
                dtype=patch_features.dtype,
            )
            index = (
                batch_flat * num_points + point_flat
            )  # [B * num_patches * patch_size]

            # Use scatter_add and count, then divide by count to get mean
            point_features_flat.scatter_add_(
                0,
                index.unsqueeze(1).expand(-1, self.cfg.unpatchify_point_dim),
                feature_flat,
            )

            # Count how many patches each point belongs to
            count = torch.zeros(
                B * num_points, device=pointclouds.device, dtype=torch.float32
            )
            count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
            count = torch.clamp(count, min=1.0)  # Avoid division by zero

            # Normalize by count to get mean
            point_features_flat = point_features_flat / count.unsqueeze(1)

            # Reshape back to [B, num_points, unpatchify_point_dim]
            point_features_flat = point_features_flat.view(
                B, num_points, self.cfg.unpatchify_point_dim
            )
            num_unpatchified_points = num_points
        else:
            raise ValueError(
                f"Unknown unpatchify_method: {self.unpatchify_method}. Must be 'mlp' or 'scatter'"
            )

        point_features_flat = self.decoder(point_features_flat)

        # Predict position and velocity
        position_pred = self.position_head(
            point_features_flat
        )  # [B, num_unpatchified_points, 3]
        velocity_pred = self.velocity_head(
            point_features_flat
        )  # [B, num_unpatchified_points, 3]

        # Concatenate position and velocity predictions
        prediction = torch.cat(
            [position_pred, velocity_pred], dim=-1
        )  # [B, num_unpatchified_points, 6]

        results = {
            "prediction": prediction,
        }

        return results

    def load_state_dict(self, state_dict, strict: bool = False):
        """
        Override to ignore decoder weights when decoder is disabled or when switching modes.
        """
        decoder_prefixes = []
        
        if not self.enable_decoder:
            # Decoder disabled: ignore all decoder-related weights
            decoder_prefixes = [
                "decoder.",
                "position_head.",
                "velocity_head.",
                "unpatchify_proj.",
                "unpatchify_mlp.",
                "relative_pos_encoder.",
                "delta_pose_head.",
            ]
        elif self.cfg.predict_delta_pose:
            # Delta pose mode: ignore point-wise prediction heads
            decoder_prefixes = [
                "decoder.",
                "position_head.",
                "velocity_head.",
                "unpatchify_proj.",
                "unpatchify_mlp.",
                "relative_pos_encoder.",
            ]
        else:
            # Point-wise mode: ignore delta pose head
            decoder_prefixes = [
                "delta_pose_head.",
            ]
        
        filtered = {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(prefix) for prefix in decoder_prefixes)
        }
        
        return super().load_state_dict(filtered, strict=strict)
    
    def encode(
        self,
        pointclouds: torch.Tensor,
        point_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode point clouds to fused tokens (for external use, e.g., delta pose predictor).
        
        Args:
            pointclouds: [B, num_points, 7] unified point cloud
            point_types: [B, num_points] - point type indices
            
        Returns:
            fused_tokens: [B, num_patches, token_dim] encoded tokens
        """
        fused_tokens, _ = self.encode_tokens(pointclouds, point_types, return_features=False)
        return fused_tokens

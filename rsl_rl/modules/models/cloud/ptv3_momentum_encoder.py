# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PTV3-based Momentum-Aware Point Cloud Encoder (patch-level next-frame prediction)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from rsl_rl.modules.util.config import ConfigBase
from rsl_rl.PointTransformerV3.model import PointTransformerV3, Point


@dataclass
class PTV3MomentumAwarePointEncoderConfig(ConfigBase):
    """Configuration for PTV3-based Momentum-Aware Point Cloud Encoder."""

    # PTV3 Backbone settings
    in_channels: int = 7  # (x, y, z, mass, vx, vy, vz)
    stride: tuple[int, ...] = (2, 2, 2)
    # Small backbone (ends at 128) for faster RL. Re-train encoder after changing these.
    enc_depths: tuple[int, ...] = (1, 1, 2, 1)
    enc_channels: tuple[int, ...] = (16, 32, 64, 128)
    enc_num_head: tuple[int, ...] = (2, 4, 8, 8)
    enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024)
    # Decoder settings
    dec_depths: tuple[int, ...] = (1, 1, 1)
    dec_channels: tuple[int, ...] = (32, 64, 128)
    dec_num_head: tuple[int, ...] = (2, 4, 8)
    dec_patch_size: tuple[int, ...] = (1024, 1024, 1024)

    # Partial loading settings
    max_dec_stage: int | None = None

    grid_size: float = 0.02
    enable_flash: bool = True

    # Prediction head
    predict_delta: bool = False
    predict_uncertainty: bool = (
        False  # Whether to predict aleatoric uncertainty (log-variance)
    )

    # Type embedding settings
    use_type_embedding: bool = True  # Whether to use learnable type embeddings
    num_point_types: int = 3  # 0: target object, 1: obstacle, 2: robot arm

    # Normalization settings
    normalize_position: bool = False  # Whether to normalize position
    normalize_velocity: bool = True  # Whether to normalize velocity
    normalize_mass: bool = True  # Whether to normalize mass
    position_momentum: float = 0.01  # Momentum for running statistics update
    velocity_momentum: float = 0.01  # Momentum for running statistics update
    mass_momentum: float = 0.01  # Momentum for running statistics update


class PTV3MomentumAwarePointEncoder(nn.Module):
    """
    PTV3-based encoder that processes unified point clouds.
    - Pre-training: Uses Encoder+Decoder (U-Net) for next-frame point prediction.
    - RL: Uses Encoder only to extract sparse patch tokens.
    """

    def __init__(self, cfg: PTV3MomentumAwarePointEncoderConfig):
        super().__init__()
        self.cfg = cfg

        dec_depths = cfg.dec_depths
        dec_channels = cfg.dec_channels
        dec_num_head = cfg.dec_num_head
        dec_patch_size = cfg.dec_patch_size

        # If max_dec_stage = 0, use cls_mode to skip decoder entirely (saves memory)
        use_cls_mode = cfg.max_dec_stage is not None and cfg.max_dec_stage == 0

        if cfg.max_dec_stage is not None:
            if use_cls_mode:
                print(
                    f"[PTV3MomentumAwarePointEncoder] max_dec_stage=0: Using cls_mode=True (encoder only, no decoder)"
                )
            else:
                print(
                    f"[PTV3MomentumAwarePointEncoder] Partial loading: Decoder up to stage {cfg.max_dec_stage}"
                )
                print(
                    f"  Note: All decoder stages will be built, but only up to stage {cfg.max_dec_stage} will be used in encode_tokens()"
                )

        self.backbone = PointTransformerV3(
            in_channels=cfg.in_channels,
            enc_depths=cfg.enc_depths,
            enc_channels=cfg.enc_channels,
            enc_num_head=cfg.enc_num_head,
            enc_patch_size=cfg.enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            stride=cfg.stride,
            enable_flash=cfg.enable_flash,
            cls_mode=use_cls_mode,
        )

        # Store actual depths for reference
        self.actual_enc_stages = len(cfg.enc_depths)
        self.actual_dec_stages = len(dec_depths)

        # Learnable type embeddings for distinguishing point cloud types
        # 0: target object, 1: obstacle, 2: robot arm
        if cfg.use_type_embedding:
            embed_dim = cfg.enc_channels[0]  # Embedding dimension after stem
            self.type_embeddings = nn.Parameter(
                torch.zeros(cfg.num_point_types, embed_dim)
            )
            # Initialize with small random values
            nn.init.normal_(self.type_embeddings, std=0.02)
        else:
            self.type_embeddings = None

        # Separate heads for position and velocity prediction
        self.position_head = nn.Sequential(
            nn.Linear(cfg.dec_channels[0], cfg.dec_channels[0] // 2),
            nn.GELU(),
            nn.Linear(cfg.dec_channels[0] // 2, 3),  # (x, y, z)
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(cfg.dec_channels[0], cfg.dec_channels[0] // 2),
            nn.GELU(),
            nn.Linear(cfg.dec_channels[0] // 2, 3),  # (vx, vy, vz)
        )

        # Uncertainty heads (predict log-variance for aleatoric uncertainty)
        if cfg.predict_uncertainty:
            self.velocity_uncertainty_head = nn.Sequential(
                nn.Linear(cfg.dec_channels[0], cfg.dec_channels[0] // 2),
                nn.GELU(),
                nn.Linear(cfg.dec_channels[0] // 2, 1),  # log-variance for velocity
            )
        else:
            self.velocity_uncertainty_head = None

        # Running statistics for normalization
        self.register_buffer("position_mean", torch.zeros(3))
        self.register_buffer("position_std", torch.ones(3))
        self.register_buffer(
            "_position_initialized", torch.tensor(False, dtype=torch.bool)
        )

        self.register_buffer("velocity_mean", torch.zeros(3))
        self.register_buffer("velocity_std", torch.ones(3))
        self.register_buffer(
            "_velocity_initialized", torch.tensor(False, dtype=torch.bool)
        )

        self.register_buffer("mass_mean", torch.zeros(1))
        self.register_buffer("mass_std", torch.ones(1))
        self.register_buffer("_mass_initialized", torch.tensor(False, dtype=torch.bool))

        if cfg.normalize_position:
            print(f"[PTV3MomentumAwarePointEncoder] Position normalization ENABLED")
            print(f"  - Momentum: {cfg.position_momentum}")
        else:
            print(f"[PTV3MomentumAwarePointEncoder] Position normalization DISABLED")

        if cfg.normalize_velocity:
            print(f"[PTV3MomentumAwarePointEncoder] Velocity normalization ENABLED")
            print(f"  - Momentum: {cfg.velocity_momentum}")
        else:
            print(f"[PTV3MomentumAwarePointEncoder] Velocity normalization DISABLED")

        if cfg.normalize_mass:
            print(f"[PTV3MomentumAwarePointEncoder] Mass normalization ENABLED")
            print(f"  - Momentum: {cfg.mass_momentum}")
        else:
            print(f"[PTV3MomentumAwarePointEncoder] Mass normalization DISABLED")

    @torch.no_grad()
    def update_position_statistics(self, position: torch.Tensor):
        if not self.training or not self.cfg.normalize_position:
            return

        # Flatten position
        position_flat = position.reshape(-1, 3)  # [B*N, 3]

        # Filter out near-zero positions (padding or invalid points)
        valid_mask = position_flat.norm(dim=-1) > 1e-6

        if valid_mask.sum() > 0:
            valid_position = position_flat[valid_mask]
            batch_mean = valid_position.mean(dim=0)
            batch_std = valid_position.std(dim=0).clamp(min=1e-6)

            # Update running statistics with momentum
            if not self._position_initialized:
                # First batch: initialize
                self.position_mean.copy_(batch_mean)
                self.position_std.copy_(batch_std)
                self._position_initialized.fill_(True)
            else:
                momentum = self.cfg.position_momentum
                self.position_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.position_std.mul_(1 - momentum).add_(batch_std, alpha=momentum)

    @torch.no_grad()
    def update_velocity_statistics(self, velocity: torch.Tensor):
        """
        Update running statistics for velocity normalization.

        Args:
            velocity: [B, N, 3] velocity values
        """
        if not self.training or not self.cfg.normalize_velocity:
            return

        # Flatten velocity
        velocity_flat = velocity.reshape(-1, 3)  # [B*N, 3]

        # Filter out near-zero velocities (padding or static points)
        valid_mask = velocity_flat.norm(dim=-1) > 1e-6

        if valid_mask.sum() > 0:
            valid_velocity = velocity_flat[valid_mask]
            batch_mean = valid_velocity.mean(dim=0)
            batch_std = valid_velocity.std(dim=0).clamp(min=1e-6)

            # Update running statistics with momentum
            if not self._velocity_initialized:
                # First batch: initialize
                self.velocity_mean.copy_(batch_mean)
                self.velocity_std.copy_(batch_std)
                self._velocity_initialized.fill_(True)
            else:
                momentum = self.cfg.velocity_momentum
                self.velocity_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.velocity_std.mul_(1 - momentum).add_(batch_std, alpha=momentum)

    @torch.no_grad()
    def update_mass_statistics(self, mass: torch.Tensor):
        """
        Update running statistics for mass normalization.

        Args:
            mass: [B, N, 1] or [B, N] mass values
        """
        if not self.training or not self.cfg.normalize_mass:
            return

        # Flatten mass
        if mass.dim() == 3:
            mass_flat = mass.reshape(-1)  # [B*N]
        else:
            mass_flat = mass.reshape(-1)  # [B*N]

        # Filter out near-zero masses (padding or invalid points)
        valid_mask = mass_flat.abs() > 1e-6

        if valid_mask.sum() > 0:
            valid_mass = mass_flat[valid_mask]
            batch_mean = valid_mass.mean()
            batch_std = valid_mass.std().clamp(min=1e-6)

            # Update running statistics with momentum
            if not self._mass_initialized:
                # First batch: initialize
                self.mass_mean.fill_(batch_mean.item())
                self.mass_std.fill_(batch_std.item())
                self._mass_initialized.fill_(True)
            else:
                momentum = self.cfg.mass_momentum
                self.mass_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.mass_std.mul_(1 - momentum).add_(batch_std, alpha=momentum)

    def normalize_position(self, position: torch.Tensor) -> torch.Tensor:
        """
        Normalize position using running statistics.

        Args:
            position: [B, N, 3] or [B*N, 3] position values

        Returns:
            normalized: Same shape as input, normalized position
        """
        if not self.cfg.normalize_position or not self._position_initialized:
            return position

        # Normalize: (p - mean) / std
        return (position - self.position_mean) / (self.position_std + 1e-8)

    def denormalize_position(self, position_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize position back to original scale.

        Args:
            position_normalized: [B, N, 3] or [B*N, 3] normalized position

        Returns:
            position: Same shape as input, denormalized position
        """
        if not self.cfg.normalize_position or not self._position_initialized:
            return position_normalized

        # Denormalize: p = p_norm * std + mean
        return position_normalized * self.position_std + self.position_mean

    def normalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_velocity or not self._velocity_initialized:
            return velocity

        return (velocity - self.velocity_mean) / (self.velocity_std + 1e-8)

    def denormalize_velocity(self, velocity_normalized: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_velocity or not self._velocity_initialized:
            return velocity_normalized

        return velocity_normalized * self.velocity_std + self.velocity_mean

    def normalize_mass(self, mass: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_mass or not self._mass_initialized:
            return mass

        return (mass - self.mass_mean) / (self.mass_std + 1e-8)

    def denormalize_mass(self, mass_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize mass back to original scale.

        Args:
            mass_normalized: [B, N, 1], [B, N], or [B*N] normalized mass

        Returns:
            mass: Same shape as input, denormalized mass
        """
        if not self.cfg.normalize_mass or not self._mass_initialized:
            return mass_normalized

        # Denormalize: m = m_norm * std + mean
        return mass_normalized * self.mass_std + self.mass_mean

    @staticmethod
    def create_point_types_from_structure(
        batch_size: int,
        num_object_points: int,
        num_obstacle_points: int = 0,
        num_robot_points: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Helper function to create point_types tensor from point cloud structure.

        Args:
            batch_size: Batch size
            num_object_points: Number of points per batch for target object
            num_obstacle_points: Number of points per batch for obstacles (total)
            num_robot_points: Number of points per batch for robot arm
            device: Device for the tensor

        Returns:
            point_types: [B, num_points] where 0=target, 1=obstacle, 2=robot
        """
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

    def _prepare_ptv3_data(
        self,
        pointclouds: torch.Tensor,
        point_types: Optional[torch.Tensor] = None,
    ):
        """
        Prepares input dictionary for PTV3.

        Args:
            pointclouds: [B, num_points, 7] - unified point cloud (x, y, z, mass, vx, vy, vz)
            point_types: [B, num_points] - point type indices (0: target, 1: obstacle, 2: robot)
                        If None, will infer from point cloud structure or use default

        Velocity is normalized using running statistics.
        """
        B, num_pts, _ = pointclouds.shape
        device = pointclouds.device

        # Update statistics (only during training)
        if self.training:
            if self.cfg.normalize_position:
                position = pointclouds[:, :, :3]  # [B, N, 3]
                self.update_position_statistics(position)
            if self.cfg.normalize_mass:
                mass = pointclouds[:, :, 3:4]  # [B, N, 1]
                self.update_mass_statistics(mass)
            if self.cfg.normalize_velocity:
                velocity = pointclouds[:, :, 4:7]  # [B, N, 3]
                self.update_velocity_statistics(velocity)

        # Normalize position, mass and velocity
        need_position_norm = self.cfg.normalize_position and bool(
            self._position_initialized
        )
        need_mass_norm = self.cfg.normalize_mass and bool(self._mass_initialized)
        need_velocity_norm = self.cfg.normalize_velocity and bool(
            self._velocity_initialized
        )
        need_norm = need_position_norm or need_mass_norm or need_velocity_norm

        if need_norm:
            pointclouds_normalized = pointclouds.clone()
            if need_position_norm:
                pointclouds_normalized[:, :, :3] = self.normalize_position(
                    pointclouds[:, :, :3]
                )
            if need_mass_norm:
                pointclouds_normalized[:, :, 3:4] = self.normalize_mass(
                    pointclouds[:, :, 3:4]
                )
            if need_velocity_norm:
                pointclouds_normalized[:, :, 4:7] = self.normalize_velocity(
                    pointclouds[:, :, 4:7]
                )
        else:
            pointclouds_normalized = pointclouds

        # Flatten batch for PTV3
        flat_pc = pointclouds_normalized.view(-1, 7)  # [B*N, 7]
        coord = flat_pc[:, :3]
        feat = flat_pc

        # Consistent batch/offset generation
        offset = torch.arange(1, B + 1, device=device) * num_pts

        data_dict = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": self.cfg.grid_size,
        }

        # Store point types if provided
        if point_types is not None:
            data_dict["point_types"] = point_types.view(-1)  # [B*N]

        return data_dict

    def encode_tokens(
        self,
        pointclouds: torch.Tensor,  # [B, num_points, 7] (Unified)
        point_types: Optional[torch.Tensor] = None,  # [B, num_points] - type indices
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        RL interface: Returns sparse patch tokens from the Encoder part.

        Args:
            pointclouds: [B, num_points, 7] - unified point cloud
            point_types: [B, num_points] - point type indices (0: target, 1: obstacle, 2: robot)
        """
        B, num_pts, _ = pointclouds.shape
        data_dict = self._prepare_ptv3_data(pointclouds, point_types)

        # 1. Serialization & Sparsify
        point = Point(data_dict)
        point.serialization(
            order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders
        )
        point.sparsify()

        # 2. Embedding
        point = self.backbone.embedding(point)

        # 2.5. Add type embeddings if enabled
        if self.cfg.use_type_embedding and self.type_embeddings is not None:
            # Get point types (either from data_dict or infer from structure)
            flat_point_types = point.point_types  # [B*N]

            # Add type embeddings to features
            type_emb = self.type_embeddings[flat_point_types]  # [B*N, embed_dim]
            point.feat = point.feat + type_emb

            # Update sparse_conv_feat if it exists
            if "sparse_conv_feat" in point.keys():
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                    point.feat
                )

        # 3. Encoder
        point = self.backbone.enc(point)

        # 4. Decoder (if needed)
        if not self.backbone.cls_mode:
            if self.cfg.max_dec_stage is not None:
                deepest_stage = self.actual_dec_stages - 1
                target_stage = self.cfg.max_dec_stage
                end_idx = target_stage - 1 if target_stage > 0 else -1
                for s in range(deepest_stage, end_idx, -1):
                    stage_module = getattr(self.backbone.dec, f"dec{s}")
                    point = stage_module(point)
            else:
                point = self.backbone.dec(point)

        patch_tokens = point.feat
        patch_coords = point.coord
        patch_batch = point.batch

        # 5. Pad to [B, Max_Patches, D] for RL
        # Group tokens by batch index
        max_patches = 0
        batch_tokens = []
        batch_coords = []

        # Simple loop for collating (could be optimized with scatter/gather but loop is robust)
        for b in range(B):
            mask = patch_batch == b
            b_tokens = patch_tokens[mask]
            b_coords = patch_coords[mask]
            batch_tokens.append(b_tokens)
            batch_coords.append(b_coords)
            max_patches = max(max_patches, b_tokens.shape[0])

        device = pointclouds.device
        padded_tokens = torch.zeros(
            B, max_patches, patch_tokens.shape[-1], device=device
        )
        attn_mask = torch.ones(
            B, max_patches, dtype=torch.bool, device=device
        )  # True=Masked/Padding

        # Padded coords & object_id (dummy for now as we don't have explicit obj IDs in unified)
        padded_coords = torch.zeros(B, max_patches, 3, device=device)
        padded_ids = torch.zeros(B, max_patches, dtype=torch.long, device=device)

        for b in range(B):
            n = batch_tokens[b].shape[0]
            if n > 0:
                padded_tokens[b, :n] = batch_tokens[b]
                padded_coords[b, :n] = batch_coords[b]
                attn_mask[b, :n] = False  # Active tokens

        # Metadata for RL (mask, coords, etc.)
        metadata = {
            "mask": attn_mask,
            "patch_coords": padded_coords,
            "patch_obj_id": padded_ids,
        }

        if return_features:
            return padded_tokens, metadata
        return padded_tokens

    def forward(
        self,
        pointclouds: torch.Tensor,  # [B, num_points, 7] (Unified)
        point_types: Optional[torch.Tensor] = None,  # [B, num_points] - type indices
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-training interface: Returns dense per-point predictions.

        Args:
            pointclouds: [B, num_points, 7] - unified point cloud
            point_types: [B, num_points] - point type indices (0: target, 1: obstacle, 2: robot)
        """
        B, num_pts, _ = pointclouds.shape
        data_dict = self._prepare_ptv3_data(pointclouds, point_types)

        # Prepare point and add type embeddings before running backbone
        point = Point(data_dict)
        point.serialization(
            order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders
        )
        point.sparsify()

        # Embedding
        point = self.backbone.embedding(point)

        # Add type embeddings if enabled
        if self.cfg.use_type_embedding and self.type_embeddings is not None:
            if "point_types" in point.keys():
                flat_point_types = point.point_types
            else:
                flat_point_types = torch.zeros(
                    point.feat.shape[0], device=point.feat.device, dtype=torch.long
                )

            type_emb = self.type_embeddings[flat_point_types]
            point.feat = point.feat + type_emb

            if "sparse_conv_feat" in point.keys():
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                    point.feat
                )

        # Run PTV3 (Enc + Dec)
        # point.feat will be [B*N, dec_channels[0]] (Dense output matching input resolution)
        point_out = self.backbone.enc(point)
        if not self.backbone.cls_mode:
            point_out = self.backbone.dec(point_out)

        # Predict delta/state using separate heads
        dense_feat = point_out.feat  # [B*N, D]
        position_pred_flat = self.position_head(dense_feat)  # [B*N, 3]
        velocity_pred_flat = self.velocity_head(dense_feat)  # [B*N, 3]

        # Concatenate position and velocity predictions
        prediction_flat = torch.cat(
            [position_pred_flat, velocity_pred_flat], dim=-1
        )  # [B*N, 6]

        # Reshape back to [B, N, 6]
        prediction = prediction_flat.view(B, num_pts, 6)

        # Predict uncertainty (log-variance) if enabled
        result = {"prediction": prediction}

        if self.cfg.predict_uncertainty:
            velocity_logvar_flat = self.velocity_uncertainty_head(
                dense_feat
            )  # [B*N, 1]
            velocity_logvar = velocity_logvar_flat.view(B, num_pts)
            result["velocity_logvar"] = velocity_logvar

        return result

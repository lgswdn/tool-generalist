# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic network using PTV3MomentumAwarePointEncoder with cross-attention architecture."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.models.cloud.ptv3_momentum_encoder import (
    PTV3MomentumAwarePointEncoder,
    PTV3MomentumAwarePointEncoderConfig,
)
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet


class ActorCriticPTV3Momentum(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        point_dim: int = 7,
        num_points: int = 512,
        num_ee_points: int = 256,
        num_obstacles: int = 0,
        ptv3_cfg: Optional[Dict[str, Any]] = None,
        ptv3_ckpt: Optional[str] = None,
        freeze_ptv3: bool = True,
        encoder_strict_load: bool = False,
        max_dec_stage: Optional[int] = None,
        # Cross-attention settings
        use_learnable_query_tokens: bool = True,  # If True, use learnable query tokens (recommended); if False, use StateDependentCrossFeatNet
        num_query_tokens: int = 4,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_ff_dim: Optional[int] = None,
        cross_attn_dropout: float = 0.1,
        # StateDependentCrossFeatNet settings (when use_learnable_query_tokens=False)
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: Optional[tuple] = None,
        fusion_hidden_dims=(256, 128, 64),
        actor_hidden_dims=(64,),
        critic_hidden_dims=(64,),
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"ActorCriticPTV3Momentum.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_obstacles = num_obstacles
        self.num_ee_points = num_ee_points
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_ptv3 = freeze_ptv3

        self._cached_point_types: dict[tuple[str, int], torch.Tensor] = {}

        # Calculate total number of objects: 1 target + num_obstacles + (1 if ee_cloud else 0)
        self.num_objects = 1 + num_obstacles + (1 if num_ee_points > 0 else 0)

        # Validate observation layout
        # Point cloud layout: [object_cloud, obstacle_clouds, ee_cloud, extra_state]
        object_pc_dim = self.num_points * self.point_dim
        obstacle_pc_dim = num_obstacles * self.num_points * self.point_dim
        ee_pc_dim = (num_ee_points * self.point_dim) if num_ee_points > 0 else 0
        self.pc_dim = object_pc_dim + obstacle_pc_dim + ee_pc_dim

        print(f"[ActorCriticPTV3Momentum] Point cloud layout:")
        print(f"  - Object points: {self.num_points} (dim: {object_pc_dim})")
        print(
            f"  - Obstacle points: {num_obstacles} x {self.num_points} = {num_obstacles * self.num_points} (dim: {obstacle_pc_dim})"
        )
        print(f"  - End-effector points: {num_ee_points} (dim: {ee_pc_dim})")
        print(f"  - Total point cloud dim: {self.pc_dim}")
        print(f"  - Total objects for encoder: {self.num_objects}")

        # Extra state dimension (all non-point-cloud observations)
        self.extra_state_dim = num_actor_obs - self.pc_dim
        print(f"  - Extra state dim: {self.extra_state_dim}")

        activation_fn = resolve_nn_activation(activation)

        # ----------------------------------------------------------------------
        # PTV3 Momentum Encoder setup
        # ----------------------------------------------------------------------
        if isinstance(ptv3_cfg, PTV3MomentumAwarePointEncoderConfig):
            cfg = ptv3_cfg
        elif isinstance(ptv3_cfg, dict):
            cfg = PTV3MomentumAwarePointEncoderConfig(**ptv3_cfg)
        else:
            # Default config
            cfg = PTV3MomentumAwarePointEncoderConfig()

        # Set max_dec_stage from parameter
        cfg.max_dec_stage = max_dec_stage

        self.ptv3_encoder = PTV3MomentumAwarePointEncoder(cfg)
        self.ptv3_cfg = cfg

        # Load pretrained weights if provided
        if ptv3_ckpt is not None:
            print(f"[ActorCriticPTV3Momentum] Loading encoder from {ptv3_ckpt}")
            state = torch.load(ptv3_ckpt, map_location="cpu")
            weights = state.get("model", state.get("state_dict", state))
            missing = self.ptv3_encoder.load_state_dict(
                weights, strict=encoder_strict_load
            )
            if isinstance(missing, tuple):
                missing_keys, unexpected_keys = missing
                if missing_keys:
                    print(f"[ActorCriticPTV3Momentum] Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(
                        f"[ActorCriticPTV3Momentum] Unexpected keys: {unexpected_keys}"
                    )
            print(f"[ActorCriticPTV3Momentum] Encoder loaded successfully")

        # Freeze encoder if specified
        if freeze_ptv3:
            for p in self.ptv3_encoder.parameters():
                p.requires_grad = False
            self.ptv3_encoder.eval()
            print(f"[ActorCriticPTV3Momentum] Encoder frozen")

        if max_dec_stage is None:
            self.encoder_feat_dim = int(cfg.dec_channels[0])
            print(
                f"[ActorCriticPTV3Momentum] Using all decoder stages, output dim: {self.encoder_feat_dim}"
            )
        elif max_dec_stage == 0:
            self.encoder_feat_dim = int(cfg.enc_channels[-1])
            print(
                f"[ActorCriticPTV3Momentum] Using encoder output only (no decoder), output dim: {self.encoder_feat_dim}"
            )
        else:
            num_dec_stages = len(cfg.dec_channels)
            reversed_idx = num_dec_stages - 1 - max_dec_stage
            self.encoder_feat_dim = int(cfg.dec_channels[reversed_idx])
            print(
                f"[ActorCriticPTV3Momentum] Using decoder up to stage {max_dec_stage}, output dim: {self.encoder_feat_dim} (reversed index: {reversed_idx})"
            )

        self.patch_token_dim = int(self.encoder_feat_dim)

        # Cross-attention setup: choose between StateDependentCrossFeatNet or learnable query tokens
        self.use_learnable_query_tokens = use_learnable_query_tokens

        if self.use_learnable_query_tokens:
            # Use learnable query tokens with TransformerDecoder (recommended)
            print(
                "[ActorCriticPTV3Momentum] Using learnable query tokens with TransformerDecoder"
            )
            self.num_query_tokens = num_query_tokens

            print(f"  - Query tokens: {num_query_tokens}")
            print(f"  - Token dimension: {self.patch_token_dim}")
            print(f"  - Cross attention heads: {cross_attn_heads}")
            print(f"  - Cross attention layers: {cross_attn_layers}")

            # Learnable query tokens
            self.query_tokens = nn.Parameter(
                torch.randn(1, num_query_tokens, self.patch_token_dim) * 0.02
            )

            # TransformerDecoder for cross-attention
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.patch_token_dim,
                nhead=cross_attn_heads,
                dim_feedforward=cross_attn_ff_dim or (self.patch_token_dim * 4),
                dropout=cross_attn_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.cross_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=cross_attn_layers
            )

            # Calculate output dimension
            cross_out_dim = num_query_tokens * self.patch_token_dim
            # Fusion input: [cross_attn_out, extra_state]
            fusion_input_dim = cross_out_dim + self.extra_state_dim
        else:
            # Use StateDependentCrossFeatNet (extra_state projected to query tokens)
            print(
                "[ActorCriticPTV3Momentum] Using StateDependentCrossFeatNet for feature fusion"
            )

            # Default query keys: use extra_state as query
            if sd_query_keys is None:
                sd_query_keys = ("extra_state",)

            # Context dimension is extra_state_dim
            sd_ctx_dim = self.extra_state_dim

            # Create StateDependentCrossFeatNet config
            # Note: We can't determine exact num_tokens in advance due to variable patch numbers,
            # but we need to specify token_dim for key/value projection
            sd_cfg = StateDependentCrossFeatNet.Config(
                dim_in=(
                    None,
                    self.patch_token_dim,
                ),  # None means variable number of tokens
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=sd_cat_ctx,
            )
            self.state_cross = StateDependentCrossFeatNet(sd_cfg)

            # Calculate SD cross output dimension
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim

            fusion_input_dim = sd_out_dim

        print(f"[ActorCriticPTV3Momentum] Fusion input dim: {fusion_input_dim}")
        print(f"  - Extra state: {self.extra_state_dim}")

        # Build fusion MLP
        self.fusion_mlp = self._build_fusion_mlp(
            fusion_input_dim, fusion_hidden_dims, activation_fn
        )
        # Ensure fusion_out_dim is an integer
        if fusion_hidden_dims:
            fusion_out_dim = int(fusion_hidden_dims[-1])
        else:
            fusion_out_dim = int(fusion_input_dim)

        # Build Actor and Critic heads
        self.actor = self._build_mlp(
            fusion_out_dim, actor_hidden_dims, activation_fn, num_actions
        )
        self.critic = self._build_mlp(
            fusion_out_dim, critic_hidden_dims, activation_fn, 1
        )

        # Action distribution parameters
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticPTV3Momentum] Initialization complete")

    # --------------------------------------------------------------------------
    # Utility builders
    # --------------------------------------------------------------------------
    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dims, activation, output_dim: Optional[int] = None
    ):
        """Build MLP with optional output layer."""
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation)
            prev_dim = hidden

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers) if layers else nn.Identity()

    @staticmethod
    def _build_fusion_mlp(input_dim: int, hidden_dims, activation):
        """Build fusion MLP (no output layer)."""
        if not hidden_dims:
            return nn.Identity()
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation)
            prev_dim = hidden
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------------
    # Observation parsing
    # --------------------------------------------------------------------------
    def _split_observations(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Extract point clouds
        offset = 0
        pc_parts = []

        # Object cloud
        object_pc = obs[:, offset : offset + self.num_points * self.point_dim]
        pc_parts.append(object_pc.view(batch_size, self.num_points, self.point_dim))
        offset += self.num_points * self.point_dim

        # Obstacle clouds
        if self.num_obstacles > 0:
            obstacle_dim = self.num_obstacles * self.num_points * self.point_dim
            obstacle_pc = obs[:, offset : offset + obstacle_dim]
            pc_parts.append(
                obstacle_pc.view(
                    batch_size, self.num_obstacles * self.num_points, self.point_dim
                )
            )
            offset += obstacle_dim

        # End-effector cloud
        if self.num_ee_points > 0:
            ee_dim = self.num_ee_points * self.point_dim
            ee_pc = obs[:, offset : offset + ee_dim]
            pc_parts.append(ee_pc.view(batch_size, self.num_ee_points, self.point_dim))
            offset += ee_dim

        # Concatenate to unified point cloud
        pointcloud = torch.cat(pc_parts, dim=1)  # [B, total_points, point_dim]

        # Extract extra state (all remaining observations)
        extra_state = obs[:, offset:]  # [B, extra_state_dim]

        return pointcloud, extra_state

    def _get_or_create_point_types(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        key = (str(device), int(batch_size))
        cached = self._cached_point_types.get(key, None)
        if cached is not None:
            return cached

        point_types = self.ptv3_encoder.create_point_types_from_structure(
            batch_size=batch_size,
            num_object_points=self.num_points,
            num_obstacle_points=self.num_obstacles * self.num_points,
            num_robot_points=self.num_ee_points,
            device=device,
        )
        self._cached_point_types[key] = point_types
        return point_types

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------
    def _get_features(self, observations: torch.Tensor):
        """
        Extract features from observations.

        Pipeline:
            1. Split observations
            2. PTV3 encoder: pointcloud → encoder output tokens
            3. Apply cross-attention or simple concatenation
            4. Fusion MLP: → fused_features

        Returns:
            fused_features: [B, fusion_out_dim]
        """
        # Split observations
        pointcloud, extra_state = self._split_observations(observations)

        # Create point_types if needed
        if self.ptv3_cfg.use_type_embedding:
            batch_size = pointcloud.shape[0]
            point_types = self._get_or_create_point_types(
                batch_size=batch_size, device=pointcloud.device
            )
        else:
            point_types = None

        # Encode tokens
        if self.freeze_ptv3:
            with torch.no_grad():
                ret = self.ptv3_encoder.encode_tokens(
                    pointcloud, point_types=point_types, return_features=True
                )
        else:
            ret = self.ptv3_encoder.encode_tokens(
                pointcloud, point_types=point_types, return_features=True
            )
        if isinstance(ret, tuple):
            encoder_tokens, metadata = ret  # [B, num_patches, encoder_feat_dim]
        else:
            encoder_tokens = ret
            metadata = None

        # Get mask if available (for padding)
        mask = metadata.get("mask", None) if metadata is not None else None

        if self.use_learnable_query_tokens:
            # Use learnable query tokens with TransformerDecoder (recommended)
            batch_size = encoder_tokens.shape[0]
            query = self.query_tokens.expand(
                batch_size, -1, -1
            )  # [B, num_query_tokens, token_dim]

            # Apply cross-attention: query attends to encoder tokens
            attn_out = self.cross_decoder(
                tgt=query, memory=encoder_tokens, memory_key_padding_mask=mask
            )  # [B, num_query_tokens, token_dim]

            attn_out_flat = attn_out.reshape(
                batch_size, -1
            )  # [B, num_query_tokens * token_dim]

            # Concatenate with extra_state
            fusion_input = torch.cat([attn_out_flat, extra_state], dim=-1)

            # Apply fusion MLP
            fused_features = self.fusion_mlp(fusion_input)
            return fused_features
        else:
            # Use StateDependentCrossFeatNet
            ctx = {"extra_state": extra_state}

            # Apply StateDependentCrossFeatNet: query from extra_state, key/value from encoder_tokens
            sd_out = self.state_cross(
                encoder_tokens, ctx=ctx, mask=mask
            )  # [B, sd_out_dim]

            fusion_input = sd_out
            fused_features = self.fusion_mlp(fusion_input)
            return fused_features

    # --------------------------------------------------------------------------
    # Actor / Critic interface
    # --------------------------------------------------------------------------
    def update_distribution(self, observations: torch.Tensor):
        """Update action distribution based on observations."""
        features = self._get_features(observations)
        mean = self.actor(features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        """Sample action from current policy."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        """Deterministic action (mean) for inference."""
        features = self._get_features(observations)
        return self.actor(features)

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        """Evaluate value function."""
        features = self._get_features(critic_observations)
        return self.critic(features)

    def reset(self, dones=None):
        """Stateless policy; nothing to reset."""
        pass

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

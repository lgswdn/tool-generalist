# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.models.cloud.momentum_aware_point_encoder import (
    MomentumAwarePointEncoder,
    MomentumAwarePointEncoderConfig,
)
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet


class ActorCriticMomentum(nn.Module):
    """
    Actor-Critic network that leverages the MomentumAwarePointEncoder for point cloud observations.

    Observation layout (per env step):
        [
            object_point_cloud (num_points * point_dim),
            obstacles_point_cloud (num_obstacles * num_points * point_dim),
            ee_point_cloud (num_ee_points * point_dim),
            extra_state (hand_state, robot_state, and other observations)
        ]
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        point_dim: int = 7,
        num_points: int = 512,
        num_obstacles: int = 1,
        num_ee_points: int = 256,
        momentum_cfg: Optional[Dict[str, Any]] = None,
        momentum_ckpt: Optional[str] = None,
        freeze_momentum: bool = True,
        encoder_strict_load: bool = False,
        # Cross-attention fusion settings
        use_learnable_query_tokens: bool = True,
        sd_num_query: int = 16,
        sd_num_query_object: Optional[int] = None,  # Number of query tokens that attend to object-only tokens. If None, defaults to sd_num_query // 2
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: Optional[tuple] = None,
        # Learnable query tokens settings (when use_learnable_query_tokens=True)
        num_query_tokens: int = 16,
        num_query_object_tokens: Optional[int] = None,  # Number of query tokens that attend to object-only tokens. If None, defaults to num_query_tokens // 2
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_ff_dim: Optional[int] = None,
        cross_attn_dropout: float = 0.1,
        fusion_hidden_dims=(256, 128, 64),
        actor_hidden_dims=(64),
        critic_hidden_dims=(64),
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"ActorCriticMomentum.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_obstacles = num_obstacles
        self.num_ee_points = num_ee_points
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_momentum = freeze_momentum

        self._cached_point_types: dict[tuple[str, int], torch.Tensor] = {}

        # Validate observation layout
        self.pc_dim = (
            (1 + self.num_obstacles) * self.num_points * self.point_dim
            + (self.num_ee_points * self.point_dim if self.num_ee_points > 0 else 0)
        )
        self.extra_state_dim = num_actor_obs - self.pc_dim

        activation_fn = resolve_nn_activation(activation)

        # ----------------------------------------------------------------------
        # Momentum encoder setup
        # ----------------------------------------------------------------------
        print(f"[ActorCriticMomentum] Initializing MomentumAwarePointEncoder...")
        print(f"  - Point cloud layout:")
        print(f"    * Object points: {num_points} (dim: {num_points * point_dim})")
        print(f"    * Obstacle points: {num_obstacles} x {num_points} = {num_obstacles * num_points} (dim: {num_obstacles * num_points * point_dim})")
        print(f"    * End-effector points: {num_ee_points} (dim: {num_ee_points * point_dim})")
        print(f"    * Total point cloud dim: {self.pc_dim}")
        
        cfg_dict = (
            dict(asdict(momentum_cfg))
            if isinstance(momentum_cfg, MomentumAwarePointEncoderConfig)
            else (momentum_cfg or {})
        )
        cfg = MomentumAwarePointEncoderConfig(**cfg_dict)
        cfg.enable_decoder = False  # encoder-only for RL
        self.momentum_encoder = MomentumAwarePointEncoder(cfg)
        self.token_dim = cfg.encoder_feature_dim

        if momentum_ckpt is not None:
            print(f"[ActorCriticMomentum] Loading encoder from {momentum_ckpt}")
            state = torch.load(momentum_ckpt, map_location="cpu")
            weights = state.get("model", state.get("state_dict", state))
            missing = self.momentum_encoder.load_state_dict(weights, strict=encoder_strict_load)
            if isinstance(missing, tuple):
                missing_keys, unexpected_keys = missing
                if missing_keys:
                    print(f"[ActorCriticMomentum] Missing keys when loading encoder: {missing_keys}")
                if unexpected_keys:
                    print(f"[ActorCriticMomentum] Unexpected keys when loading encoder: {unexpected_keys}")
            print(f"[ActorCriticMomentum] Encoder loaded successfully")

        if freeze_momentum:
            for p in self.momentum_encoder.parameters():
                p.requires_grad = False
            self.momentum_encoder.eval()
            print(f"[ActorCriticMomentum] Encoder frozen")

        self.num_patches_per_cloud = num_points // cfg.patch_size
        self.num_objects_total = 1 + self.num_obstacles + (1 if self.num_ee_points > 0 else 0)
        self.num_cls_tokens = 1 if getattr(cfg, "vit_use_cls_token", True) else 0
        self.total_num_tokens = self.num_cls_tokens + self.num_objects_total * self.num_patches_per_cloud

        self.use_learnable_query_tokens = use_learnable_query_tokens
        if not self.use_learnable_query_tokens:
            # Option 1: Use StateDependentCrossFeatNet (extra_state projected to query tokens)
            print("[ActorCriticMomentum] Using StateDependentCrossFeatNet for feature fusion")
            # Default query keys: use extra_state (hand_state + remaining observations) as query
            if sd_query_keys is None:
                sd_query_keys = ("extra_state",)
            
            sd_ctx_dim = self.extra_state_dim
            
            # Determine number of query tokens for object-only vs all tokens
            if sd_num_query_object is None:
                sd_num_query_object = max(1, sd_num_query // 2)  # At least 1 query token for object
            else:
                # Validate the specified value
                if sd_num_query_object < 1:
                    raise ValueError(f"sd_num_query_object must be >= 1, got {sd_num_query_object}")
                if sd_num_query_object > sd_num_query:
                    raise ValueError(
                        f"sd_num_query_object ({sd_num_query_object}) cannot exceed "
                        f"sd_num_query ({sd_num_query})"
                    )
            sd_num_query_all = sd_num_query - sd_num_query_object
            
            # Object-only tokens: CLS (if present) + object patches.
            # Encoder orders tokens as: [CLS], object, obstacles..., ee.
            object_only_num_tokens = self.num_cls_tokens + self.num_patches_per_cloud
            
            print(f"  - Query tokens: {sd_num_query_object} for object-only, {sd_num_query_all} for all tokens")
            print(f"  - Token dimension: {self.token_dim}")
            print(f"  - Object-only tokens: {object_only_num_tokens} (CLS: {self.num_cls_tokens}, object patches: {self.num_patches_per_cloud})")
            print(f"  - Context dimension: {sd_ctx_dim}")
            print(f"  - Embedding dimension: {sd_emb_dim}")
            
            sd_cfg_object = StateDependentCrossFeatNet.Config(
                dim_in=(object_only_num_tokens, self.token_dim),
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query_object,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=False,
            )
            self.state_cross_object = StateDependentCrossFeatNet(sd_cfg_object) if sd_num_query_object > 0 else None
            
            sd_cfg_all = StateDependentCrossFeatNet.Config(
                dim_in=(self.total_num_tokens, self.token_dim),
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query_all,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=False,
            )
            self.state_cross_all = StateDependentCrossFeatNet(sd_cfg_all) if sd_num_query_all > 0 else None
            
            # Store configuration for later use
            self.sd_num_query_object = sd_num_query_object
            self.sd_num_query_all = sd_num_query_all
            self.object_only_num_tokens = object_only_num_tokens
            self.sd_cat_ctx = sd_cat_ctx
            
            # Calculate output dimension
            sd_out_dim = 0
            if sd_num_query_object > 0:
                sd_out_dim += sd_num_query_object * sd_emb_dim
                if sd_cat_query:
                    sd_out_dim += sd_num_query_object * sd_emb_dim
            if sd_num_query_all > 0:
                sd_out_dim += sd_num_query_all * sd_emb_dim
                if sd_cat_query:
                    sd_out_dim += sd_num_query_all * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim
            
            fusion_input_dim = sd_out_dim + self.extra_state_dim
        else:
            # Option 2: Use learnable query tokens with TransformerDecoder
            print("[ActorCriticMomentum] Using learnable query tokens with TransformerDecoder")
            self.num_query_tokens = num_query_tokens
            
            if num_query_object_tokens is None:
                num_query_object = max(1, num_query_tokens // 2)  # At least 1 query token for object
            else:
                # Validate the specified value
                if num_query_object_tokens < 1:
                    raise ValueError(f"num_query_object_tokens must be >= 1, got {num_query_object_tokens}")
                if num_query_object_tokens > num_query_tokens:
                    raise ValueError(
                        f"num_query_object_tokens ({num_query_object_tokens}) cannot exceed "
                        f"num_query_tokens ({num_query_tokens})"
                    )
                num_query_object = num_query_object_tokens
            num_query_all = num_query_tokens - num_query_object
            
            # Object-only tokens: CLS (if present) + object patches.
            # Encoder orders tokens as: [CLS], object, obstacles..., ee.
            object_only_num_tokens = self.num_cls_tokens + self.num_patches_per_cloud
            
            print(f'  - Query tokens: {num_query_object} for object-only, {num_query_all} for all tokens')
            print(f'  - Token dimension: {self.token_dim}')
            print(f'  - Object-only tokens: {object_only_num_tokens} (CLS: {self.num_cls_tokens}, object patches: {self.num_patches_per_cloud})')
            print(f'  - Cross attention heads: {cross_attn_heads}')
            print(f'  - Cross attention layers: {cross_attn_layers}')
            
            # Learnable query tokens
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, self.token_dim) * 0.02)
            
            # TransformerDecoder for cross-attention
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.token_dim,
                nhead=cross_attn_heads,
                dim_feedforward=cross_attn_ff_dim or (self.token_dim * 2),
                dropout=cross_attn_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.cross_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cross_attn_layers)
            
            # Store configuration for later use
            self.num_query_object = num_query_object
            self.num_query_all = num_query_all
            self.object_only_num_tokens = object_only_num_tokens
            
            # Calculate output dimension
            cross_out_dim = num_query_tokens * self.token_dim
            fusion_input_dim = cross_out_dim + self.extra_state_dim
        
        self.fusion_mlp = self._build_fusion_mlp(fusion_input_dim, fusion_hidden_dims, activation_fn)
        fusion_out_dim = fusion_hidden_dims[-1] if len(fusion_hidden_dims) > 0 else fusion_input_dim

        print(f"[ActorCriticMomentum] Fusion input dim: {fusion_input_dim}")
        print(f"  - Extra state: {self.extra_state_dim}")

        # Actor / Critic heads take fused features
        self.actor = self._build_mlp(fusion_out_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = self._build_mlp(fusion_out_dim, critic_hidden_dims, activation_fn, 1)

        # Action distribution params
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticMomentum] Initialization complete")

    # --------------------------------------------------------------------------
    # Utility builders
    # --------------------------------------------------------------------------
    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims, activation, output_dim: Optional[int] = None):
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
        """
        Split observations into point clouds and extra_state.
        
        Observation layout:
            [point_clouds, extra_state...]
        """
        object_dim = self.num_points * self.point_dim
        obstacles_dim = self.num_obstacles * self.num_points * self.point_dim
        ee_dim = (self.num_ee_points * self.point_dim) if self.num_ee_points > 0 else 0

        object_cloud_flat = obs[:, :object_dim]
        obstacles_flat = obs[:, object_dim : object_dim + obstacles_dim]
        ee_flat = obs[:, object_dim + obstacles_dim : object_dim + obstacles_dim + ee_dim]
        extra_state = obs[:, object_dim + obstacles_dim + ee_dim :]

        object_cloud = object_cloud_flat.view(-1, self.num_points, self.point_dim)
        if self.num_obstacles > 0:
            obstacles_cloud = obstacles_flat.view(-1, self.num_obstacles, self.num_points, self.point_dim)
        else:
            obstacles_cloud = torch.empty(
                obs.shape[0], 0, self.num_points, self.point_dim, device=obs.device, dtype=obs.dtype
            )

        # EE cloud
        if self.num_ee_points > 0:
            ee_cloud = ee_flat.view(-1, self.num_ee_points, self.point_dim)
        else:
            ee_cloud = torch.empty(
                obs.shape[0], 0, self.point_dim, device=obs.device, dtype=obs.dtype
            )

        if self.num_obstacles > 0:
            obstacles_flat = obstacles_cloud.view(-1, self.num_obstacles * self.num_points, self.point_dim)
        else:
            obstacles_flat = torch.empty(
                obs.shape[0], 0, self.point_dim, device=obs.device, dtype=obs.dtype
            )

        all_clouds = torch.cat([object_cloud, obstacles_flat, ee_cloud], dim=1)
        return all_clouds, extra_state

    def _get_or_create_point_types(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create point_types tensor with caching."""
        key = (str(device), int(batch_size))
        cached = self._cached_point_types.get(key, None)
        if cached is not None:
            return cached

        point_types = self.momentum_encoder.create_point_types_from_structure(
            batch_size=batch_size,
            num_object_points=self.num_points,
            num_obstacle_points=self.num_obstacles * self.num_points,
            num_robot_points=self.num_ee_points,
            device=device,
        )
        self._cached_point_types[key] = point_types
        return point_types

    # --------------------------------------------------------------------------
    # Feature extraction via StateDependentCrossFeatNet
    # --------------------------------------------------------------------------
    def _tokenize(self, observations: torch.Tensor):
        """
        Extract tokens from momentum encoder and split observations.
        
        Returns:
            fused_tokens: [B, total_num_tokens, token_dim] - all tokens from momentum encoder
            extra_state: [B, extra_state_dim] - all non-point-cloud observations
        """
        pointclouds, extra_state = self._split_observations(observations)

        # Create point_types if needed (for type embedding)
        point_types = None
        if self.momentum_encoder.cfg.use_type_embedding:
            batch_size = pointclouds.shape[0]
            point_types = self._get_or_create_point_types(
                batch_size=batch_size, device=pointclouds.device
            )

        # Encode tokens (with no_grad if encoder is frozen)
        if self.freeze_momentum:
            with torch.no_grad():
                ret = self.momentum_encoder.encode_tokens(
                    pointclouds, point_types=point_types, return_features=False
                )
        else:
            ret = self.momentum_encoder.encode_tokens(
                pointclouds, point_types=point_types, return_features=False
            )
        
        fused_tokens = ret[0]
        
        return fused_tokens, extra_state

    def _get_features(self, observations: torch.Tensor):
        """
        Get fused features using either StateDependentCrossFeatNet or learnable query tokens.
        """
        fused_tokens, extra_state = self._tokenize(
            observations
        )  # [B, total_num_tokens, token_dim], [B, extra_state_dim]

        if not self.use_learnable_query_tokens:
            ctx = {"extra_state": extra_state}
            
            # Extract object-only tokens (first object_only_num_tokens tokens)
            object_only_tokens = fused_tokens[:, :self.object_only_num_tokens, :]  # [B, object_only_num_tokens, token_dim]
            
            # Differentiated attention: some query tokens attend to object only, rest attend to all
            sd_out_parts = []
            
            # First query tokens attend to object-only tokens
            if self.sd_num_query_object > 0 and self.state_cross_object is not None:
                object_features = self.state_cross_object(object_only_tokens, ctx=ctx, mask=None)  # [B, sd_out_object_dim]
                sd_out_parts.append(object_features)
            
            # Remaining query tokens attend to all tokens
            if self.sd_num_query_all > 0 and self.state_cross_all is not None:
                all_features = self.state_cross_all(fused_tokens, ctx=ctx, mask=None)  # [B, sd_out_all_dim]
                sd_out_parts.append(all_features)
            
            # Concatenate all features
            sd_out = torch.cat(sd_out_parts, dim=-1) if sd_out_parts else torch.empty(fused_tokens.shape[0], 0, device=fused_tokens.device)
            
            if self.sd_cat_ctx:
                sd_out = torch.cat([sd_out, extra_state], dim=-1)
            
            fusion_input = torch.cat([sd_out, extra_state], dim=-1)
            fused_features = self.fusion_mlp(fusion_input)
            return fused_features
        else:
            batch = fused_tokens.shape[0]
            query = self.query_tokens.expand(batch, -1, -1)  # [B, num_query_tokens, token_dim]
            
            object_only_tokens = fused_tokens[:, :self.object_only_num_tokens, :]  # [B, object_only_num_tokens, token_dim]
            
            attn_out_parts = []
            
            if self.num_query_object > 0:
                object_queries = query[:, :self.num_query_object, :]  # [B, num_query_object, token_dim]
                object_attn_out = self.cross_decoder(tgt=object_queries, memory=object_only_tokens)  # [B, num_query_object, token_dim]
                attn_out_parts.append(object_attn_out)
            
            if self.num_query_all > 0:
                all_queries = query[:, self.num_query_object:, :]  # [B, num_query_all, token_dim]
                all_attn_out = self.cross_decoder(tgt=all_queries, memory=fused_tokens, memory_key_padding_mask=None)  # [B, num_query_all, token_dim]
                attn_out_parts.append(all_attn_out)
            
            attn_out = torch.cat(attn_out_parts, dim=1) if attn_out_parts else torch.empty(batch, 0, self.token_dim, device=fused_tokens.device)
            attn_out_flat = attn_out.reshape(batch, -1)  # [B, num_query_tokens * token_dim]
            
            fusion_input = torch.cat([attn_out_flat, extra_state], dim=-1)
            
            fused_features = self.fusion_mlp(fusion_input)
            return fused_features

    # --------------------------------------------------------------------------
    # Actor / Critic helpers
    # --------------------------------------------------------------------------
    def update_distribution(self, observations: torch.Tensor):
        features = self._get_features(observations)
        mean = self.actor(features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        features = self._get_features(observations)
        return self.actor(features)

    def reset(self, dones=None):
        """Stateless policy; nothing to reset."""
        pass

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        features = self._get_features(critic_observations)
        return self.critic(features)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)



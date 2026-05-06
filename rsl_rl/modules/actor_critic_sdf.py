# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic network using SDFPointCloudEncoder (joint ViT encoder).

This actor-critic uses the pretrained SDFPointCloudEncoder from geometry pretraining
as the point cloud encoder. The encoder outputs:
  - global_feat:  CLS token (B, D)   — joint scene summary
  - tool_tokens:  (B, P, D)          — tool patch tokens
  - obj_tokens:   (B, P, D)          — object patch tokens

These features are fused with robot state via SD-Cross attention (or learnable query
tokens with TransformerDecoder) for policy learning.

Observation layout:
    object_cloud (num_points*3) | tool_cloud (num_points*3) | extra_state
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.models.cloud.sdf_encoder import SDFPointCloudEncoder, SDFEncoderCfg
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet


class ActorCriticSDF(nn.Module):
    """Actor-critic network using SDFPointCloudEncoder.

    Mirrors the ActorCriticMomentum architecture but uses SDFPointCloudEncoder
    (3D point clouds only) instead of MomentumAwarePointEncoder.

    Observation layout:
        object_cloud (num_points*3) | tool_cloud (num_points*3) | extra_state

    The encoder processes tool_pc and obj_pc jointly, producing cross-stream-aware
    tokens. Point clouds are centered around the object center before encoding
    to match the pretraining distribution. The object center is prepended
    to extra_state as additional context.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        # Point cloud settings
        num_points: int = 512,
        point_dim: int = 3,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        # Encoder weights
        encoder_weights_path: Optional[str] = None,
        freeze_encoder: bool = True,
        # Cross-attention fusion settings
        use_learnable_query_tokens: bool = False,
        sd_num_query: int = 16,
        sd_num_query_object: Optional[int] = None,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: Optional[tuple] = None,
        # Learnable query tokens settings (when use_learnable_query_tokens=True)
        num_query_tokens: int = 16,
        num_query_object_tokens: Optional[int] = None,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_ff_dim: Optional[int] = None,
        cross_attn_dropout: float = 0.0,
        # Network architecture
        fusion_hidden_dims=(512, 256, 128),
        actor_hidden_dims=(64,),
        critic_hidden_dims=(128,),
        # Activation / noise
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"ActorCriticSDF.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_encoder = freeze_encoder

        # Validate observation layout
        # Layout: object_cloud | tool_cloud | obj_centroid(3) | tool_centroid(3) | extra_state
        self.pc_dim = 2 * num_points * point_dim
        self.centroid_dim = 6            # obj_centroid(3) + tool_centroid(3)
        self.extra_state_dim = num_actor_obs - self.pc_dim - self.centroid_dim

        activation_fn = resolve_nn_activation(activation)

        # ------------------------------------------------------------------
        # SDF encoder setup
        # ------------------------------------------------------------------
        print(f"[ActorCriticSDF] Initializing SDFPointCloudEncoder...")
        print(f"  - Point cloud layout:")
        print(f"    * Object points: {num_points} (dim: {num_points * point_dim})")
        print(f"    * Tool points: {num_points} (dim: {num_points * point_dim})")
        print(f"    * Total point cloud dim: {self.pc_dim}")

        enc_cfg = SDFEncoderCfg(
            num_pts=num_points,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            weights_path=encoder_weights_path,
            freeze=freeze_encoder,
        )
        self.encoder = SDFPointCloudEncoder(enc_cfg)

        D = self.encoder.feature_dim   # Token dimension
        P = self.encoder.num_patches   # Patches per cloud
        self.token_dim = D

        # Token layout from SDF encoder (CLS is stripped internally):
        #   tool_tokens (P) + obj_tokens (P) = 2P total
        self.num_cls_tokens = 0
        self.total_num_tokens = 2 * P
        self.num_patches_per_cloud = P

        print(f"[ActorCriticSDF] Encoder config: D={D}, P={P}, vit_depth={vit_depth}, vit_heads={vit_heads}")

        # ------------------------------------------------------------------
        # Feature fusion
        # ------------------------------------------------------------------
        self.use_learnable_query_tokens = use_learnable_query_tokens

        # Context dimension for SD-Cross or fusion:
        # [obj_centroid(3), tool_centroid(3), extra_state]
        sd_ctx_dim = self.centroid_dim + self.extra_state_dim

        if not self.use_learnable_query_tokens:
            # Option 1: StateDependentCrossFeatNet
            print("[ActorCriticSDF] Using StateDependentCrossFeatNet for feature fusion")

            if sd_query_keys is None:
                sd_query_keys = ("extra_state",)

            # Determine query token split
            if sd_num_query_object is None:
                sd_num_query_object = max(1, sd_num_query // 2)
            else:
                if sd_num_query_object < 1:
                    raise ValueError(f"sd_num_query_object must be >= 1, got {sd_num_query_object}")
                if sd_num_query_object > sd_num_query:
                    raise ValueError(
                        f"sd_num_query_object ({sd_num_query_object}) cannot exceed "
                        f"sd_num_query ({sd_num_query})"
                    )
            sd_num_query_all = sd_num_query - sd_num_query_object

            # Object-only tokens: obj_tokens (P) — no CLS in output
            object_only_num_tokens = P

            print(f"  - Query tokens: {sd_num_query_object} for object-only, {sd_num_query_all} for all tokens")
            print(f"  - Token dimension: {self.token_dim}")
            print(f"  - Object-only tokens: {object_only_num_tokens} (object patches: {P})")
            print(f"  - Context dimension: {sd_ctx_dim}")
            print(f"  - Embedding dimension: {sd_emb_dim}")

            sd_cfg_object = StateDependentCrossFeatNet.Config(
                dim_in=(object_only_num_tokens, D),
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
                dim_in=(self.total_num_tokens, D),
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query_all,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=False,
            )
            self.state_cross_all = StateDependentCrossFeatNet(sd_cfg_all) if sd_num_query_all > 0 else None

            # Store configuration
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

            fusion_input_dim = sd_out_dim + sd_ctx_dim
        else:
            # Option 2: Learnable query tokens with TransformerDecoder
            print("[ActorCriticSDF] Using learnable query tokens with TransformerDecoder")
            self.num_query_tokens = num_query_tokens

            if num_query_object_tokens is None:
                num_query_object = max(1, num_query_tokens // 2)
            else:
                if num_query_object_tokens < 1:
                    raise ValueError(f"num_query_object_tokens must be >= 1, got {num_query_object_tokens}")
                if num_query_object_tokens > num_query_tokens:
                    raise ValueError(
                        f"num_query_object_tokens ({num_query_object_tokens}) cannot exceed "
                        f"num_query_tokens ({num_query_tokens})"
                    )
                num_query_object = num_query_object_tokens
            num_query_all = num_query_tokens - num_query_object

            # Object-only tokens: obj_tokens (P) — no CLS in output
            object_only_num_tokens = P

            print(f"  - Query tokens: {num_query_object} for object-only, {num_query_all} for all tokens")
            print(f"  - Token dimension: {self.token_dim}")
            print(f"  - Object-only tokens: {object_only_num_tokens} (object patches: {P})")
            print(f"  - Cross attention heads: {cross_attn_heads}")
            print(f"  - Cross attention layers: {cross_attn_layers}")

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

            # Store configuration
            self.num_query_object = num_query_object
            self.num_query_all = num_query_all
            self.object_only_num_tokens = object_only_num_tokens

            # Calculate output dimension
            cross_out_dim = num_query_tokens * self.token_dim
            fusion_input_dim = cross_out_dim + sd_ctx_dim

        self.fusion_mlp = self._build_fusion_mlp(fusion_input_dim, fusion_hidden_dims, activation_fn)
        fusion_out_dim = fusion_hidden_dims[-1] if len(fusion_hidden_dims) > 0 else fusion_input_dim

        print(f"[ActorCriticSDF] Fusion input dim: {fusion_input_dim}")
        print(f"  - Extra state: {self.extra_state_dim}")

        # Actor / Critic heads
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

        print(f"[ActorCriticSDF] Initialization complete")

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
        """Split observations into point clouds, centroids, and extra_state.

        Layout: object_cloud | tool_cloud | obj_centroid(3) | tool_centroid(3) | extra_state
        """
        object_dim = self.num_points * self.point_dim
        tool_dim = self.num_points * self.point_dim
        centroid_start = object_dim + tool_dim

        object_cloud = obs[:, :object_dim].view(-1, self.num_points, self.point_dim)
        tool_cloud = obs[:, object_dim:object_dim + tool_dim].view(-1, self.num_points, self.point_dim)
        obj_centroid = obs[:, centroid_start:centroid_start + 3]      # (B, 3)
        tool_centroid = obs[:, centroid_start + 3:centroid_start + 6]  # (B, 3)
        extra_state = obs[:, centroid_start + 6:]

        return object_cloud, tool_cloud, obj_centroid, tool_centroid, extra_state

    # --------------------------------------------------------------------------
    # Tokenization via SDF encoder
    # --------------------------------------------------------------------------
    def _tokenize(self, observations: torch.Tensor):
        """Extract tokens from SDF encoder and split observations.

        Point clouds arrive PRE-CENTERED at (0,0,0) from the environment
        observation functions (get_object_pointcloud_in_env_frame and
        get_tool_pointcloud_in_env_frame).  The corresponding env-frame
        centroids are available as separate observation terms (get_obj_centroid,
        get_tool_centroid) and are explicitly sliced out here.

        This mirrors pretraining (contact_gen.py filter_and_save):
          - tool_pts_canonical : (P,3) centered at (0,0,0)   <- encoder input
          - obj_pts_canonical  : (Q,3) centered at (0,0,0)   <- encoder input
          - tool_translations  : (N,3) world-frame centroid  <- pose context
          - obj_centroid       : (3,)  world-frame centroid  <- pose context

        Returns:
            all_tokens:    (B, 2P, D) — tool_tokens + obj_tokens
            ctx_vec:       (B, centroid_dim + extra_state_dim) — conditioning vector
        """
        object_cloud, tool_cloud, obj_centroid, tool_centroid, extra_state = \
            self._split_observations(observations)

        # Clouds are already centered — feed directly to encoder
        if self.freeze_encoder:
            with torch.no_grad():
                res = self.encoder.encode(tool_cloud, object_cloud)
        else:
            res = self.encoder.encode(tool_cloud, object_cloud)

        all_tokens = res.fused_tokens  # (B, 2P, D)

        # Context vector: [obj_centroid(3), tool_centroid(3), extra_state(...)]
        ctx_vec = torch.cat([obj_centroid, tool_centroid, extra_state], dim=-1)

        return all_tokens, ctx_vec

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------
    def _get_features(self, observations: torch.Tensor):
        """Get fused features using either SD-Cross or learnable query tokens."""
        all_tokens, ctx_vec = self._tokenize(observations)

        if not self.use_learnable_query_tokens:
            ctx = {"extra_state": ctx_vec}

            # Object-only tokens: obj_tokens (last P tokens)
            # Token layout: [tool_0..tool_{P-1}, obj_0..obj_{P-1}]
            P = self.num_patches_per_cloud
            object_only_tokens = all_tokens[:, P:, :]  # (B, P, D)

            # Differentiated attention
            sd_out_parts = []

            if self.sd_num_query_object > 0 and self.state_cross_object is not None:
                object_features = self.state_cross_object(object_only_tokens, ctx=ctx, mask=None)
                sd_out_parts.append(object_features)

            if self.sd_num_query_all > 0 and self.state_cross_all is not None:
                all_features = self.state_cross_all(all_tokens, ctx=ctx, mask=None)
                sd_out_parts.append(all_features)

            sd_out = torch.cat(sd_out_parts, dim=-1) if sd_out_parts else torch.empty(
                all_tokens.shape[0], 0, device=all_tokens.device
            )

            if self.sd_cat_ctx:
                sd_out = torch.cat([sd_out, ctx_vec], dim=-1)

            fusion_input = torch.cat([sd_out, ctx_vec], dim=-1)
            fused_features = self.fusion_mlp(fusion_input)
            return fused_features
        else:
            batch = all_tokens.shape[0]
            query = self.query_tokens.expand(batch, -1, -1)  # (B, num_query_tokens, D)

            P = self.num_patches_per_cloud
            object_only_tokens = all_tokens[:, P:, :]  # (B, P, D)

            attn_out_parts = []

            if self.num_query_object > 0:
                object_queries = query[:, :self.num_query_object, :]
                object_attn_out = self.cross_decoder(tgt=object_queries, memory=object_only_tokens)
                attn_out_parts.append(object_attn_out)

            if self.num_query_all > 0:
                all_queries = query[:, self.num_query_object:, :]
                all_attn_out = self.cross_decoder(tgt=all_queries, memory=all_tokens, memory_key_padding_mask=None)
                attn_out_parts.append(all_attn_out)

            attn_out = torch.cat(attn_out_parts, dim=1) if attn_out_parts else torch.empty(
                batch, 0, self.token_dim, device=all_tokens.device
            )
            attn_out_flat = attn_out.reshape(batch, -1)  # (B, num_query_tokens * D)

            fusion_input = torch.cat([attn_out_flat, ctx_vec], dim=-1)

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

    def forward(self):
        raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
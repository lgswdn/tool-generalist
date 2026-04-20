# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic network using SDFPointCloudEncoder (joint ViT encoder).

This actor-critic uses the pretrained SDFPointCloudEncoder from geometry pretraining
as the point cloud encoder, replacing ICPNet. The encoder outputs:
  - global_feat: CLS token (B, D) — joint scene summary
  - tool_tokens: (B, P, D) — tool patch tokens
  - obj_tokens:  (B, P, D) — object patch tokens

These features are fused with robot state via SD-Cross attention for policy learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

from rsl_rl.modules.models.cloud.sdf_encoder import SDFPointCloudEncoder, SDFEncoderCfg
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet


class ActorCriticSDF(nn.Module):
    """Actor-critic network using SDFPointCloudEncoder.

    Observation layout (env_tool):
        object_cloud (512*3=1536) | tool_cloud (512*3=1536) | hand_state (9) | rest (robot_state+prev_action+rel_goal+phys_params)

    The encoder processes tool_pc and obj_pc jointly, producing cross-stream-aware features.

    IMPORTANT: Point clouds are centered around the object center before encoding,
    to match the pretraining distribution where objects are centered at origin.
    The object center position is then fed as context to the fusion network.
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        # Point cloud settings
        num_points: int = 512,
        point_dim: int = 3,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        # Encoder weights
        encoder_weights_path: str | None = None,
        freeze_encoder: bool = True,
        # Fusion settings
        fusion_hidden_dims: list[int] = None,
        fusion_use_norm: bool = True,
        fusion_norm_type: str = "layer",
        # Actor/Critic heads
        actor_hidden_dims: list[int] = None,
        actor_use_norm: bool = True,
        actor_norm_type: str = "layer",
        actor_output_activation: bool = False,
        critic_hidden_dims: list[int] = None,
        critic_use_norm: bool = True,
        critic_norm_type: str = "layer",
        # SD-Cross settings
        use_sd_cross: bool = True,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: tuple | None = None,
        # Activation / noise
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"[ActorCriticSDF] __init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Save configuration
        self.num_points = num_points
        self.point_dim = point_dim

        # Calculate dimensions
        object_pc_dim = num_points * point_dim
        tool_pc_dim = num_points * point_dim
        hand_state_dim = 9
        self.nonpc_obs_dim = num_actor_obs - object_pc_dim - tool_pc_dim - hand_state_dim

        print(f"[ActorCriticSDF] Observation dimensions:")
        print(f"  - Object point cloud: {object_pc_dim} ({num_points} points × {point_dim}D)")
        print(f"  - Tool point cloud: {tool_pc_dim} ({num_points} points × {point_dim}D)")
        print(f"  - Hand state: {hand_state_dim}D")
        print(f"  - Regular observations: {self.nonpc_obs_dim}")
        print(f"  - Total: {num_actor_obs}")

        # Create SDF encoder
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

        D = self.encoder.feature_dim  # Token dimension
        P = self.encoder.num_patches   # Patches per cloud

        print(f"[ActorCriticSDF] Encoder config: D={D}, P={P}, vit_depth={vit_depth}, vit_heads={vit_heads}")

        # Activation function
        activation_fn = resolve_nn_activation(activation)

        # Set default hidden dimensions
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        if actor_hidden_dims is None:
            actor_hidden_dims = [64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [128]

        # Feature fusion
        self.use_sd_cross = use_sd_cross

        if self.use_sd_cross:
            if sd_query_keys is None:
                sd_query_keys = ("rest",)

            # SD-Cross context: obj_center (3) + hand_state + regular_obs
            sd_ctx_dim = 3 + hand_state_dim + self.nonpc_obs_dim

            # Total tokens: tool_tokens (P) + obj_tokens (P) + global_feat (1)
            total_num_tokens = 2 * P + 1

            # Split queries: half for object, half for tool/global
            num_query_obj = max(1, sd_num_query // 2)
            num_query_all = sd_num_query - num_query_obj

            print(f"[ActorCriticSDF] SD-Cross config: total_tokens={total_num_tokens}, sd_emb_dim={sd_emb_dim}")
            print(f"  - Query tokens: {num_query_obj} for object, {num_query_all} for all")

            # SD-Cross for object tokens only
            if num_query_obj > 0:
                sd_cfg_obj = StateDependentCrossFeatNet.Config(
                    dim_in=(P, D),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_obj,
                    ctx_dim=sd_ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,
                )
                self.sd_cross_obj = StateDependentCrossFeatNet(sd_cfg_obj)
            else:
                self.sd_cross_obj = None

            # SD-Cross for all tokens (tool + obj + global)
            if num_query_all > 0:
                sd_cfg_all = StateDependentCrossFeatNet.Config(
                    dim_in=(total_num_tokens, D),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_all,
                    ctx_dim=sd_ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,
                )
                self.sd_cross_all = StateDependentCrossFeatNet(sd_cfg_all)
            else:
                self.sd_cross_all = None

            # Calculate fusion input dim
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim

            fusion_input_dim = sd_out_dim

            self.num_query_obj = num_query_obj
            self.num_query_all = num_query_all
            self.sd_cat_ctx = sd_cat_ctx
        else:
            # Simple concatenation: flatten all tokens + context (obj_center + hand_state + regular_obs)
            total_feat_dim = (2 * P + 1) * D + 3 + hand_state_dim + self.nonpc_obs_dim
            fusion_input_dim = total_feat_dim

        # Build fusion MLP (no final output layer - just hidden layers)
        self.feature_fusion = self._build_fusion_mlp(
            fusion_input_dim, fusion_hidden_dims, activation_fn,
            fusion_use_norm, fusion_norm_type
        )

        # Actor network
        self.actor = self._build_mlp(
            fusion_hidden_dims[-1], actor_hidden_dims, num_actions,
            activation_fn, actor_use_norm, actor_norm_type,
            is_actor=True, output_activation=actor_output_activation
        )

        # Critic network
        self.critic = self._build_mlp(
            fusion_hidden_dims[-1], critic_hidden_dims, 1,
            activation_fn, critic_use_norm, critic_norm_type,
            is_actor=False
        )

        print(f"[ActorCriticSDF] Network architecture:")
        print(f"  - Encoder: SDFPointCloudEncoder (freeze={freeze_encoder})")
        print(f"  - Fusion MLP: {fusion_hidden_dims}")
        if self.use_sd_cross:
            print(f"  - SD-Cross (obj): {self.sd_cross_obj}")
            print(f"  - SD-Cross (all): {self.sd_cross_all}")
        print(f"  - Actor MLP: {actor_hidden_dims} → {num_actions}")
        print(f"  - Critic MLP: {critic_hidden_dims} → 1")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown std type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_fusion_mlp(
        self, input_dim, hidden_dims, activation, use_norm, norm_type
    ):
        """Build fusion MLP (hidden layers only, no output layer)."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_mlp(
        self, input_dim, hidden_dims, output_dim,
        activation, use_norm, norm_type, is_actor=False, output_activation=True
    ):
        """Build MLP with optional normalization."""
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if is_actor and output_activation:
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def _split_obs(self, obs):
        """Split observations into components."""
        object_pc_dim = self.num_points * self.point_dim
        tool_pc_dim = self.num_points * self.point_dim
        hand_state_dim = 9

        # Object cloud
        object_cloud = obs[:, :object_pc_dim].view(-1, self.num_points, self.point_dim)

        # Tool cloud
        tool_start = object_pc_dim
        tool_end = tool_start + tool_pc_dim
        tool_cloud = obs[:, tool_start:tool_end].view(-1, self.num_points, self.point_dim)

        # Hand state
        hand_start = tool_end
        hand_end = hand_start + hand_state_dim
        hand_state = obs[:, hand_start:hand_end]

        # Regular observations
        regular_obs = obs[:, hand_end:]

        return object_cloud, tool_cloud, hand_state, regular_obs

    def _get_fused_features(self, observations):
        """Encode point clouds and fuse with state.

        Point clouds are centered around object center to match pretrain distribution.
        Object center position is fed as context to fusion network.
        """
        object_cloud, tool_cloud, hand_state, regular_obs = self._split_obs(observations)

        batch_size = object_cloud.size(0)

        # Center point clouds around object center (matches pretrain distribution)
        obj_center = object_cloud.mean(dim=1)  # (B, 3)
        object_cloud_centered = object_cloud - obj_center.unsqueeze(1)  # (B, N, 3)
        tool_cloud_centered = tool_cloud - obj_center.unsqueeze(1)  # (B, N, 3)

        # Forward through SDF encoder with centered clouds
        with torch.no_grad() if self.encoder.cfg.freeze else torch.enable_grad():
            res = self.encoder.encode(tool_cloud_centered, object_cloud_centered)

        # Combine all tokens: global_feat + tool_tokens + obj_tokens
        global_feat = res.global_feat.unsqueeze(1)  # (B, 1, D)
        all_tokens = torch.cat([global_feat, res.tool_tokens, res.obj_tokens], dim=1)  # (B, 1+2P, D)

        P = self.encoder.num_patches
        D = self.encoder.feature_dim

        if self.use_sd_cross:
            # Context for SD-Cross: obj_center + hand_state + regular_obs
            ctx = torch.cat([obj_center, hand_state, regular_obs], dim=-1)
            sd_ctx = {'rest': ctx}

            features_parts = []

            # Object-only attention
            if self.num_query_obj > 0 and self.sd_cross_obj is not None:
                obj_features = self.sd_cross_obj(res.obj_tokens, ctx=sd_ctx)
                features_parts.append(obj_features)

            # All tokens attention
            if self.num_query_all > 0 and self.sd_cross_all is not None:
                all_features = self.sd_cross_all(all_tokens, ctx=sd_ctx)
                features_parts.append(all_features)

            # Concatenate
            base_features = torch.cat(features_parts, dim=-1) if features_parts else torch.empty(batch_size, 0, device=observations.device)

            # Add context if configured
            if self.sd_cat_ctx:
                base_features = torch.cat([base_features, ctx], dim=-1)

            fused_features = self.feature_fusion(base_features)
        else:
            # Simple concatenation
            all_tokens_flat = all_tokens.flatten(start_dim=1)
            ctx = torch.cat([obj_center, hand_state, regular_obs], dim=-1)
            raw_features = torch.cat([ctx, all_tokens_flat], dim=-1)
            fused_features = self.feature_fusion(raw_features)

        return fused_features

    def update_distribution(self, observations):
        features = self._get_fused_features(observations)
        mean = self.actor(features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")

        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        features = self._get_fused_features(observations)
        return self.actor(features)

    def evaluate(self, critic_observations, **kwargs):
        features = self._get_fused_features(critic_observations)
        return self.critic(features)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        if self.encoder.cfg.freeze:
            self.encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
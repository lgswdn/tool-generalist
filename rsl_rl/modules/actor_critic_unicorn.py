#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.models.cloud.unicorn import MLPEncoder
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet
from rsl_rl.modules.train.hf_hub import download_ckpt
from rsl_rl.utils import resolve_nn_activation


class ActorCriticUnicorn(nn.Module):
    """
    Actor-Critic network that uses the Unicorn (MLPEncoder) point cloud encoder.

    Observation layout follows ``ActorCriticICP``:
        [point_cloud_flattened, hand_state(9), remaining_obs]
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        pc_point_dim: int = 3,
        pc_num_points: int = 512,
        unicorn_cfg: Optional[Dict[str, Any]] = None,
        unicorn_ckpt: Optional[str] = None,
        freeze_unicorn: bool = True,
        encoder_strict_load: bool = True,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        fusion_hidden_dims=None,
        fusion_use_norm: bool = True,
        fusion_norm_type: Optional[str] = "layer",
        actor_use_norm: bool = True,
        actor_norm_type: Optional[str] = "batch",
        actor_output_activation: bool = False,
        critic_use_norm: bool = False,
        critic_norm_type: Optional[str] = None,
        use_sd_cross: bool = True,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticUnicorn.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.pc_point_dim = pc_point_dim
        self.pc_num_points = pc_num_points
        self.nonpc_obs_dim = num_actor_obs - (pc_point_dim * pc_num_points) - 9
        if self.nonpc_obs_dim < 0:
            raise ValueError(
                f"num_actor_obs ({num_actor_obs}) is smaller than point cloud + hand_state dims "
                f"({pc_point_dim * pc_num_points} + 9)"
            )

        # Build Unicorn encoder
        cfg = MLPEncoder.Config(**(unicorn_cfg or {}))
        self.unicorn_encoder = MLPEncoder(cfg)
        self.unicorn_feature_dim = cfg.model_dim
        self.unicorn_frozen = freeze_unicorn
        
        num_patch_tokens = max(1, self.pc_num_points // cfg.patch_size)
        self.num_tokens = num_patch_tokens + 1  # +1 for embedding token
        
        # Initialize embedding token by default (single token)
        self.embedding_token = nn.Parameter(
            torch.randn(1, 1, self.unicorn_feature_dim) * 0.02,
            requires_grad=True
        )
        self.embedding_token_proj = None  # Will be set if needed when loading weights

        if unicorn_ckpt is not None:
            self._load_unicorn_weights(unicorn_ckpt, strict=encoder_strict_load)

        if freeze_unicorn:
            for param in self.unicorn_encoder.parameters():
                param.requires_grad = False
            self.unicorn_encoder.eval()

        activation_fn = resolve_nn_activation(activation)

        self.use_sd_cross = use_sd_cross
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]

        if self.use_sd_cross:
            if sd_query_keys is None:
                sd_query_keys = ("rest",)
            sd_ctx_dim = self.nonpc_obs_dim
            sd_cfg = StateDependentCrossFeatNet.Config(
                dim_in=(self.num_tokens, self.unicorn_feature_dim),
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=sd_cat_ctx,
            )
            self.state_cross = StateDependentCrossFeatNet(sd_cfg)

            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim
            fusion_input_dim = sd_out_dim
        else:
            fusion_input_dim = self.nonpc_obs_dim + self.unicorn_feature_dim

        self.feature_fusion = self._build_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            activation=activation_fn,
            use_norm=fusion_use_norm,
            norm_type=fusion_norm_type,
        )

        actor_in = fusion_hidden_dims[-1]
        critic_in = fusion_hidden_dims[-1]

        self.actor = self._build_actor_or_critic(
            input_dim=actor_in,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_fn,
            use_norm=actor_use_norm,
            norm_type=actor_norm_type,
            is_actor=True,
            output_activation=actor_output_activation,
        )
        self.critic = self._build_actor_or_critic(
            input_dim=critic_in,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation_fn,
            use_norm=critic_use_norm,
            norm_type=critic_norm_type,
            is_actor=False,
            output_activation=False,
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # --------------------------------------------------------------------- #
    # Helper builders
    # --------------------------------------------------------------------- #
    def _build_mlp(self, input_dim, hidden_dims, activation, use_norm, norm_type):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type is not None:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            layers.append(activation)
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_actor_or_critic(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation,
        use_norm,
        norm_type,
        is_actor,
        output_activation,
    ):
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type is not None:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if is_actor and output_activation:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------- #
    # Observation utilities
    # --------------------------------------------------------------------- #
    def _split_obs(self, obs: torch.Tensor):
        pc_flat = obs[:, : self.pc_num_points * self.pc_point_dim]
        pc = pc_flat.view(-1, self.pc_num_points, self.pc_point_dim)

        hand_state_start = self.pc_num_points * self.pc_point_dim
        hand_state_end = hand_state_start + 9
        hand_state = obs[:, hand_state_start:hand_state_end]

        rest = obs[:, hand_state_end:]
        return pc, hand_state, rest

    # --------------------------------------------------------------------- #
    # Encoder loading helpers
    # --------------------------------------------------------------------- #
    def _load_unicorn_weights(self, path: str, strict: bool = True):
        ckpt_path = path
        if not os.path.exists(ckpt_path) and ":" in ckpt_path:
            repo_id, name = ckpt_path.split(":", maxsplit=1)
            ckpt_path = download_ckpt(repo_id, name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Unicorn checkpoint not found: {path}")

        raw_state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw_state, dict):
            if "model" in raw_state:
                raw_state = raw_state["model"]
            elif "encoder" in raw_state:
                raw_state = raw_state["encoder"]

        target_state = self.unicorn_encoder.state_dict()
        remapped_state = {}
        removed_keys = []
        query_token_value = None
        
        for key, value in raw_state.items():
            new_key = key
            if new_key.startswith("query_encoder."):
                new_key = new_key[len("query_encoder.") :]
            
            # Extract query_token (1D tensor [128])
            if new_key == "query_token" or key == "query_token":
                query_token_value = value
                continue
            
            if new_key in target_state:
                remapped_state[new_key] = value
            else:
                removed_keys.append(key)

        missing, unexpected = self.unicorn_encoder.load_state_dict(remapped_state, strict=False)

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Unable to strictly load Unicorn weights. Missing: {missing}, unexpected: {unexpected}"
            )

        if missing:
            print(f"[ActorCriticUnicorn] Missing keys: {missing}")
        if unexpected:
            print(f"[ActorCriticUnicorn] Unexpected keys: {unexpected}")
        if removed_keys:
            print(
                f"[ActorCriticUnicorn] Ignored {len(removed_keys)} keys that do not match the current encoder (e.g. query/decoder tokens)."
            )
        
        # Load query_token: [128] -> [1, 1, 128]
        query_token_value = query_token_value.unsqueeze(0).unsqueeze(0)
        embedding_dim = query_token_value.shape[2]
        
        # Project to match unicorn_feature_dim if needed
        if embedding_dim != self.unicorn_feature_dim:
            self.embedding_token_proj = nn.Linear(embedding_dim, self.unicorn_feature_dim)
        else:
            self.embedding_token_proj = None
        
        self.embedding_token = nn.Parameter(query_token_value, requires_grad=True)

    # --------------------------------------------------------------------- #
    # Feature extraction
    # --------------------------------------------------------------------- #
    def _encode_point_cloud(self, point_cloud: torch.Tensor):
        ctx = torch.no_grad() if self.unicorn_frozen else torch.enable_grad()
        with ctx:
            # Use embedding token as z_ctx - it will be concatenated with point cloud tokens and participate in self-attention
            batch = point_cloud.shape[0]
            # Get embedding token: [1, 1, emb_dim] -> [B, 1, emb_dim]
            embedding_token = self.embedding_token.expand(batch, -1, -1)
            
            # Project if needed to match unicorn_feature_dim
            if self.embedding_token_proj is not None:
                embedding_token = self.embedding_token_proj(embedding_token)
            
            tokens = self.unicorn_encoder(point_cloud, z_ctx=embedding_token)
        return tokens

    def _get_fused_features(self, observations: torch.Tensor):
        point_cloud, hand_state, rest = self._split_obs(observations)
        tokens = self._encode_point_cloud(point_cloud)  # [B, num_tokens, feature_dim]

        if self.use_sd_cross:
            # StateDependentCrossFeatNet expects [B, num_tokens, feature_dim]
            sd_ctx = {"rest": rest}
            base_features = self.state_cross(tokens, ctx=sd_ctx)
            fused_features = self.feature_fusion(base_features)
            return fused_features

        features = tokens.reshape(tokens.shape[0], -1)  # [B, num_tokens * feature_dim]
        if rest.shape[-1] > 0:
            raw_features = torch.cat([rest, features], dim=-1)
        else:
            raw_features = features
        fused_features = self.feature_fusion(raw_features)
        return fused_features

    # --------------------------------------------------------------------- #
    # RL interface
    # --------------------------------------------------------------------- #
    def update_distribution(self, observations: torch.Tensor):
        fused = self._get_fused_features(observations)
        mean = self.actor(fused)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def reset(self, dones=None):
        # Placeholder to match PPO interface; nothing to reset yet.
        pass

    def act_inference(self, observations: torch.Tensor):
        fused = self._get_fused_features(observations)
        return self.actor(fused)

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        fused = self._get_fused_features(critic_observations)
        return self.critic(fused)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.unicorn_frozen:
            self.unicorn_encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


class ActorCriticMultiUnicorn(nn.Module):
    """
    Multi-cloud variant of the Unicorn policy.

    Expected observation layout:
        [object_cloud, stacked_obstacle_clouds, remaining_obs]

    - Object cloud: the primary object point cloud.
    - All obstacles: concatenated obstacle clouds, uniformly downsampled.
    - Closest obstacle: the obstacle whose centroid is closest to the object.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        pc_point_dim: int = 3,
        object_pc_num_points: int = 512,
        obstacle_pc_num_points: int = 512,
        num_obstacles: int = 2,
        all_obstacles_num_points: int = 512,
        closest_obstacle_num_points: int = 512,
        unicorn_cfg: Optional[Dict[str, Any]] = None,
        unicorn_ckpt: Optional[str] = None,
        freeze_unicorn: bool = True,
        encoder_strict_load: bool = True,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        fusion_hidden_dims=None,
        fusion_use_norm: bool = True,
        fusion_norm_type: Optional[str] = "layer",
        actor_use_norm: bool = True,
        actor_norm_type: Optional[str] = "batch",
        actor_output_activation: bool = False,
        critic_use_norm: bool = False,
        critic_norm_type: Optional[str] = None,
        use_sd_cross: bool = True,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMultiUnicorn.__init__ got unexpected arguments, ignoring: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.pc_point_dim = pc_point_dim
        self.object_pc_num_points = object_pc_num_points
        self.obstacle_pc_num_points = obstacle_pc_num_points
        self.num_obstacles = num_obstacles
        self.all_obstacles_num_points = all_obstacles_num_points
        self.closest_obstacle_num_points = closest_obstacle_num_points

        self.object_pc_dim = self.object_pc_num_points * pc_point_dim
        self.obstacles_pc_dim = (
            self.num_obstacles * self.obstacle_pc_num_points * self.pc_point_dim
        )

        total_pc_dim = self.object_pc_dim + self.obstacles_pc_dim
        self.nonpc_obs_dim = num_actor_obs - total_pc_dim
        if self.nonpc_obs_dim < 0:
            raise ValueError(
                "num_actor_obs is too small for the configured point clouds "
                f"(need at least {total_pc_dim}, got {num_actor_obs})."
            )
        self.ctx_dim = self.nonpc_obs_dim

        cfg = MLPEncoder.Config(**(unicorn_cfg or {}))
        self.unicorn_encoder = MLPEncoder(cfg)
        self.unicorn_feature_dim = cfg.model_dim
        self.unicorn_frozen = freeze_unicorn
        
        # Initialize embedding token by default (single token)
        self.embedding_token = nn.Parameter(
            torch.randn(1, 1, self.unicorn_feature_dim) * 0.02,
            requires_grad=True
        )
        self.embedding_token_proj = None  # Will be set if needed when loading weights

        if unicorn_ckpt is not None:
            self._load_unicorn_weights(unicorn_ckpt, strict=encoder_strict_load)

        if freeze_unicorn:
            for param in self.unicorn_encoder.parameters():
                param.requires_grad = False
            self.unicorn_encoder.eval()

        self.object_num_tokens = max(1, self.object_pc_num_points // cfg.patch_size)
        self.obstacles_num_tokens = max(1, self.all_obstacles_num_points // cfg.patch_size)
        self.closest_num_tokens = max(1, self.closest_obstacle_num_points // cfg.patch_size)
        self.total_num_tokens = (
            self.object_num_tokens + self.obstacles_num_tokens + self.closest_num_tokens
        )

        self.object_type_token = nn.Parameter(torch.zeros(1, 1, self.unicorn_feature_dim))
        self.obstacles_type_token = nn.Parameter(torch.zeros(1, 1, self.unicorn_feature_dim))
        self.closest_type_token = nn.Parameter(torch.zeros(1, 1, self.unicorn_feature_dim))
        nn.init.normal_(self.object_type_token, std=0.02)
        nn.init.normal_(self.obstacles_type_token, std=0.02)
        nn.init.normal_(self.closest_type_token, std=0.02)

        activation_fn = resolve_nn_activation(activation)
        self.use_sd_cross = use_sd_cross
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]

        if self.use_sd_cross:
            if sd_query_keys is None:
                sd_query_keys = ("rest",)
            
            # First 1/2 attend to object only, rest attend to all tokens
            num_query_object = 4  # At least 1 query token for object
            num_query_all = sd_num_query - num_query_object
            
            print(f'[ActorCriticMultiUnicorn] unicorn_feature_dim: {self.unicorn_feature_dim}, total_num_tokens: {self.total_num_tokens}, object_num_tokens: {self.object_num_tokens}, sd_emb_dim: {sd_emb_dim}')
            print(f'  - Query tokens: {num_query_object} for object-only, {num_query_all} for all tokens')
            
            # StateDependentCrossFeatNet for object-only attention (first 1/4 query tokens)
            if num_query_object > 0:
                sd_cfg_object = StateDependentCrossFeatNet.Config(
                    dim_in=(self.object_num_tokens, self.unicorn_feature_dim),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_object,
                    ctx_dim=self.ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,  # Don't cat ctx here, will cat at the end
                )
                self.state_cross_encoder_object = StateDependentCrossFeatNet(sd_cfg_object)
            else:
                self.state_cross_encoder_object = None
            
            # StateDependentCrossFeatNet for all tokens attention (remaining query tokens)
            if num_query_all > 0:
                sd_cfg_all = StateDependentCrossFeatNet.Config(
                    dim_in=(self.total_num_tokens, self.unicorn_feature_dim),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_all,
                    ctx_dim=self.ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,  # Don't cat ctx here, will cat at the end
                )
                self.state_cross_encoder_all = StateDependentCrossFeatNet(sd_cfg_all)
            else:
                self.state_cross_encoder_all = None
            
            # Calculate output dimension
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += self.ctx_dim
            
            fusion_input_dim = sd_out_dim
            
            # Store configuration for later use
            self.num_query_object = num_query_object
            self.num_query_all = num_query_all
            self.sd_cat_ctx = sd_cat_ctx  # Store for later use in _get_fused_features
        else:
            fusion_input_dim = self.ctx_dim + self.total_num_tokens * self.unicorn_feature_dim

        self.feature_fusion = self._build_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            activation=activation_fn,
            use_norm=fusion_use_norm,
            norm_type=fusion_norm_type,
        )

        actor_in = fusion_hidden_dims[-1]
        critic_in = fusion_hidden_dims[-1]

        self.actor = self._build_actor_or_critic(
            input_dim=actor_in,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_fn,
            use_norm=actor_use_norm,
            norm_type=actor_norm_type,
            is_actor=True,
            output_activation=actor_output_activation,
        )
        self.critic = self._build_actor_or_critic(
            input_dim=critic_in,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation_fn,
            use_norm=critic_use_norm,
            norm_type=critic_norm_type,
            is_actor=False,
            output_activation=False,
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be either 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ----------------------------- Helper Builders ----------------------------- #
    def _build_mlp(self, input_dim, hidden_dims, activation, use_norm, norm_type):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type is not None:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            layers.append(activation)
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_actor_or_critic(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation,
        use_norm,
        norm_type,
        is_actor,
        output_activation,
    ):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm and norm_type is not None:
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if is_actor and output_activation:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    # ----------------------------- Encoder helpers ----------------------------- #
    def _load_unicorn_weights(self, path: str, strict: bool = True):
        ckpt_path = path
        if not os.path.exists(ckpt_path) and ":" in ckpt_path:
            repo_id, name = ckpt_path.split(":", maxsplit=1)
            ckpt_path = download_ckpt(repo_id, name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Unicorn checkpoint not found: {path}")

        raw_state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw_state, dict):
            if "model" in raw_state:
                raw_state = raw_state["model"]
            elif "encoder" in raw_state:
                raw_state = raw_state["encoder"]

        target_state = self.unicorn_encoder.state_dict()
        remapped_state = {}
        removed_keys = []
        query_token_value = None
        
        for key, value in raw_state.items():
            new_key = key
            if new_key.startswith("query_encoder."):
                new_key = new_key[len("query_encoder.") :]
            
            # Extract query_token (1D tensor [128])
            if new_key == "query_token" or key == "query_token":
                query_token_value = value
                continue
            
            if new_key in target_state:
                remapped_state[new_key] = value
            else:
                removed_keys.append(key)

        missing, unexpected = self.unicorn_encoder.load_state_dict(remapped_state, strict=False)

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Unable to strictly load Unicorn weights. Missing: {missing}, unexpected: {unexpected}"
            )

        if missing:
            print(f"[ActorCriticMultiUnicorn] Missing keys: {missing}")
        if unexpected:
            print(f"[ActorCriticMultiUnicorn] Unexpected keys: {unexpected}")
        if removed_keys:
            print(
                f"[ActorCriticMultiUnicorn] Ignored {len(removed_keys)} keys that do not match the current encoder (e.g., query/decoder tokens)."
            )
        
        # Load query_token: [128] -> [1, 1, 128]
        query_token_value = query_token_value.unsqueeze(0).unsqueeze(0)
        embedding_dim = query_token_value.shape[2]
        
        # Project to match unicorn_feature_dim if needed
        if embedding_dim != self.unicorn_feature_dim:
            self.embedding_token_proj = nn.Linear(embedding_dim, self.unicorn_feature_dim)
        else:
            self.embedding_token_proj = None
        
        self.embedding_token = nn.Parameter(query_token_value, requires_grad=True)

    def _encode_point_cloud(self, point_cloud: torch.Tensor):
        ctx = torch.no_grad() if self.unicorn_frozen else torch.enable_grad()
        with ctx:
            # Use embedding token as z_ctx - it will be concatenated with point cloud tokens and participate in self-attention
            batch = point_cloud.shape[0]
            # Get embedding token: [1, 1, emb_dim] -> [B, 1, emb_dim]
            embedding_token = self.embedding_token.expand(batch, -1, -1)
            
            # Project if needed to match unicorn_feature_dim
            if self.embedding_token_proj is not None:
                embedding_token = self.embedding_token_proj(embedding_token)
            
            z_ctx = embedding_token
            
            tokens = self.unicorn_encoder(point_cloud, z_ctx=z_ctx)
        return tokens

    # ----------------------------- Observation utils ----------------------------- #
    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        start = 0
        end = start + self.object_pc_dim
        object_cloud = obs[:, start:end].view(-1, self.object_pc_num_points, self.pc_point_dim)

        start = end
        end = start + self.obstacles_pc_dim
        if self.num_obstacles > 0:
            obstacles_clouds = obs[:, start:end].view(
                -1, self.num_obstacles, self.obstacle_pc_num_points, self.pc_point_dim
            )
        else:
            obstacles_clouds = None

        rest = obs[:, end:]
        return object_cloud, obstacles_clouds, rest

    def _sample_points(self, points: torch.Tensor, target: int):
        if target <= 0:
            raise ValueError("target number of points must be positive")

        batch, num_points, _ = points.shape
        if num_points == target:
            return points
        if num_points > target:
            idx = torch.linspace(0, num_points - 1, target, device=points.device)
            idx = idx.round().long()
            sampled = points.index_select(1, idx)
            return sampled

        repeats = (target + num_points - 1) // num_points
        expanded = points.repeat(1, repeats, 1)
        return expanded[:, :target, :]

    def _merge_obstacles(
        self, obstacles_clouds: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        merged = obstacles_clouds.reshape(batch_size, -1, self.pc_point_dim)
        return self._sample_points(merged, self.all_obstacles_num_points)

    def _select_closest_obstacle(
        self, object_cloud: torch.Tensor, obstacles_clouds: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch = object_cloud.shape[0]
        device = object_cloud.device
        if obstacles_clouds is None or self.num_obstacles == 0:
            return torch.zeros(
                batch, self.closest_obstacle_num_points, self.pc_point_dim, device=device
            )

        object_center = object_cloud.mean(dim=1, keepdim=True)  # [B,1,3]
        obstacle_centers = obstacles_clouds.mean(dim=2)  # [B, num_obstacles, 3]
        distances = torch.norm(obstacle_centers - object_center, dim=-1)  # [B, num_obstacles]
        closest_indices = distances.argmin(dim=-1)  # [B]
        batch_indices = torch.arange(batch, device=device)
        closest_cloud = obstacles_clouds[batch_indices, closest_indices]  # [B, P, 3]
        return self._sample_points(closest_cloud, self.closest_obstacle_num_points)

    def _encode_all_clouds(
        self, object_cloud: torch.Tensor, obstacles_clouds: Optional[torch.Tensor]
    ):
        batch_size = object_cloud.shape[0]
        device = object_cloud.device
        all_obstacles_cloud = self._merge_obstacles(obstacles_clouds, batch_size, device)
        closest_cloud = self._select_closest_obstacle(object_cloud, obstacles_clouds)
        tokens = []
        for cloud, token in (
            (object_cloud, self.object_type_token),
            (all_obstacles_cloud, self.obstacles_type_token),
            (closest_cloud, self.closest_type_token),
        ):
            cloud_tokens = self._encode_point_cloud(cloud) + token
            tokens.append(cloud_tokens)
        return torch.cat(tokens, dim=-2)

    # ----------------------------- Feature extraction ----------------------------- #
    def _get_fused_features(self, observations: torch.Tensor):
        object_cloud, obstacles_clouds, rest = self._split_obs(observations)
        context_obs = rest
        tokens = self._encode_all_clouds(object_cloud, obstacles_clouds)

        if self.use_sd_cross:
            sd_ctx = {"rest": context_obs}
            
            # Extract object-only tokens (first object_num_tokens tokens)
            object_tokens = tokens[:, :self.object_num_tokens, :]  # [B, object_num_tokens, feature_dim]
            
            # Differentiated attention: first 1/4 query tokens attend to object only, rest attend to all
            base_features_parts = []
            
            # First 1/4 query tokens attend to object-only tokens
            if self.num_query_object > 0 and self.state_cross_encoder_object is not None:
                object_features = self.state_cross_encoder_object(object_tokens, ctx=sd_ctx)
                base_features_parts.append(object_features)
            
            # Remaining query tokens attend to all tokens
            if self.num_query_all > 0 and self.state_cross_encoder_all is not None:
                all_features = self.state_cross_encoder_all(tokens, ctx=sd_ctx)
                base_features_parts.append(all_features)
            
            # Concatenate all features
            base_features = torch.cat(base_features_parts, dim=-1) if base_features_parts else torch.empty(tokens.shape[0], 0, device=tokens.device)
            
            # Add ctx if needed (from original sd_cat_ctx config)
            if hasattr(self, 'sd_cat_ctx') and self.sd_cat_ctx:
                base_features = torch.cat([base_features, context_obs], dim=-1)
            
            fused_features = self.feature_fusion(base_features)
            return fused_features

        tokens_flat = tokens.flatten(start_dim=1)
        if context_obs.shape[-1] > 0:
            raw_features = torch.cat([context_obs, tokens_flat], dim=-1)
        else:
            raw_features = tokens_flat
        fused_features = self.feature_fusion(raw_features)
        return fused_features

    # ----------------------------- RL interface ----------------------------- #
    def update_distribution(self, observations: torch.Tensor):
        fused = self._get_fused_features(observations)
        mean = self.actor(fused)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def reset(self, dones=None):
        pass

    def act_inference(self, observations: torch.Tensor):
        fused = self._get_fused_features(observations)
        return self.actor(fused)

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        fused = self._get_fused_features(critic_observations)
        return self.critic(fused)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.unicorn_frozen:
            self.unicorn_encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


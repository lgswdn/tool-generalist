# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic network using Point2Vec encoder with TG shared policy heads."""

from __future__ import annotations

import copy
import os
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder
from point2vec.utils import transforms
from point2vec.utils.checkpoint import extract_model_checkpoint
from rsl_rl.modules.tg_policy_common import (
    ObservationLayout,
    build_context_vector,
    build_fusion_mlp,
    build_mlp,
    build_state_cross_attention,
    center_clouds_by_bbox,
    context_dim,
    split_observations,
)
from rsl_rl.utils import resolve_nn_activation


class ActorCriticPoint2Vec(nn.Module):
    """Point2Vec encoder plugged into the same learnable head as ActorCriticTG.

    Observations keep env-frame point clouds.  The actor subtracts object/tool
    mesh AABB centers before sending each cloud to Point2Vec and preserves the
    bbox centers in the shared context vector.
    """

    is_recurrent = False
    tracks_encoder_feature_calls = True
    supports_cached_features = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        point_dim: int = 3,
        num_points: int = 512,
        encoder_weights_path: Optional[str] = None,
        point2vec_ckpt_path: Optional[str] = None,
        freeze_encoder: bool = True,
        freeze_point2vec: Optional[bool] = None,
        separate_actor_critic_fusion: bool = False,
        tokenizer_num_groups: int = 128,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: Optional[float] = None,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        train_transformations: Optional[list[str]] = None,
        val_transformations: Optional[list[str]] = None,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: Optional[tuple[str, ...]] = None,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_ff_dim: Optional[int] = None,
        cross_attn_dropout: float = 0.0,
        fusion_hidden_dims=(512, 256, 128),
        actor_hidden_dims=(64,),
        critic_hidden_dims=(128,),
        hand_state_dim: int = 9,
        robot_state_dim: int = 14,
        previous_action_dim: Optional[int] = None,
        relative_goal_dim: int = 9,
        object_velocity_dim: int = 0,
        physics_dim: int = 7,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "ActorCriticPoint2Vec.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if point_dim != 3:
            raise ValueError(f"Point2Vec requires point_dim=3, got {point_dim}")
        if encoder_weights_path and point2vec_ckpt_path:
            if os.path.abspath(os.path.expanduser(encoder_weights_path)) != os.path.abspath(
                os.path.expanduser(point2vec_ckpt_path)
            ):
                raise ValueError("encoder_weights_path and point2vec_ckpt_path must match")
        checkpoint_path = encoder_weights_path or point2vec_ckpt_path
        if not checkpoint_path:
            raise ValueError("ActorCriticPoint2Vec requires encoder_weights_path")
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Point2Vec checkpoint does not exist: {checkpoint_path}")

        self.point_dim = int(point_dim)
        self.num_points = int(num_points)
        self.num_actions = int(num_actions)
        self.noise_std_type = noise_std_type
        self.freeze_point2vec = bool(freeze_encoder if freeze_point2vec is None else freeze_point2vec)
        self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)
        self.previous_action_dim = int(previous_action_dim) if previous_action_dim is not None else int(num_actions)
        self.object_velocity_dim = int(object_velocity_dim)
        self.physics_dim = int(physics_dim)
        self.sd_cat_ctx = bool(sd_cat_ctx)
        self.total_num_tokens = 2 * int(tokenizer_num_groups)
        self.token_dim = int(encoder_dim)
        self.cross_attn_heads = int(cross_attn_heads)
        self.cross_attn_layers = int(cross_attn_layers)
        self.cross_attn_ff_dim = cross_attn_ff_dim
        self.cross_attn_dropout = float(cross_attn_dropout)

        self.obs_layout = ObservationLayout.build(
            num_points=self.num_points,
            point_dim=self.point_dim,
            hand_state_dim=hand_state_dim,
            robot_state_dim=robot_state_dim,
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=relative_goal_dim,
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        if num_actor_obs != self.obs_layout.total_dim:
            raise ValueError(
                "ActorCriticPoint2Vec observation layout mismatch: "
                f"num_actor_obs={num_actor_obs}, expected={self.obs_layout.total_dim}"
            )
        if num_critic_obs != num_actor_obs:
            raise ValueError(
                "ActorCriticPoint2Vec expects critic observations to use the same named layout "
                f"as actor observations, got num_critic_obs={num_critic_obs}, "
                f"num_actor_obs={num_actor_obs}"
            )

        def build_transformation(name: str) -> transforms.Transform:
            if name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            if name == "scale":
                return transforms.PointcloudScaling(min=0.8, max=1.2)
            if name == "rotate":
                return transforms.PointcloudRotation(dims=[1], deg=None)
            if name == "translate":
                return transforms.PointcloudTranslation(0.2)
            raise RuntimeError(
                f"Unsupported Point2Vec transform {name!r}; clouds are already bbox-centered"
            )

        train_transformations = train_transformations or ["unit_sphere"]
        val_transformations = val_transformations or ["unit_sphere"]
        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )
        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )
        dpr = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )
        self._load_pretrained_checkpoint(checkpoint_path)

        if self.freeze_point2vec:
            for module in (self.tokenizer, self.positional_encoding, self.encoder):
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

        activation_fn = resolve_nn_activation(activation)
        ctx_dim = context_dim(
            hand_state_dim=hand_state_dim,
            robot_state_dim=robot_state_dim,
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=relative_goal_dim,
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        self.context_dim = ctx_dim
        if sd_query_keys is None:
            sd_query_keys = ("context",)
        self.state_cross_all, fusion_input_dim = build_state_cross_attention(
            total_num_tokens=self.total_num_tokens,
            token_dim=self.token_dim,
            ctx_dim=ctx_dim,
            sd_num_query=sd_num_query,
            sd_emb_dim=sd_emb_dim,
            sd_cat_query=sd_cat_query,
            sd_query_keys=tuple(sd_query_keys),
        )
        if self.sd_cat_ctx:
            fusion_input_dim += ctx_dim

        self.fusion_mlp = build_fusion_mlp(fusion_input_dim, fusion_hidden_dims, activation_fn)
        self.critic_state_cross_all = None
        self.critic_fusion_mlp = None
        if self.separate_actor_critic_fusion:
            self.critic_state_cross_all = copy.deepcopy(self.state_cross_all)
            self.critic_fusion_mlp = copy.deepcopy(self.fusion_mlp)
        fusion_out_dim = int(fusion_hidden_dims[-1]) if fusion_hidden_dims else int(fusion_input_dim)
        self.actor = build_mlp(fusion_out_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = build_mlp(fusion_out_dim, critic_hidden_dims, activation_fn, 1)

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _load_pretrained_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = extract_model_checkpoint(ckpt_path)
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if state_dict is None and isinstance(checkpoint, dict):
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Point2Vec checkpoint has no state_dict payload: {ckpt_path}")
        normalized = self._normalize_checkpoint_keys(state_dict)
        self.tokenizer.load_state_dict(self._submodule_state(normalized, "tokenizer.", ckpt_path), strict=True)
        self.positional_encoding.load_state_dict(
            self._submodule_state(normalized, "positional_encoding.", ckpt_path),
            strict=True,
        )
        self.encoder.load_state_dict(self._submodule_state(normalized, "encoder.", ckpt_path), strict=True)

    @staticmethod
    def _normalize_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            clean = str(key)
            for prefix in ("module.", "model.", "point2vec."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            normalized[clean] = value
        return normalized

    @staticmethod
    def _submodule_state(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        ckpt_path: str,
    ) -> dict[str, torch.Tensor]:
        selected = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise RuntimeError(f"Point2Vec checkpoint missing {prefix!r} keys: {ckpt_path}")
        return selected

    def _split_observations(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_observations(obs, self.obs_layout)

    def _encode_single_cloud(self, pointcloud: torch.Tensor, *, is_training: bool) -> torch.Tensor:
        transform = self.train_transformations if is_training else self.val_transformations
        pointcloud = transform(pointcloud)
        tokens, centers = self.tokenizer(pointcloud)
        pos_embeddings = self.positional_encoding(centers)
        with torch.set_grad_enabled(not self.freeze_point2vec):
            output = self.encoder(tokens, pos_embeddings, return_hidden_states=False)
        return output.last_hidden_state

    def _tokenize(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self._split_observations(observations)
        object_cloud_rel, tool_cloud_rel = center_clouds_by_bbox(
            parts["object_cloud"],
            parts["tool_cloud"],
            parts["object_bbox_center"],
            parts["tool_bbox_center"],
        )
        is_training = self.training and not self.freeze_point2vec
        if self.freeze_point2vec:
            with torch.no_grad():
                object_tokens = self._encode_single_cloud(object_cloud_rel, is_training=is_training)
                tool_tokens = self._encode_single_cloud(tool_cloud_rel, is_training=is_training)
        else:
            object_tokens = self._encode_single_cloud(object_cloud_rel, is_training=is_training)
            tool_tokens = self._encode_single_cloud(tool_cloud_rel, is_training=is_training)
        all_tokens = torch.cat([tool_tokens, object_tokens], dim=1)
        ctx_vec = build_context_vector(parts)
        return all_tokens, ctx_vec

    def _features_from_tokens_context(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
        *,
        branch: str = "actor",
    ) -> torch.Tensor:
        critic_branch = branch == "critic" and self.separate_actor_critic_fusion
        state_cross_all = self.critic_state_cross_all if critic_branch else self.state_cross_all
        fusion_mlp = self.critic_fusion_mlp if critic_branch else self.fusion_mlp
        if state_cross_all is None or fusion_mlp is None:
            raise RuntimeError("critic fusion requested before critic fusion module was initialized")
        sd_out = state_cross_all(all_tokens, ctx={"context": ctx_vec}, mask=None)
        if self.sd_cat_ctx:
            sd_out = torch.cat([sd_out, ctx_vec], dim=-1)
        return fusion_mlp(sd_out)

    def _get_features(self, observations: torch.Tensor, *, branch: str = "actor") -> torch.Tensor:
        all_tokens, ctx_vec = self._tokenize(observations)
        return self._features_from_tokens_context(all_tokens, ctx_vec, branch=branch)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(self._get_features(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, torch.clamp(std, min=1e-6))

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        return self.actor(self._get_features(observations))

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(self._get_features(critic_observations, branch="critic"))

    def reset(self, dones=None):
        pass

    def get_cached_encoder_features(self, observations: torch.Tensor):
        return self._tokenize(observations)

    def act_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, torch.clamp(std, min=1e-6))
        return self.distribution.sample()

    def evaluate_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        return self.critic(self._features_from_tokens_context(all_tokens, ctx_vec, branch="critic"))

    def get_actions_log_prob_from_cached_features(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        return self.actor(self._features_from_tokens_context(all_tokens, ctx_vec))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_point2vec:
            self.tokenizer.eval()
            self.positional_encoding.eval()
            self.encoder.eval()
        return self

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

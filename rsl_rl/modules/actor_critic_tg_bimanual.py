# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual Tool-Generalist actor-critic with a 3-stream TCE encoder."""

from __future__ import annotations

import copy
from typing import Any, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from rsl_rl.modules.actor_critic_tg import ActorCriticTG, _make_pretrain_style_mlp
from rsl_rl.modules.tg_bimanual_policy_common import (
    BimanualObservationLayout,
    bimanual_context_dim,
    build_bimanual_context_vector,
    center_bimanual_clouds_by_bbox,
    split_bimanual_observations,
)
from rsl_rl.modules.tg_policy_common import build_fusion_mlp, build_mlp
from rsl_rl.utils import resolve_nn_activation


class BimanualTCEEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool1_patch_idx: torch.Tensor
    tool2_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool1_patch_centers: torch.Tensor
    tool2_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class BimanualTCEPointCloudEncoder(TCEPointCloudEncoder):
    """TCE encoder variant with two tool streams and one object stream."""

    def __init__(self, cfg: TCEPointCloudEncoderCfg):
        super().__init__(cfg)
        self.type_embed = nn.Parameter(torch.zeros(3, cfg.encoder_channel))
        nn.init.normal_(self.type_embed, std=0.02)
        if cfg.freeze:
            for param in self.parameters():
                param.requires_grad_(False)

    def encode(
        self,
        tool1_pc: torch.Tensor,
        tool2_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> BimanualTCEEncodeResult:
        tool1_tok, tool1_idx, tool1_centers = self._encode_one(tool1_pc, type_id=0)
        tool2_tok, tool2_idx, tool2_centers = self._encode_one(tool2_pc, type_id=1)
        obj_tok, obj_idx, obj_centers = self._encode_one(obj_pc, type_id=2)
        tool1_tok, tool1_idx, tool1_centers = self._pad_to_num_patches(tool1_tok, tool1_idx, tool1_centers)
        tool2_tok, tool2_idx, tool2_centers = self._pad_to_num_patches(tool2_tok, tool2_idx, tool2_centers)
        obj_tok, obj_idx, obj_centers = self._pad_to_num_patches(obj_tok, obj_idx, obj_centers)
        fused = torch.cat((tool1_tok, tool2_tok, obj_tok), dim=1)
        cls = self.cls_token.expand(fused.shape[0], -1, -1)
        fused = torch.cat((cls, fused), dim=1)
        for block in self.vit:
            fused = block(fused)
        fused = self.norm(fused)
        fused = fused[:, 1:, :]
        return BimanualTCEEncodeResult(
            fused_tokens=fused,
            tool1_patch_idx=tool1_idx,
            tool2_patch_idx=tool2_idx,
            obj_patch_idx=obj_idx,
            tool1_patch_centers=tool1_centers,
            tool2_patch_centers=tool2_centers,
            obj_patch_centers=obj_centers,
        )

    def forward(
        self,
        tool1_pc: torch.Tensor,
        tool2_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> BimanualTCEEncodeResult:
        return self.encode(tool1_pc, tool2_pc, obj_pc)


class BimanualRelativeContextCrossAttention(nn.Module):
    """Cross-attention with shared per-tool 3D relative-translation queries.

    The original single-tool pretrain ``query_A`` maps a 3D
    ``tool_bbox_center - object_bbox_center`` vector to query tokens.  Bimanual
    reuse keeps that semantic contract by applying the same ``query_A`` to
    tool1-object and tool2-object translations independently, then concatenating
    both query groups before attending over all 3-stream TCE tokens.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        ctx_dim: int,
        total_query_tokens: int,
        relative_translation_query_tokens: int,
        n_heads: int,
        n_layers: int,
        condition_mlp_hidden_dims: tuple[int, ...] = (128, 128),
        cat_query: bool = False,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.total_query_tokens = int(total_query_tokens)
        self.relative_translation_query_tokens = int(relative_translation_query_tokens)
        self.relative_query_tokens = 2 * self.relative_translation_query_tokens
        self.context_query_tokens = self.total_query_tokens - self.relative_query_tokens
        self.cat_query = bool(cat_query)
        if self.total_query_tokens <= 0:
            raise ValueError("sd_num_query must be > 0")
        if self.relative_translation_query_tokens < 0:
            raise ValueError("relative_translation_query_tokens must be >= 0")
        if self.relative_query_tokens > self.total_query_tokens:
            raise ValueError(
                "2 * relative_translation_query_tokens must be <= sd_num_query "
                "for bimanual per-tool 3D queries"
            )

        self.query_A = (
            _make_pretrain_style_mlp(
                (3,) + tuple(condition_mlp_hidden_dims) + (self.relative_translation_query_tokens * self.token_dim,)
            )
            if self.relative_translation_query_tokens > 0
            else None
        )
        self.context_query = (
            _make_pretrain_style_mlp(
                (int(ctx_dim),) + tuple(condition_mlp_hidden_dims) + (self.context_query_tokens * self.token_dim,)
            )
            if self.context_query_tokens > 0
            else None
        )
        self.layers = nn.ModuleList()
        for _ in range(int(n_layers)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "query_cross_attn": nn.MultiheadAttention(
                            embed_dim=self.token_dim,
                            num_heads=int(n_heads),
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(self.token_dim),
                        "norm2": nn.LayerNorm(self.token_dim),
                        "ff": nn.Sequential(
                            nn.Linear(self.token_dim, self.token_dim * 4),
                            nn.GELU(),
                            nn.Linear(self.token_dim * 4, self.token_dim),
                        ),
                    }
                )
            )

    @property
    def output_dim(self) -> int:
        dim = self.total_query_tokens * self.token_dim
        if self.cat_query:
            dim += self.total_query_tokens * self.token_dim
        return dim

    def forward(self, tokens: torch.Tensor, tool_object_rel: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        queries = []
        if self.query_A is not None:
            tool_object_rel = tool_object_rel.view(batch, 2, 3)
            rel_queries = self.query_A(tool_object_rel.reshape(batch * 2, 3))
            queries.append(rel_queries.view(batch, self.relative_query_tokens, self.token_dim))
        if self.context_query is not None:
            queries.append(self.context_query(ctx_vec).view(batch, self.context_query_tokens, self.token_dim))
        query = torch.cat(queries, dim=1)
        initial_query = query
        for layer in self.layers:
            residual = query
            query_norm = layer["norm1"](query)
            attn_out, _ = layer["query_cross_attn"](query=query_norm, key=tokens, value=tokens)
            query = residual + attn_out
            query = query + layer["ff"](layer["norm2"](query))
        out = query.reshape(batch, -1)
        if self.cat_query:
            out = torch.cat((out, initial_query.reshape(batch, -1)), dim=-1)
        return out

    def strict_load_pretrain_pose_cross_attn(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ActorCriticTG._checkpoint_state_dict(ckpt, checkpoint_path)
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        missing: list[str] = []
        mismatched: list[str] = []

        def copy_param(source_name: str, target_tensor: torch.Tensor) -> None:
            source_tensor = state.get(source_name)
            if source_tensor is None:
                missing.append(source_name)
                return
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                mismatched.append(
                    f"{source_name}: checkpoint={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
                )
                return
            with torch.no_grad():
                target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))

        if self.query_A is not None:
            for name, tensor in self.query_A.state_dict().items():
                copy_param(f"pose_cross_attn.query_generator.query_A.{name}", tensor)
        for layer_i, layer in enumerate(self.layers):
            for module_name in ("query_cross_attn", "norm1", "norm2", "ff"):
                module = layer[module_name]
                for name, tensor in module.state_dict().items():
                    copy_param(f"pose_cross_attn.layers.{layer_i}.{module_name}.{name}", tensor)
        if missing or mismatched:
            raise RuntimeError(
                "reuse_pretrain_pose_cross_attn=True but pretrain pose_cross_attn "
                "weights are incompatible with bimanual per-tool 3D fusion. "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}; "
                f"mismatched={mismatched[:8]}{'...' if len(mismatched) > 8 else ''}"
            )


class ActorCriticTGBimanual(nn.Module):
    """Actor-critic network for bimanual tool/object point clouds.

    Observation layout:
        object_cloud | tool1_cloud | tool2_cloud |
        object_bbox_center | tool1_bbox_center | tool2_bbox_center |
        hand1_state | hand2_state | robot1_state | robot2_state |
        previous_action | relative_goal_pose | object_velocity | physics
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
        num_points: int = 512,
        point_dim: int = 3,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 12,
        vit_heads: int = 4,
        encoder_weights_path: Optional[str] = None,
        freeze_encoder: bool = True,
        separate_actor_critic_fusion: bool = False,
        use_learnable_query_tokens: bool = False,
        sd_num_query: int = 16,
        sd_num_query_object: Optional[int] = None,
        sd_emb_dim: int = 128,
        relative_translation_query_tokens: int = 2,
        reuse_pretrain_pose_cross_attn: bool = False,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys: Optional[tuple] = None,
        num_query_tokens: int = 16,
        num_query_object_tokens: Optional[int] = None,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_ff_dim: Optional[int] = None,
        cross_attn_dropout: float = 0.0,
        fusion_hidden_dims=(512, 256, 128),
        actor_hidden_dims=(64,),
        critic_hidden_dims=(128,),
        hand_state_dim: int = 18,
        robot_state_dim: int = 28,
        previous_action_dim: Optional[int] = None,
        relative_goal_dim: int = 9,
        object_velocity_dim: int = 0,
        physics_dim: int = 7,
        model_input_centering: str = "bbox_center",
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTGBimanual.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if not encoder_weights_path:
            raise ValueError("ActorCriticTGBimanual requires encoder_weights_path")
        if hand_state_dim % 2 != 0:
            raise ValueError("ActorCriticTGBimanual hand_state_dim must be even")
        if robot_state_dim % 2 != 0:
            raise ValueError("ActorCriticTGBimanual robot_state_dim must be even")
        self.point_dim = int(point_dim)
        self.num_points = int(num_points)
        self.num_actions = int(num_actions)
        self.noise_std_type = str(noise_std_type)
        self.freeze_encoder = bool(freeze_encoder)
        self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)
        self.model_input_centering = str(model_input_centering)
        if self.model_input_centering not in {"bbox_center", "object_center"}:
            raise ValueError("ActorCriticTGBimanual model_input_centering must be bbox_center or object_center")
        self.previous_action_dim = int(previous_action_dim) if previous_action_dim is not None else int(num_actions)
        self.object_velocity_dim = int(object_velocity_dim)
        self.physics_dim = int(physics_dim)

        self.obs_layout = BimanualObservationLayout.build(
            num_points=self.num_points,
            point_dim=self.point_dim,
            hand_state_dim=int(hand_state_dim),
            robot_state_dim=int(robot_state_dim),
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=int(relative_goal_dim),
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        if num_actor_obs != self.obs_layout.total_dim:
            raise ValueError(
                "ActorCriticTGBimanual observation layout mismatch: "
                f"num_actor_obs={num_actor_obs}, expected={self.obs_layout.total_dim}"
            )
        if num_critic_obs != num_actor_obs:
            raise ValueError("ActorCriticTGBimanual expects critic obs to match actor obs")

        enc_cfg = TCEPointCloudEncoderCfg(
            num_pts=self.num_points,
            patch_size=int(patch_size),
            encoder_channel=int(encoder_channel),
            vit_depth=int(vit_depth),
            vit_heads=int(vit_heads),
            freeze=self.freeze_encoder,
        )
        self.encoder = BimanualTCEPointCloudEncoder(enc_cfg)
        self._load_bimanual_tce_encoder_checkpoint(
            encoder_weights_path,
            expected_dims={
                "num_pts": self.num_points,
                "patch_size": int(patch_size),
                "encoder_channel": int(encoder_channel),
            },
        )

        token_dim = self.encoder.feature_dim
        patches_per_cloud = self.encoder.num_patches
        self.token_dim = token_dim
        self.num_patches_per_cloud = patches_per_cloud
        self.total_num_tokens = 3 * patches_per_cloud
        activation_fn = resolve_nn_activation(activation)

        self.use_learnable_query_tokens = bool(use_learnable_query_tokens)
        ctx_dim = bimanual_context_dim(
            hand_state_dim=int(hand_state_dim),
            robot_state_dim=int(robot_state_dim),
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=int(relative_goal_dim),
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        self.context_dim = ctx_dim
        if sd_num_query_object is not None:
            print("  - sd_num_query_object is ignored; ActorCriticTGBimanual attends all 3-stream tokens")
        if num_query_object_tokens is not None:
            print("  - num_query_object_tokens is ignored; ActorCriticTGBimanual attends all 3-stream tokens")

        if not self.use_learnable_query_tokens:
            if sd_query_keys is None:
                sd_query_keys = ("context",)
            if int(sd_emb_dim) != int(token_dim):
                raise ValueError(
                    "ActorCriticTGBimanual relative/context fusion requires sd_emb_dim "
                    f"to match encoder token dim {token_dim}, got {sd_emb_dim}"
                )
            self.state_cross_all = BimanualRelativeContextCrossAttention(
                token_dim=token_dim,
                ctx_dim=ctx_dim,
                total_query_tokens=int(sd_num_query),
                relative_translation_query_tokens=int(relative_translation_query_tokens),
                n_heads=int(cross_attn_heads),
                n_layers=int(cross_attn_layers),
                cat_query=bool(sd_cat_query),
            )
            if reuse_pretrain_pose_cross_attn:
                self.state_cross_all.strict_load_pretrain_pose_cross_attn(encoder_weights_path)
            fusion_input_dim = self.state_cross_all.output_dim
            self.sd_cat_ctx = bool(sd_cat_ctx)
            if self.sd_cat_ctx:
                fusion_input_dim += ctx_dim
            self.sd_num_query = int(sd_num_query)
        else:
            self.num_query_tokens = int(sd_num_query if num_query_tokens is None else num_query_tokens)
            self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, token_dim) * 0.02)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=int(cross_attn_heads),
                dim_feedforward=cross_attn_ff_dim or (token_dim * 2),
                dropout=float(cross_attn_dropout),
                batch_first=True,
                activation="gelu",
            )
            self.cross_decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(cross_attn_layers))
            self.sd_cat_ctx = bool(sd_cat_ctx)
            fusion_input_dim = self.num_query_tokens * token_dim + (ctx_dim if self.sd_cat_ctx else 0)

        self.fusion_mlp = build_fusion_mlp(fusion_input_dim, fusion_hidden_dims, activation_fn)
        fusion_out_dim = fusion_hidden_dims[-1] if len(fusion_hidden_dims) > 0 else fusion_input_dim
        self.critic_fusion_mlp = None
        self.critic_state_cross_all = None
        self.critic_query_tokens = None
        self.critic_cross_decoder = None
        if self.separate_actor_critic_fusion:
            self.critic_fusion_mlp = copy.deepcopy(self.fusion_mlp)
            if not self.use_learnable_query_tokens:
                self.critic_state_cross_all = copy.deepcopy(self.state_cross_all)
            else:
                self.critic_query_tokens = nn.Parameter(self.query_tokens.detach().clone())
                self.critic_cross_decoder = copy.deepcopy(self.cross_decoder)

        self.actor = build_mlp(fusion_out_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = build_mlp(fusion_out_dim, critic_hidden_dims, activation_fn, 1)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(float(init_noise_std) * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(float(init_noise_std) * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(
            "[ActorCriticTGBimanual] initialized "
            f"tokens={self.total_num_tokens} patches_per_cloud={patches_per_cloud} "
            f"context_dim={self.context_dim} fusion_input_dim={fusion_input_dim}"
        )

    def _load_bimanual_tce_encoder_checkpoint(self, checkpoint_path: str, *, expected_dims: dict[str, int]) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = ActorCriticTG._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = ActorCriticTG._extract_tce_encoder_state_dict(raw_state, checkpoint_path)
        ActorCriticTG._validate_tce_checkpoint_metadata(
            metadata,
            expected_dims,
            checkpoint_path,
            encoder_state=encoder_state,
        )
        encoder_state = self._expand_legacy_type_embedding(encoder_state)
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"TCE encoder checkpoint is incompatible with ActorCriticTGBimanual: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Bimanual TCE encoder checkpoint key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    @staticmethod
    def _expand_legacy_type_embedding(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = dict(state)
        type_embed = state.get("type_embed")
        if type_embed is None:
            return state
        if type_embed.ndim != 2 or type_embed.shape[0] != 2:
            if type_embed.ndim == 2 and type_embed.shape[0] == 3:
                return state
            raise RuntimeError(f"Cannot expand TCE type_embed with shape {tuple(type_embed.shape)}")
        state_shape = (3, type_embed.shape[1])
        expanded = torch.empty(state_shape, dtype=type_embed.dtype, device=type_embed.device)
        expanded[0] = type_embed[0]
        expanded[1] = type_embed[0]
        expanded[2] = type_embed[1]
        state["type_embed"] = expanded
        return state

    def _split_observations(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_bimanual_observations(obs, self.obs_layout)

    def _tokenize(self, observations: torch.Tensor):
        parts = self._split_observations(observations)
        object_cloud = parts["object_cloud"]
        tool1_cloud = parts["tool1_cloud"]
        tool2_cloud = parts["tool2_cloud"]
        obj_bbox_center = parts["object_bbox_center"]
        tool1_bbox_center = parts["tool1_bbox_center"]
        tool2_bbox_center = parts["tool2_bbox_center"]

        if self.model_input_centering == "object_center":
            object_cloud_rel = object_cloud - obj_bbox_center.unsqueeze(1)
            tool1_cloud_rel = tool1_cloud - obj_bbox_center.unsqueeze(1)
            tool2_cloud_rel = tool2_cloud - obj_bbox_center.unsqueeze(1)
        else:
            object_cloud_rel, tool1_cloud_rel, tool2_cloud_rel = center_bimanual_clouds_by_bbox(
                object_cloud,
                tool1_cloud,
                tool2_cloud,
                obj_bbox_center,
                tool1_bbox_center,
                tool2_bbox_center,
            )

        if self.freeze_encoder:
            with torch.no_grad():
                res = self.encoder.encode(tool1_cloud_rel, tool2_cloud_rel, object_cloud_rel)
        else:
            res = self.encoder.encode(tool1_cloud_rel, tool2_cloud_rel, object_cloud_rel)

        return res.fused_tokens, build_bimanual_context_vector(parts)

    def _features_from_tokens_context(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor, *, branch: str = "actor"):
        critic_branch = branch == "critic" and self.separate_actor_critic_fusion
        fusion_mlp = self.critic_fusion_mlp if critic_branch else self.fusion_mlp
        if fusion_mlp is None:
            raise RuntimeError("critic fusion requested before critic fusion module was initialized")

        if not self.use_learnable_query_tokens:
            state_cross_all = self.critic_state_cross_all if critic_branch else self.state_cross_all
            if state_cross_all is None:
                raise RuntimeError("critic cross-attention requested before initialization")
            sd_out = state_cross_all(all_tokens, tool_object_rel=ctx_vec[:, :6], ctx_vec=ctx_vec)
            if self.sd_cat_ctx:
                sd_out = torch.cat([sd_out, ctx_vec], dim=-1)
            return fusion_mlp(sd_out)

        batch = all_tokens.shape[0]
        query_tokens = self.critic_query_tokens if critic_branch else self.query_tokens
        cross_decoder = self.critic_cross_decoder if critic_branch else self.cross_decoder
        if query_tokens is None or cross_decoder is None:
            raise RuntimeError("critic learnable-query fusion requested before initialization")
        query = query_tokens.expand(batch, -1, -1)
        attn_out = cross_decoder(tgt=query, memory=all_tokens, memory_key_padding_mask=None)
        attn_out_flat = attn_out.reshape(batch, -1)
        fusion_input = torch.cat([attn_out_flat, ctx_vec], dim=-1) if self.sd_cat_ctx else attn_out_flat
        return fusion_mlp(fusion_input)

    def _get_features(self, observations: torch.Tensor, *, branch: str = "actor"):
        all_tokens, ctx_vec = self._tokenize(observations)
        return self._features_from_tokens_context(all_tokens, ctx_vec, branch=branch)

    def update_distribution(self, observations: torch.Tensor):
        features = self._get_features(observations)
        mean = self.actor(features)
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

    def reset(self, dones=None):
        pass

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(self._get_features(critic_observations, branch="critic"))

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

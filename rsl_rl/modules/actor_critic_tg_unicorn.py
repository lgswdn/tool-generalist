# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic network using the UniCORN pretrained point-cloud encoder."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn

from pretrain.unicorn_model import UnicornGeometryEncoder, UnicornGeometryEncoderCfg
from rsl_rl.modules.actor_critic_tg import RelativeContextCrossAttention
from rsl_rl.modules.tg_policy_common import (
    ObservationLayout as TGObservationLayout,
    TGActorCriticHeadMixin,
    build_context_vector,
    build_fusion_mlp,
    build_mlp,
    context_dim,
    initialize_action_noise,
    split_observations,
    validate_observation_layout,
)
from rsl_rl.utils import resolve_nn_activation


_PRETRAIN_CHECKPOINT_SCHEMA = "pretrain_checkpoint_v1"


class ActorCriticTGUnicorn(TGActorCriticHeadMixin, nn.Module):
    """Single-arm TG policy backed by the UniCORN geometry encoder.

    Observation layout matches ``ActorCriticTG``:
        object_cloud | tool_cloud | object_bbox_center | tool_bbox_center |
        hand_state | robot_state | previous_action | relative_goal_pose | physics

    UniCORN is applied independently to object and tool clouds with shared
    weights.  Unlike ``ActorCriticTG``, the point clouds are passed to the
    encoder in the observation frame without bbox-center subtraction, matching
    the UniCORN contact pretraining setup.
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
        num_patches: int = 16,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
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
        hand_state_dim: int = 9,
        robot_state_dim: int = 14,
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
                "ActorCriticTGUnicorn.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = int(point_dim)
        self.num_points = int(num_points)
        self.num_actions = int(num_actions)
        self.noise_std_type = str(noise_std_type)
        self.freeze_encoder = bool(freeze_encoder)
        self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)
        self.model_input_centering = str(model_input_centering)
        self.previous_action_dim = int(previous_action_dim) if previous_action_dim is not None else int(num_actions)
        self.object_velocity_dim = int(object_velocity_dim)
        self.physics_dim = int(physics_dim)

        self.obs_layout = TGObservationLayout.build(
            num_points=self.num_points,
            point_dim=self.point_dim,
            hand_state_dim=int(hand_state_dim),
            robot_state_dim=int(robot_state_dim),
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=int(relative_goal_dim),
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        validate_observation_layout(
            policy_name="ActorCriticTGUnicorn",
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            layout=self.obs_layout,
        )
        self.pc_dim = 2 * self.num_points * self.point_dim

        if not encoder_weights_path:
            raise ValueError("ActorCriticTGUnicorn requires encoder_weights_path")

        enc_cfg = UnicornGeometryEncoderCfg(
            num_points=self.num_points,
            num_patches=int(num_patches),
            patch_size=int(patch_size),
            encoder_channel=int(encoder_channel),
            vit_depth=int(vit_depth),
            vit_heads=int(vit_heads),
        )
        self.encoder = UnicornGeometryEncoder(enc_cfg)
        self._load_unicorn_encoder_checkpoint(
            encoder_weights_path,
            expected_dims={
                "num_pts": self.num_points,
                "patch_size": int(patch_size),
                "encoder_channel": int(encoder_channel),
                "num_patches": int(num_patches),
            },
        )
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()

        D = self.encoder.feature_dim
        P = self.encoder.num_patches
        self.token_dim = D
        self.num_patches_per_cloud = P
        self.tokens_per_cloud = P + 1
        self.total_num_tokens = 2 * self.tokens_per_cloud
        activation_fn = resolve_nn_activation(activation)

        self.use_learnable_query_tokens = bool(use_learnable_query_tokens)
        sd_ctx_dim = context_dim(
            hand_state_dim=int(hand_state_dim),
            robot_state_dim=int(robot_state_dim),
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=int(relative_goal_dim),
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        self.context_dim = sd_ctx_dim

        if not self.use_learnable_query_tokens:
            if sd_query_keys is None:
                sd_query_keys = ("context",)
            if sd_num_query_object is not None:
                print("  - sd_num_query_object is ignored; ActorCriticTGUnicorn attends all UniCORN tokens")
            if num_query_object_tokens is not None:
                print("  - num_query_object_tokens is ignored; ActorCriticTGUnicorn attends all UniCORN tokens")
            if reuse_pretrain_pose_cross_attn:
                raise ValueError(
                    "ActorCriticTGUnicorn does not support reuse_pretrain_pose_cross_attn; "
                    "UniCORN contact checkpoints do not contain pose cross-attention weights."
                )
            if int(sd_emb_dim) != int(D):
                raise ValueError(
                    "ActorCriticTGUnicorn relative/context fusion requires sd_emb_dim "
                    f"to match encoder token dim {D}, got {sd_emb_dim}"
                )
            self.state_cross_all = RelativeContextCrossAttention(
                token_dim=D,
                ctx_dim=sd_ctx_dim,
                total_query_tokens=int(sd_num_query),
                relative_translation_query_tokens=int(relative_translation_query_tokens),
                n_heads=int(cross_attn_heads),
                n_layers=int(cross_attn_layers),
                cat_query=bool(sd_cat_query),
            )
            fusion_input_dim = self.state_cross_all.output_dim
            self.sd_num_query = int(sd_num_query)
            self.sd_cat_ctx = bool(sd_cat_ctx)
            if self.sd_cat_ctx:
                fusion_input_dim += sd_ctx_dim
        else:
            self.num_query_tokens = int(sd_num_query if num_query_tokens is None else num_query_tokens)
            self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, self.token_dim) * 0.02)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.token_dim,
                nhead=int(cross_attn_heads),
                dim_feedforward=cross_attn_ff_dim or (self.token_dim * 2),
                dropout=float(cross_attn_dropout),
                batch_first=True,
                activation="gelu",
            )
            self.cross_decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(cross_attn_layers))
            self.sd_cat_ctx = bool(sd_cat_ctx)
            fusion_input_dim = self.num_query_tokens * self.token_dim + (sd_ctx_dim if self.sd_cat_ctx else 0)

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
        initialize_action_noise(
            self,
            num_actions=num_actions,
            init_noise_std=float(init_noise_std),
            noise_std_type=self.noise_std_type,
        )

        print(
            "[ActorCriticTGUnicorn] initialized "
            f"tokens={self.total_num_tokens} patches_per_cloud={P} token_dim={D} "
            f"context_dim={self.context_dim} fusion_input_dim={fusion_input_dim} "
            f"separate_actor_critic_fusion={self.separate_actor_critic_fusion}"
        )

    def _load_unicorn_encoder_checkpoint(self, checkpoint_path: str, *, expected_dims: dict[str, int]) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = self._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = self._extract_unicorn_encoder_state_dict(raw_state, checkpoint_path)
        self._validate_unicorn_checkpoint_metadata(metadata, expected_dims, checkpoint_path)
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"UniCORN encoder checkpoint is incompatible with ActorCriticTGUnicorn: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "UniCORN encoder checkpoint key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    @staticmethod
    def _checkpoint_state_dict(ckpt: Any, checkpoint_path: str) -> dict[str, torch.Tensor]:
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "encoder"):
                state = ckpt.get(key)
                if isinstance(state, dict):
                    return state
            if all(isinstance(key, str) for key in ckpt.keys()):
                return ckpt
        raise RuntimeError(f"UniCORN checkpoint has no state_dict payload: {checkpoint_path}")

    @staticmethod
    def _extract_unicorn_encoder_state_dict(
        state_dict: dict[str, torch.Tensor],
        checkpoint_path: str,
    ) -> dict[str, torch.Tensor]:
        for prefix in ("module.encoder.", "encoder."):
            selected = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if selected:
                return selected
        if any(key.startswith(("patch_tokenizer.", "pos_embed.", "encoder.", "norm.", "emb_token")) for key in state_dict):
            return dict(state_dict)
        raise RuntimeError(
            "UniCORN checkpoint does not contain encoder keys under "
            f"'encoder.'/'module.encoder.' or direct UniCORN encoder keys: {checkpoint_path}"
        )

    @staticmethod
    def _validate_unicorn_checkpoint_metadata(
        metadata: Any,
        expected_dims: dict[str, int],
        checkpoint_path: str,
    ) -> None:
        if not isinstance(metadata, dict):
            return
        schema = metadata.get("schema_version")
        if schema != _PRETRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError(
                f"UniCORN checkpoint schema mismatch: expected {_PRETRAIN_CHECKPOINT_SCHEMA}, "
                f"got {schema!r} in {checkpoint_path}"
            )
        family = (metadata.get("model") or {}).get("family")
        if family is not None and str(family) != "unicorn":
            raise RuntimeError(f"UniCORN checkpoint family mismatch: got {family!r} in {checkpoint_path}")
        dims = metadata.get("model_dims") or (metadata.get("model") or {}).get("dims") or {}
        dim_aliases = {"num_pts": ("num_pts", "num_points")}
        for key, expected in expected_dims.items():
            names = dim_aliases.get(key, (key,))
            actual = next((dims[name] for name in names if name in dims and dims[name] is not None), None)
            if actual is not None and int(actual) != int(expected):
                raise RuntimeError(
                    f"UniCORN checkpoint dim mismatch for {key}: expected {expected}, "
                    f"got {actual} in {checkpoint_path}"
                )

    def _split_observations(self, obs: torch.Tensor):
        return split_observations(obs, self.obs_layout)

    def _encode_cloud(self, cloud: torch.Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(cloud)
        return self.encoder(cloud)

    def _tokenize(self, observations: torch.Tensor):
        parts = self._split_observations(observations)
        object_res = self._encode_cloud(parts["object_cloud"])
        tool_res = self._encode_cloud(parts["tool_cloud"])
        tool_tokens = torch.cat((tool_res.patch_tokens, tool_res.global_token.unsqueeze(1)), dim=1)
        object_tokens = torch.cat((object_res.patch_tokens, object_res.global_token.unsqueeze(1)), dim=1)
        all_tokens = torch.cat((tool_tokens, object_tokens), dim=1)
        ctx_vec = build_context_vector(parts)
        return all_tokens, ctx_vec

    def _features_from_tokens_context(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
        *,
        branch: str = "actor",
    ):
        critic_branch = branch == "critic" and self.separate_actor_critic_fusion
        fusion_mlp = self.critic_fusion_mlp if critic_branch else self.fusion_mlp
        if fusion_mlp is None:
            raise RuntimeError("critic fusion requested before critic fusion module was initialized")

        if not self.use_learnable_query_tokens:
            rel_t = ctx_vec[:, :3]
            state_cross_all = self.critic_state_cross_all if critic_branch else self.state_cross_all
            if state_cross_all is None:
                raise RuntimeError("critic cross-attention requested before initialization")
            sd_out = state_cross_all(all_tokens, rel_t=rel_t, ctx_vec=ctx_vec)
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

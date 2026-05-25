# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic network using the canonical pretrained TCE point-cloud encoder.

This actor-critic uses pretrain.model.TCEPointCloudEncoder and strict checkpoint
loading.  It accepts full pretrain checkpoints whose encoder keys are stored
under ``encoder.``/``module.encoder.`` or direct encoder state_dicts with
pretrain_checkpoint_v1 metadata.

These features are fused with robot state via SD-Cross attention (or learnable query
tokens with TransformerDecoder) for policy learning.

Observation layout:
    object_cloud | tool_cloud | object_bbox_center | tool_bbox_center |
    hand_state | robot_state | previous_action | relative_goal_pose | physics
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.tg_policy_common import (
    ObservationLayout as TGObservationLayout,
    build_context_vector,
    build_fusion_mlp,
    build_mlp,
    center_clouds_by_bbox,
    context_dim,
    split_observations,
)


_PRETRAIN_CHECKPOINT_SCHEMA = "pretrain_checkpoint_v1"
_TCE_ROOT_PREFIXES = (
    "patch_pointnet.",
    "patch_center_pos.",
    "type_embedding.",
    "joint_transformer.",
    "patch_enc.",
    "pos_embed.",
    "type_embed",
    "cls_token",
    "vit.",
    "norm.",
)


def _make_pretrain_style_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RelativeContextCrossAttention(nn.Module):
    """RL fusion with explicit relative-translation query tokens.

    ``sd_num_query`` is the total query count. The first
    ``relative_translation_query_tokens`` are generated from
    ``tool_bbox_center - object_bbox_center`` using the pretrain query-A MLP
    structure. The remaining queries are generated from the full RL context.
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
        self.context_query_tokens = self.total_query_tokens - self.relative_translation_query_tokens
        self.cat_query = bool(cat_query)
        if self.total_query_tokens <= 0:
            raise ValueError("sd_num_query must be > 0")
        if self.relative_translation_query_tokens < 0:
            raise ValueError("relative_translation_query_tokens must be >= 0")
        if self.relative_translation_query_tokens > self.total_query_tokens:
            raise ValueError("relative_translation_query_tokens must be <= sd_num_query")

        self.query_A = _make_pretrain_style_mlp(
            (3,) + tuple(condition_mlp_hidden_dims) + (self.relative_translation_query_tokens * self.token_dim,)
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

    def forward(self, tokens: torch.Tensor, rel_t: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        queries = []
        if self.relative_translation_query_tokens > 0:
            queries.append(
                self.query_A(rel_t).view(B, self.relative_translation_query_tokens, self.token_dim)
            )
        if self.context_query is not None:
            queries.append(
                self.context_query(ctx_vec).view(B, self.context_query_tokens, self.token_dim)
            )
        query = torch.cat(queries, dim=1)
        initial_query = query
        for layer in self.layers:
            residual = query
            query_norm = layer["norm1"](query)
            attn_out, _ = layer["query_cross_attn"](
                query=query_norm,
                key=tokens,
                value=tokens,
            )
            query = residual + attn_out
            query = query + layer["ff"](layer["norm2"](query))
        out = query.reshape(B, -1)
        if self.cat_query:
            out = torch.cat((out, initial_query.reshape(B, -1)), dim=-1)
        return out

    def strict_load_pretrain_pose_cross_attn(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ActorCriticTG._checkpoint_state_dict(ckpt, checkpoint_path)
        if any(key.startswith("module.") for key in state):
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
        missing: list[str] = []
        mismatched: list[str] = []

        def copy_param(target_name: str, source_name: str, target_tensor: torch.Tensor) -> None:
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

        for name, tensor in self.query_A.state_dict().items():
            copy_param(
                f"query_A.{name}",
                f"pose_cross_attn.query_generator.query_A.{name}",
                tensor,
            )
        for layer_i, layer in enumerate(self.layers):
            for module_name in ("query_cross_attn", "norm1", "norm2", "ff"):
                module = layer[module_name]
                for name, tensor in module.state_dict().items():
                    copy_param(
                        f"layers.{layer_i}.{module_name}.{name}",
                        f"pose_cross_attn.layers.{layer_i}.{module_name}.{name}",
                        tensor,
                    )
        if missing or mismatched:
            raise RuntimeError(
                "reuse_pretrain_pose_cross_attn=True but pretrain pose_cross_attn "
                "weights are incompatible. "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}; "
                f"mismatched={mismatched[:8]}{'...' if len(mismatched) > 8 else ''}"
            )


class ActorCriticTG(nn.Module):
    """Actor-critic network using canonical TCEPointCloudEncoder.

    Observation layout:
        object_cloud | tool_cloud | object_bbox_center | tool_bbox_center |
        hand_state | robot_state | previous_action | relative_goal_pose | physics

    The encoder processes tool_pc and obj_pc jointly, producing cross-stream-aware
    tokens. Point clouds stay in env-frame observations and are made relative
    to supplied bbox centers inside the model before encoding.  The bbox centers
    are also kept as pose context.
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
        # Point cloud settings
        num_points: int = 512,
        point_dim: int = 3,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 12,   # must match new_pretrain/config.py
        vit_heads: int = 4,
        # Encoder weights
        encoder_weights_path: Optional[str] = None,
        freeze_encoder: bool = True,
        separate_actor_critic_fusion: bool = False,
        # Cross-attention fusion settings
        use_learnable_query_tokens: bool = False,
        sd_num_query: int = 16,
        sd_num_query_object: Optional[int] = None,
        sd_emb_dim: int = 128,
        relative_translation_query_tokens: int = 2,
        reuse_pretrain_pose_cross_attn: bool = False,
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
        hand_state_dim: int = 9,
        robot_state_dim: int = 14,
        previous_action_dim: Optional[int] = None,
        relative_goal_dim: int = 9,
        object_velocity_dim: int = 0,
        physics_dim: int = 7,
        model_input_centering: str = "bbox_center",
        # Activation / noise
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"ActorCriticTG.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_encoder = freeze_encoder
        self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)
        self.model_input_centering = str(model_input_centering)
        if self.model_input_centering not in {"bbox_center", "object_center"}:
            raise ValueError(
                "ActorCriticTG model_input_centering must be 'bbox_center' or "
                f"'object_center', got {self.model_input_centering!r}"
            )
        self.previous_action_dim = int(previous_action_dim) if previous_action_dim is not None else int(num_actions)
        self.object_velocity_dim = int(object_velocity_dim)
        self.physics_dim = int(physics_dim)

        self.obs_layout = TGObservationLayout.build(
            num_points=num_points,
            point_dim=point_dim,
            hand_state_dim=hand_state_dim,
            robot_state_dim=robot_state_dim,
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=relative_goal_dim,
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        if num_actor_obs != self.obs_layout.total_dim:
            raise ValueError(
                "ActorCriticTG observation layout mismatch: "
                f"num_actor_obs={num_actor_obs}, expected={self.obs_layout.total_dim}"
            )
        if num_critic_obs != num_actor_obs:
            raise ValueError(
                "ActorCriticTG expects critic observations to use the same named layout "
                f"as actor observations, got num_critic_obs={num_critic_obs}, "
                f"num_actor_obs={num_actor_obs}"
            )
        self.pc_dim = 2 * num_points * point_dim

        activation_fn = resolve_nn_activation(activation)

        # ------------------------------------------------------------------
        # TCE encoder setup
        # ------------------------------------------------------------------
        print(f"[ActorCriticTG] Initializing TCEPointCloudEncoder...")
        print(f"  - Point cloud layout:")
        print(f"    * Object points: {num_points} (dim: {num_points * point_dim})")
        print(f"    * Tool points: {num_points} (dim: {num_points * point_dim})")
        print(f"    * Total point cloud dim: {self.pc_dim}")

        if not encoder_weights_path:
            raise ValueError(
                "ActorCriticTG requires encoder_weights_path from a canonical "
                "pretrain_checkpoint_v1 TCE checkpoint; refusing random encoder init."
            )

        enc_cfg = TCEPointCloudEncoderCfg(
            num_pts=num_points,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder,
        )
        self.encoder = TCEPointCloudEncoder(enc_cfg)
        self._load_tce_encoder_checkpoint(
            encoder_weights_path,
            expected_dims={
                "num_pts": int(num_points),
                "patch_size": int(patch_size),
                "encoder_channel": int(encoder_channel),
            },
        )

        D = self.encoder.feature_dim   # Token dimension
        P = self.encoder.num_patches   # Patches per cloud
        self.token_dim = D

        # Token layout from TCE encoder:
        #   tool_tokens (P) + obj_tokens (P) = 2P total
        self.num_cls_tokens = 0
        self.total_num_tokens = 2 * P
        self.num_patches_per_cloud = P

        print(f"[ActorCriticTG] Encoder config: D={D}, P={P}, vit_depth={vit_depth}, vit_heads={vit_heads}")

        # ------------------------------------------------------------------
        # Feature fusion
        # ------------------------------------------------------------------
        self.use_learnable_query_tokens = use_learnable_query_tokens

        # Context is strictly:
        # [tool_bbox_center-object_bbox_center, object_bbox_center, hand_state,
        #  robot_state, previous_action, relative_goal_pose, physics]
        sd_ctx_dim = context_dim(
            hand_state_dim=hand_state_dim,
            robot_state_dim=robot_state_dim,
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=relative_goal_dim,
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
        )
        self.context_dim = sd_ctx_dim

        if not self.use_learnable_query_tokens:
            # Option 1: explicit relative-translation queries + context queries.
            # sd_num_query is the total query count; the first
            # relative_translation_query_tokens are generated from
            # tool_bbox_center - object_bbox_center using the pretrain query-A
            # structure, and the rest are generated from the full context.
            print("[ActorCriticTG] Using relative/context cross-attention over all TCE tokens")

            if sd_query_keys is None:
                sd_query_keys = ("context",)
            if sd_num_query_object is not None:
                print("  - sd_num_query_object is ignored; ActorCriticTG attends all TCE tokens")
            if int(sd_emb_dim) != int(D):
                raise ValueError(
                    "ActorCriticTG relative/context fusion requires sd_emb_dim to "
                    f"match encoder token dim {D}, got {sd_emb_dim}"
                )

            print(f"  - Query tokens: {sd_num_query} over all tokens")
            print(f"  - Relative-translation query tokens: {relative_translation_query_tokens}")
            print(f"  - Context query tokens: {int(sd_num_query) - int(relative_translation_query_tokens)}")
            print(f"  - Token dimension: {self.token_dim}")
            print(f"  - All tokens: {self.total_num_tokens} (tool patches: {P}, object patches: {P})")
            print(f"  - Context dimension: {sd_ctx_dim}")
            print(f"  - Embedding dimension: {sd_emb_dim}")

            self.state_cross_all = RelativeContextCrossAttention(
                token_dim=D,
                ctx_dim=sd_ctx_dim,
                total_query_tokens=int(sd_num_query),
                relative_translation_query_tokens=int(relative_translation_query_tokens),
                n_heads=int(cross_attn_heads),
                n_layers=int(cross_attn_layers),
                cat_query=bool(sd_cat_query),
            )
            sd_out_dim = self.state_cross_all.output_dim
            if reuse_pretrain_pose_cross_attn:
                self.state_cross_all.strict_load_pretrain_pose_cross_attn(encoder_weights_path)

            # Store configuration
            self.sd_num_query = sd_num_query
            self.sd_cat_ctx = sd_cat_ctx

            # Calculate output dimension
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim

            fusion_input_dim = sd_out_dim
        else:
            # Option 2: Learnable query tokens with TransformerDecoder
            print("[ActorCriticTG] Using learnable query tokens with TransformerDecoder")
            self.num_query_tokens = int(sd_num_query if num_query_tokens is None else num_query_tokens)
            if num_query_object_tokens is not None:
                print("  - num_query_object_tokens is ignored; learnable queries attend all TCE tokens")

            print(f"  - Query tokens: {self.num_query_tokens} over all tokens")
            print(f"  - Token dimension: {self.token_dim}")
            print(f"  - Cross attention heads: {cross_attn_heads}")
            print(f"  - Cross attention layers: {cross_attn_layers}")

            # Learnable query tokens
            self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, self.token_dim) * 0.02)

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

            # Calculate output dimension
            cross_out_dim = self.num_query_tokens * self.token_dim
            fusion_input_dim = cross_out_dim + (sd_ctx_dim if sd_cat_ctx else 0)
            self.sd_cat_ctx = sd_cat_ctx

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

        print(f"[ActorCriticTG] Fusion input dim: {fusion_input_dim}")
        print(f"  - Context dim: {self.context_dim}")
        print(f"  - Separate actor/critic fusion: {self.separate_actor_critic_fusion}")

        # Actor / Critic heads
        self.actor = build_mlp(fusion_out_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = build_mlp(fusion_out_dim, critic_hidden_dims, activation_fn, 1)

        # Action distribution params
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticTG] Initialization complete")

    def _load_tce_encoder_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = self._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = self._extract_tce_encoder_state_dict(raw_state, checkpoint_path)
        self._validate_tce_checkpoint_metadata(
            metadata,
            expected_dims,
            checkpoint_path,
            encoder_state=encoder_state,
        )
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"TCE encoder checkpoint is incompatible with ActorCriticTG: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "TCE encoder checkpoint key mismatch: "
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
        raise RuntimeError(f"TCE checkpoint has no state_dict payload: {checkpoint_path}")

    @staticmethod
    def _validate_tce_checkpoint_metadata(
        metadata: Any,
        expected_dims: dict[str, int],
        checkpoint_path: str,
        *,
        encoder_state: dict[str, torch.Tensor],
    ) -> None:
        if not isinstance(metadata, dict):
            if any(key.startswith(("patch_enc.", "pos_embed.", "vit.", "norm.")) for key in encoder_state):
                return
            raise RuntimeError(
                f"TCE checkpoint missing metadata schema '{_PRETRAIN_CHECKPOINT_SCHEMA}': {checkpoint_path}"
            )
        schema = metadata.get("schema_version")
        if schema != _PRETRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError(
                f"TCE checkpoint schema mismatch: expected {_PRETRAIN_CHECKPOINT_SCHEMA}, "
                f"got {schema!r} in {checkpoint_path}"
            )
        dims = metadata.get("model_dims") or metadata.get("model", {}).get("dims") or {}
        dim_aliases = {"num_pts": ("num_pts", "num_points")}
        for key, expected in expected_dims.items():
            names = dim_aliases.get(key, (key,))
            actual = next((dims[name] for name in names if name in dims and dims[name] is not None), None)
            if actual is not None and int(actual) != int(expected):
                raise RuntimeError(
                    f"TCE checkpoint dim mismatch for {key}: expected {expected}, "
                    f"got {actual} in {checkpoint_path}"
                )

    @staticmethod
    def _extract_tce_encoder_state_dict(
        state_dict: dict[str, torch.Tensor],
        checkpoint_path: str,
    ) -> dict[str, torch.Tensor]:
        keys = tuple(state_dict.keys())
        for prefix in ("module.encoder.", "encoder."):
            selected = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if selected:
                return ActorCriticTG._map_legacy_former_encoder_keys(selected)
        if any(key.startswith(_TCE_ROOT_PREFIXES) for key in keys):
            return ActorCriticTG._map_legacy_former_encoder_keys(dict(state_dict))
        raise RuntimeError(
            "TCE checkpoint does not contain canonical encoder keys under "
            f"'encoder.'/'module.encoder.' or direct TCE keys: {checkpoint_path}"
        )

    @staticmethod
    def _map_legacy_former_encoder_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map legacy Former/SDF encoder keys to canonical TCE key names.

        The current TCE architecture intentionally matches the legacy Former
        PC/ViT backbone, but the module path names differ. This mapping keeps
        strict loading deterministic while accepting old Former checkpoints.
        """

        mapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            if key == "type_embed":
                new_key = "type_embed"
            elif key == "cls_token":
                new_key = "cls_token"
            elif key.startswith("patch_enc."):
                new_key = key
            elif key.startswith("pos_embed."):
                new_key = key
            elif key.startswith("vit."):
                new_key = key
            elif key.startswith("norm."):
                new_key = key
            mapped[new_key] = value
        return mapped

    # --------------------------------------------------------------------------
    # Observation parsing
    # --------------------------------------------------------------------------
    def _split_observations(self, obs: torch.Tensor):
        """Split observations into named fields matching env_tool.PolicyCfg.

        Layout:
            object_cloud | tool_cloud | object_bbox_center | tool_bbox_center |
            hand_state | robot_state | previous_action | relative_goal_pose | physics
        """
        return split_observations(obs, self.obs_layout)

    # --------------------------------------------------------------------------
    # Tokenization via TCE encoder
    # --------------------------------------------------------------------------
    def _tokenize(self, observations: torch.Tensor):
        """Extract tokens from TCE encoder and split observations.

        Point clouds arrive in env-frame coordinates.  The env observation
        also supplies object/tool bbox centers as separate fields.  The encoder
        gets relative clouds after subtracting those bbox centers; the context
        vector keeps the bbox centers for pose conditioning.

        Returns:
            all_tokens:    (B, 2P, D) — tool_tokens + obj_tokens
            ctx_vec:       (B, context_dim) — strict policy conditioning vector
        """
        parts = self._split_observations(observations)
        object_cloud = parts["object_cloud"]
        tool_cloud = parts["tool_cloud"]
        obj_bbox_center = parts["object_bbox_center"]
        tool_bbox_center = parts["tool_bbox_center"]

        if self.model_input_centering == "object_center":
            object_cloud_rel = object_cloud - obj_bbox_center.unsqueeze(1)
            tool_cloud_rel = tool_cloud - obj_bbox_center.unsqueeze(1)
        else:
            object_cloud_rel, tool_cloud_rel = center_clouds_by_bbox(
                object_cloud,
                tool_cloud,
                obj_bbox_center,
                tool_bbox_center,
            )

        if self.freeze_encoder:
            with torch.no_grad():
                res = self.encoder.encode(tool_cloud_rel, object_cloud_rel)
        else:
            res = self.encoder.encode(tool_cloud_rel, object_cloud_rel)

        all_tokens = res.fused_tokens  # (B, 2P, D)

        # Strict context:
        # [tool_bbox_center-object_bbox_center, object_bbox_center, hand_state,
        #  robot_state, previous_action, relative_goal_pose, physics]
        ctx_vec = build_context_vector(parts)

        return all_tokens, ctx_vec

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------
    def _features_from_tokens_context(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
        *,
        branch: str = "actor",
    ):
        """Get fused features from precomputed encoder tokens and context."""
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
        query = query_tokens.expand(batch, -1, -1)  # (B, num_query_tokens, D)

        attn_out = cross_decoder(
            tgt=query,
            memory=all_tokens,
            memory_key_padding_mask=None,
        )
        attn_out_flat = attn_out.reshape(batch, -1)  # (B, num_query_tokens * D)

        fusion_input = (
            torch.cat([attn_out_flat, ctx_vec], dim=-1)
            if self.sd_cat_ctx
            else attn_out_flat
        )

        return fusion_mlp(fusion_input)

    def _get_features(self, observations: torch.Tensor, *, branch: str = "actor"):
        """Get fused features using either SD-Cross or learnable query tokens."""
        all_tokens, ctx_vec = self._tokenize(observations)
        return self._features_from_tokens_context(all_tokens, ctx_vec, branch=branch)

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
        features = self._get_features(critic_observations, branch="critic")
        return self.critic(features)

    def get_cached_encoder_features(self, observations: torch.Tensor):
        return self._tokenize(observations)

    def act_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        mean = self.actor(features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)
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

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

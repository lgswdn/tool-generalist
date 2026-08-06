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

from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from pretrain.unicorn_model import UnicornGeometryEncoderCfg, UnicornPairEncoder
from rsl_rl.modules.patch_distance_pointnet_encoder import PatchDistancePointNetEncoder
from rsl_rl.modules.official_unicorn_encoder import OfficialUnicornPairEncoder
from rsl_rl.modules.oracle_patch_encoder import OraclePatchDistanceEncoder
from rsl_rl.modules.oracle_pointmesh_pointnet_encoder import OraclePointMeshPointNetEncoder
from rsl_rl.modules.oracle_pointcloud_pointnet_encoder import OraclePointCloudPointNetEncoder
from rsl_rl.modules.oracle_pointcloud_patch_oracle_encoder import (
    OraclePointCloudPatchOracleEncoder,
)
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.tg_policy_common import (
    ObservationLayout as TGObservationLayout,
    TGActorCriticHeadMixin,
    build_context_vector,
    build_fusion_mlp,
    build_mlp,
    center_clouds_by_bbox,
    context_dim,
    initialize_action_noise,
    split_observations,
    validate_observation_layout,
)


_PRETRAIN_CHECKPOINT_SCHEMA = "pretrain_checkpoint_v1"
_VIT_ATTENTION_CONTRACT = "explicit_v1"
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
        token_order: str = "joint",
        condition_mlp_hidden_dims: tuple[int, ...] = (128, 128),
        cat_query: bool = False,
        need_attention_weights: bool = True,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.total_query_tokens = int(total_query_tokens)
        self.relative_translation_query_tokens = int(relative_translation_query_tokens)
        self.context_query_tokens = self.total_query_tokens - self.relative_translation_query_tokens
        self.cat_query = bool(cat_query)
        self.need_attention_weights = bool(need_attention_weights)
        self.token_order = str(token_order).strip().lower()
        if self.total_query_tokens <= 0:
            raise ValueError("sd_num_query must be > 0")
        if self.relative_translation_query_tokens < 0:
            raise ValueError("relative_translation_query_tokens must be >= 0")
        if self.relative_translation_query_tokens > self.total_query_tokens:
            raise ValueError("relative_translation_query_tokens must be <= sd_num_query")
        if self.token_order not in {"joint", "tool_then_object"}:
            raise ValueError("cross-attention token_order must be joint or tool_then_object")
        if self.token_order == "tool_then_object" and int(n_layers) != 2:
            raise ValueError("tool_then_object cross-attention requires exactly two layers")

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
        if self.token_order == "tool_then_object":
            if tokens.shape[1] % 2 != 0:
                raise RuntimeError(
                    "tool_then_object fusion requires equal tool/object token counts"
                )
            tool_tokens, object_tokens = tokens.chunk(2, dim=1)
            layer_memories = (tool_tokens, object_tokens)
        else:
            layer_memories = tuple(tokens for _ in self.layers)
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
        for layer, memory in zip(self.layers, layer_memories):
            residual = query
            query_norm = layer["norm1"](query)
            attn_out, _ = layer["query_cross_attn"](
                query=query_norm,
                key=memory,
                value=memory,
                need_weights=self.need_attention_weights,
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


class ActorCriticTG(TGActorCriticHeadMixin, nn.Module):
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
        num_patches: int = 16,
        encoder_channel: int = 128,
        vit_depth: int = 12,   # must match new_pretrain/config.py
        vit_heads: int = 4,
        vit_attention_mode: str | None = None,
        kinematic_conditioning: bool = False,
        kinematic_attention_layers: int = 1,
        oracle_contact_eps: float = 0.002,
        oracle_center_scale_m: float = 0.30,
        oracle_distance_scale_m: float = 0.10,
        oracle_patch_relative_scale_m: float = 0.05,
        oracle_log_distance_resolution_m: float = 0.005,
        oracle_log_distance_cap_m: float = 0.05,
        oracle_normalization_clip: float = 5.0,
        oracle_pointmesh_coordinate_scale_m: float = 0.30,
        oracle_pointmesh_distance_scale_m: float = 0.10,
        oracle_pointmesh_normalization_clip: float = 5.0,
        oracle_pointcloud_nearest_frame_batch_size: int = 64,
        oracle_pointcloud_feature_mode: str = "fast11",
        oracle_pointcloud_load_fitted_weights: bool = True,
        oracle_pointcloud_use_rank10_bottleneck: bool = True,
        oracle_pointcloud_token_mode: str = "patches",
        oracle_pointcloud_input_normalization: str = "identity",
        oracle_pointcloud_checkpoint_adapter: str = "oracle_pointcloud_pointnet_strict",
        patch_distance_point_scale_m: float = 0.05,
        patch_distance_patch_center_scale_m: float = 0.30,
        # Encoder weights
        encoder_weights_path: Optional[str] = None,
        encoder_backend: str = "tce",
        unicorn_token_source: str = "encoder",
        encoder_token_pca_rank: int = 128,
        encoder_token_pca_path: Optional[str] = None,
        encoder_token_bottleneck_rank: int = 128,
        encoder_token_bottleneck_pca_path: Optional[str] = None,
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
        cross_attn_token_order: str = "joint",
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
        task_embedding_dim: int = 0,
        physics_dim: int = 7,
        model_input_centering: str = "bbox_center",
        # Activation / noise
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                "ActorCriticTG received unexpected arguments: "
                f"{sorted(kwargs)}"
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_encoder = freeze_encoder
        self.kinematic_conditioning = bool(kinematic_conditioning)
        self.kinematic_attention_layers = int(kinematic_attention_layers)
        if vit_attention_mode not in {"joint_self", "cross_only"}:
            raise ValueError(
                "ActorCriticTG requires explicit vit_attention_mode="
                f"joint_self or cross_only, got {vit_attention_mode!r}"
            )
        self.encoder_backend = str(encoder_backend).lower()
        self.unicorn_token_source = str(unicorn_token_source).lower()
        self.encoder_token_pca_rank = int(encoder_token_pca_rank)
        self.encoder_token_pca_path = encoder_token_pca_path
        self.encoder_token_bottleneck_rank = int(encoder_token_bottleneck_rank)
        self.encoder_token_bottleneck_pca_path = encoder_token_bottleneck_pca_path
        if self.encoder_backend not in {
            "tce",
            "unicorn",
            "oracle_patch",
            "oracle_pointmesh_pointnet",
            "oracle_pointcloud_pointnet",
            "oracle_pointcloud_patch_oracle",
            "patch_distance_pointnet",
        }:
            raise ValueError(
                "ActorCriticTG encoder_backend must be 'tce', 'unicorn', or "
                "'oracle_patch', 'oracle_pointmesh_pointnet', or "
                "'oracle_pointcloud_pointnet', 'oracle_pointcloud_patch_oracle', "
                "or 'patch_distance_pointnet', "
                f"got {encoder_backend!r}"
            )
        if self.unicorn_token_source not in {"encoder", "contact_head_hidden"}:
            raise ValueError(
                "unicorn_token_source must be 'encoder' or 'contact_head_hidden', "
                f"got {unicorn_token_source!r}"
            )
        if (
            self.encoder_backend not in {"unicorn", "tce"}
            and self.unicorn_token_source != "encoder"
        ):
            raise ValueError(
                "contact_head_hidden token source requires encoder_backend='unicorn' or 'tce'"
            )
        if not 1 <= self.encoder_token_pca_rank <= int(encoder_channel):
            raise ValueError("encoder_token_pca_rank must be in [1, encoder_channel]")
        if self.encoder_token_pca_rank < int(encoder_channel) and self.encoder_backend != "tce":
            raise ValueError("encoder-token PCA projection currently requires encoder_backend='tce'")
        if not 1 <= self.encoder_token_bottleneck_rank <= int(encoder_channel):
            raise ValueError("encoder_token_bottleneck_rank must be in [1, encoder_channel]")
        if (
            self.encoder_token_bottleneck_rank < int(encoder_channel)
            and self.encoder_backend != "tce"
        ):
            raise ValueError(
                "encoder-token bottleneck currently requires encoder_backend='tce'"
            )
        if (
            self.encoder_token_bottleneck_rank < int(encoder_channel)
            and self.encoder_token_pca_rank < int(encoder_channel)
        ):
            raise ValueError(
                "fixed encoder-token PCA and trainable bottleneck cannot both be enabled"
            )
        if (
            self.encoder_token_bottleneck_rank < int(encoder_channel)
            and self.unicorn_token_source != "encoder"
        ):
            raise ValueError(
                "encoder-token bottleneck requires unicorn_token_source='encoder'"
            )
        if self.kinematic_conditioning:
            if self.encoder_backend != "tce":
                raise ValueError(
                    "Kinematic conditioning requires encoder_backend='tce'"
                )
            if self.unicorn_token_source != "encoder":
                raise ValueError(
                    "Kinematic conditioning requires unicorn_token_source='encoder'"
                )
            if self.encoder_token_pca_rank != int(encoder_channel):
                raise ValueError(
                    "Kinematic conditioning does not support encoder-token PCA"
                )
        self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)
        self.model_input_centering = str(model_input_centering)
        if self.model_input_centering not in {"bbox_center", "object_center"}:
            raise ValueError(
                "ActorCriticTG model_input_centering must be 'bbox_center' or "
                f"'object_center', got {self.model_input_centering!r}"
            )
        self.previous_action_dim = int(previous_action_dim) if previous_action_dim is not None else int(num_actions)
        self.object_velocity_dim = int(object_velocity_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.physics_dim = int(physics_dim)
        self.patch_distance_patch_center_scale_m = float(
            patch_distance_patch_center_scale_m
        )
        if self.patch_distance_patch_center_scale_m <= 0.0:
            raise ValueError("patch_distance_patch_center_scale_m must be > 0")

        self.obs_layout = TGObservationLayout.build(
            num_points=num_points,
            point_dim=point_dim,
            hand_state_dim=hand_state_dim,
            robot_state_dim=robot_state_dim,
            previous_action_dim=self.previous_action_dim,
            relative_goal_dim=relative_goal_dim,
            object_velocity_dim=self.object_velocity_dim,
            physics_dim=self.physics_dim,
            task_embedding_dim=self.task_embedding_dim,
            oracle_mesh_sdf_dim=(2 * num_points if self.encoder_backend == "oracle_patch" else 0),
            oracle_mesh_unsigned_distance_dim=(
                2 * num_points if self.encoder_backend == "oracle_pointmesh_pointnet" else 0
            ),
            include_kinematic_gripper_clouds=self.kinematic_conditioning,
        )
        validate_observation_layout(
            policy_name="ActorCriticTG",
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            layout=self.obs_layout,
        )
        self.pc_dim = 2 * num_points * point_dim

        activation_fn = resolve_nn_activation(activation)

        # ------------------------------------------------------------------
        # Encoder setup. Everything after this block is backend-independent.
        # ------------------------------------------------------------------
        scratch_pointcloud_encoder = (
            self.encoder_backend == "oracle_pointcloud_pointnet"
            and not bool(oracle_pointcloud_load_fitted_weights)
        )
        if (
            self.encoder_backend != "oracle_patch"
            and not scratch_pointcloud_encoder
            and not encoder_weights_path
        ):
            raise ValueError(
                "ActorCriticTG requires encoder_weights_path from a canonical "
                "pretrain_checkpoint_v1 checkpoint; refusing random encoder init."
            )

        expected_dims = {
            "num_pts": int(num_points),
            "patch_size": int(patch_size),
            "encoder_channel": int(encoder_channel),
        }
        if self.encoder_backend == "oracle_patch":
            oracle_kwargs = dict(
                num_points=num_points,
                num_patches=num_patches,
                patch_size=patch_size,
                feature_dim=encoder_channel,
                contact_eps=oracle_contact_eps,
                center_scale_m=oracle_center_scale_m,
                distance_scale_m=oracle_distance_scale_m,
                normalization_clip=oracle_normalization_clip,
                patch_relative_scale_m=oracle_patch_relative_scale_m,
                log_distance_resolution_m=oracle_log_distance_resolution_m,
                log_distance_cap_m=oracle_log_distance_cap_m,
            )
            self.encoder = OraclePatchDistanceEncoder(**oracle_kwargs)
        elif self.encoder_backend == "oracle_pointmesh_pointnet":
            self.encoder = OraclePointMeshPointNetEncoder(
                num_points=num_points,
                num_patches=num_patches,
                patch_size=patch_size,
                feature_dim=encoder_channel,
                coordinate_scale_m=oracle_pointmesh_coordinate_scale_m,
                distance_scale_m=oracle_pointmesh_distance_scale_m,
                normalization_clip=oracle_pointmesh_normalization_clip,
            )
            expected_dims["num_patches"] = int(num_patches)
            self._load_oracle_pointmesh_pointnet_checkpoint(
                encoder_weights_path,
                expected_dims=expected_dims,
            )
        elif self.encoder_backend == "oracle_pointcloud_pointnet":
            self.encoder = OraclePointCloudPointNetEncoder(
                num_points=num_points,
                num_patches=num_patches,
                patch_size=patch_size,
                feature_dim=encoder_channel,
                nearest_frame_batch_size=oracle_pointcloud_nearest_frame_batch_size,
                feature_mode=oracle_pointcloud_feature_mode,
                use_rank10_bottleneck=oracle_pointcloud_use_rank10_bottleneck,
                token_mode=oracle_pointcloud_token_mode,
                input_normalization=oracle_pointcloud_input_normalization,
            )
            print(
                "[ActorCriticTG] oracle_pointcloud_pointnet "
                f"token_mode={self.encoder.token_mode} "
                f"tokens_per_body={self.encoder.num_patches} "
                f"features={self.encoder.feature_mode} "
                f"rank10_bottleneck={self.encoder.use_rank10_bottleneck} "
                f"input_normalization={self.encoder.input_normalization} "
                f"initialization={'fitted' if oracle_pointcloud_load_fitted_weights else 'scratch'}"
            )
            expected_dims["num_patches"] = int(num_patches)
            if oracle_pointcloud_load_fitted_weights:
                if not encoder_weights_path:
                    raise RuntimeError("fitted oracle point-cloud PointNet requires a checkpoint")
                self._load_oracle_pointcloud_pointnet_checkpoint(
                    encoder_weights_path,
                    expected_dims=expected_dims,
                    load_fitted_weights=True,
                    checkpoint_adapter=oracle_pointcloud_checkpoint_adapter,
                )
            elif encoder_weights_path and self.encoder.feature_mode == "fast11":
                # Backward-compatible scratch ablation: reuse only fixed 11D
                # normalization when a fitted checkpoint is explicitly supplied.
                self._load_oracle_pointcloud_pointnet_checkpoint(
                    encoder_weights_path,
                    expected_dims=expected_dims,
                    load_fitted_weights=False,
                    checkpoint_adapter=oracle_pointcloud_checkpoint_adapter,
                )
        elif self.encoder_backend == "oracle_pointcloud_patch_oracle":
            if (int(num_points), int(num_patches), int(patch_size), int(encoder_channel)) != (
                512,
                16,
                32,
                128,
            ):
                raise RuntimeError(
                    "analytic point-cloud patch oracle requires num_points=512, "
                    "num_patches=16, patch_size=32, encoder_channel=128"
                )
            self.encoder = OraclePointCloudPatchOracleEncoder(
                num_points=num_points,
                num_patches=num_patches,
                patch_size=patch_size,
                feature_dim=encoder_channel,
                nearest_frame_batch_size=oracle_pointcloud_nearest_frame_batch_size,
            )
            metadata = self.encoder.load_fitted_checkpoint(encoder_weights_path)
            metrics = metadata.get("metrics") or {}
            print(
                "[ActorCriticTG] oracle_pointcloud_patch_oracle "
                f"checkpoint_epoch={metrics.get('epoch')} "
                f"validation_r2={metrics.get('r2')}"
            )
        elif self.encoder_backend == "patch_distance_pointnet":
            self.encoder = PatchDistancePointNetEncoder(
                num_points=int(num_points),
                num_patches=int(num_patches),
                patch_size=int(patch_size),
                feature_dim=int(encoder_channel),
                point_scale_m=float(patch_distance_point_scale_m),
            )
            expected_dims["num_patches"] = int(num_patches)
            self._load_patch_distance_pointnet_checkpoint(
                encoder_weights_path,
                expected_dims=expected_dims,
            )
        elif self.encoder_backend == "unicorn":
            if self._is_released_unicorn_checkpoint(encoder_weights_path):
                self.encoder = OfficialUnicornPairEncoder(
                    num_points=int(num_points),
                    num_patches=int(num_patches),
                    patch_size=int(patch_size),
                    feature_dim=int(encoder_channel),
                    num_layers=int(vit_depth),
                )
                self._load_released_unicorn_encoder_checkpoint(encoder_weights_path)
                print(
                    "[ActorCriticTG] loaded authors' released UniCORN encoder "
                    "(multiresolution level 1: 16 patches x 32 points)"
                )
            else:
                self.encoder = UnicornPairEncoder(
                    UnicornGeometryEncoderCfg(
                        num_points=int(num_points),
                        num_patches=int(num_patches),
                        patch_size=int(patch_size),
                        encoder_channel=int(encoder_channel),
                        vit_depth=int(vit_depth),
                        vit_heads=int(vit_heads),
                    )
                )
                expected_dims["num_patches"] = int(num_patches)
                self._load_unicorn_encoder_checkpoint(
                    encoder_weights_path,
                    expected_dims=expected_dims,
                )
            if self.unicorn_token_source == "contact_head_hidden":
                (
                    self.unicorn_tool_contact_hidden,
                    self.unicorn_obj_contact_hidden,
                ) = self._load_unicorn_contact_hidden_heads(
                    encoder_weights_path,
                    expected_input_dim=int(encoder_channel),
                )
        else:
            self.encoder = TCEPointCloudEncoder(
                TCEPointCloudEncoderCfg(
                    num_pts=num_points,
                    patch_size=patch_size,
                    encoder_channel=encoder_channel,
                    vit_depth=vit_depth,
                    vit_heads=vit_heads,
                    freeze=freeze_encoder,
                    vit_attention_mode=vit_attention_mode,
                    kinematic_conditioning=self.kinematic_conditioning,
                    kinematic_attention_layers=self.kinematic_attention_layers,
                )
            )
            self._load_tce_encoder_checkpoint(
                encoder_weights_path,
                expected_dims=expected_dims,
                expected_vit_attention_mode=vit_attention_mode,
                expected_kinematic_conditioning=self.kinematic_conditioning,
            )
            if self.unicorn_token_source == "contact_head_hidden":
                (
                    self.unicorn_tool_contact_hidden,
                    self.unicorn_obj_contact_hidden,
                ) = self._load_unicorn_contact_hidden_heads(
                    encoder_weights_path,
                    expected_input_dim=int(encoder_channel),
                )
        if self.encoder_token_pca_rank < int(encoder_channel):
            self._load_encoder_token_pca(
                self.encoder_token_pca_path,
                rank=self.encoder_token_pca_rank,
                expected_dim=int(encoder_channel),
            )
        if self.encoder_token_bottleneck_rank < int(encoder_channel):
            self._initialize_encoder_token_bottleneck(
                self.encoder_token_bottleneck_pca_path,
                rank=self.encoder_token_bottleneck_rank,
                expected_dim=int(encoder_channel),
            )
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()
            for name in ("unicorn_tool_contact_hidden", "unicorn_obj_contact_hidden"):
                module = getattr(self, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                    module.eval()

        # The fast point-cloud oracle has fixed discrete searches followed by
        # trainable PointNet layers.  PPO can cache the searches independently
        # of whether the PointNet is frozen.
        self.supports_trainable_preprocessing_cache = bool(
            self.encoder_backend
            in {"oracle_pointcloud_pointnet", "oracle_pointcloud_patch_oracle"}
            and not self.freeze_encoder
        )

        D = self.encoder.feature_dim   # Token dimension
        P = self.encoder.num_patches   # Patches per cloud
        self.token_dim = D

        # Token layout from TCE encoder:
        #   tool_tokens (P) + obj_tokens (P) = 2P total
        self.num_cls_tokens = 0
        self.total_num_tokens = 2 * P + (
            3 if self.kinematic_conditioning else 0
        )
        self.num_patches_per_cloud = P
        if self.encoder_backend == "patch_distance_pointnet":
            self.patch_distance_position_embedding = nn.Sequential(
                nn.Linear(3, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, D),
                nn.LayerNorm(D),
            )
            self.patch_distance_body_embedding = nn.Parameter(torch.zeros(2, D))
            nn.init.normal_(self.patch_distance_body_embedding, std=0.02)

        # ------------------------------------------------------------------
        # Feature fusion
        # ------------------------------------------------------------------
        self.use_learnable_query_tokens = use_learnable_query_tokens
        self.cross_attn_token_order = str(cross_attn_token_order).strip().lower()
        if self.use_learnable_query_tokens and self.cross_attn_token_order != "joint":
            raise ValueError(
                "ordered tool/object fusion is supported only for state-generated queries"
            )

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
            task_embedding_dim=self.task_embedding_dim,
        )
        self.context_dim = sd_ctx_dim

        if not self.use_learnable_query_tokens:
            # Option 1: explicit relative-translation queries + context queries.
            # sd_num_query is the total query count; the first
            # relative_translation_query_tokens are generated from
            # tool_bbox_center - object_bbox_center using the pretrain query-A
            # structure, and the rest are generated from the full context.
            if sd_query_keys is None:
                sd_query_keys = ("context",)
            if int(sd_emb_dim) != int(D):
                raise ValueError(
                    "ActorCriticTG relative/context fusion requires sd_emb_dim to "
                    f"match encoder token dim {D}, got {sd_emb_dim}"
                )

            self.state_cross_all = RelativeContextCrossAttention(
                token_dim=D,
                ctx_dim=sd_ctx_dim,
                total_query_tokens=int(sd_num_query),
                relative_translation_query_tokens=int(relative_translation_query_tokens),
                n_heads=int(cross_attn_heads),
                n_layers=int(cross_attn_layers),
                token_order=self.cross_attn_token_order,
                cat_query=bool(sd_cat_query),
                need_attention_weights=not (
                    self.encoder_backend == "oracle_pointcloud_pointnet"
                    and str(oracle_pointcloud_token_mode).strip().lower() == "points"
                ),
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
            self.num_query_tokens = int(sd_num_query if num_query_tokens is None else num_query_tokens)

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
        self.fusion_out_dim = int(fusion_out_dim)
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

        # Actor / Critic heads
        self.actor = build_mlp(fusion_out_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = build_mlp(fusion_out_dim, critic_hidden_dims, activation_fn, 1)

        initialize_action_noise(
            self,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            noise_std_type=self.noise_std_type,
        )


    def _load_unicorn_encoder_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = self._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = self._extract_unicorn_encoder_state_dict(raw_state, checkpoint_path)
        self._validate_unicorn_checkpoint_metadata(metadata, expected_dims, checkpoint_path)
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Paired UniCORN encoder checkpoint is incompatible with ActorCriticTG; "
                "retrain with the canonical paired-encoder pretrain path: "
                f"{checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Paired UniCORN encoder checkpoint key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    @staticmethod
    def _is_released_unicorn_checkpoint(checkpoint_path: str) -> bool:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        return bool(
            isinstance(state, dict)
            and isinstance(state.get("query_token"), torch.Tensor)
            and any(key.startswith("encoder.patch.") for key in state)
            and any(key.startswith("query_encoder.patch.") for key in state)
        )

    def _load_released_unicorn_encoder_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError(
                f"released UniCORN checkpoint lacks its model state: {checkpoint_path}"
            )
        selected = {
            key: value
            for key, value in state.items()
            if key == "query_token" or key.startswith("encoder.")
        }
        if tuple(state["query_token"].shape) != (128,):
            raise RuntimeError(
                f"released UniCORN query_token must be 128D: {checkpoint_path}"
            )
        level_one_weight = state.get("encoder.patch.1.mlp.model.0.linear.weight")
        if not isinstance(level_one_weight, torch.Tensor) or tuple(level_one_weight.shape) != (
            256,
            96,
        ):
            raise RuntimeError(
                "released UniCORN checkpoint does not contain the expected "
                f"32-point multiresolution branch: {checkpoint_path}"
            )
        try:
            incompatible = self.encoder.load_state_dict(selected, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"released UniCORN encoder is incompatible: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "released UniCORN encoder key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def _load_patch_distance_pointnet_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        raw_state = self._checkpoint_state_dict(checkpoint, checkpoint_path)
        encoder_state = self._extract_unicorn_encoder_state_dict(
            raw_state, checkpoint_path
        )
        if not isinstance(metadata, dict) or metadata.get("schema_version") != _PRETRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError(
                "patch-distance PointNet checkpoint must use pretrain_checkpoint_v1: "
                f"{checkpoint_path}"
            )
        family = (metadata.get("model") or {}).get("family")
        if family != "patch_distance_pointnet":
            raise RuntimeError(
                "patch-distance PointNet checkpoint family mismatch: "
                f"got {family!r} in {checkpoint_path}"
            )
        dims = metadata.get("model_dims") or (metadata.get("model") or {}).get("dims") or {}
        aliases = {"num_pts": ("num_pts", "num_points")}
        for key, expected in expected_dims.items():
            names = aliases.get(key, (key,))
            actual = next((dims[name] for name in names if dims.get(name) is not None), None)
            if actual is not None and int(actual) != int(expected):
                raise RuntimeError(
                    f"patch-distance PointNet {key} mismatch: expected {expected}, "
                    f"got {actual} in {checkpoint_path}"
                )
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"patch-distance PointNet checkpoint is incompatible: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "patch-distance PointNet key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def _load_oracle_pointmesh_pointnet_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = self._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = self._extract_unicorn_encoder_state_dict(raw_state, checkpoint_path)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != _PRETRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError(
                "oracle pointmesh PointNet checkpoint must use pretrain_checkpoint_v1: "
                f"{checkpoint_path}"
            )
        family = (metadata.get("model") or {}).get("family")
        if family != "oracle_pointmesh_pointnet":
            raise RuntimeError(
                "oracle pointmesh PointNet checkpoint family mismatch: "
                f"got {family!r} in {checkpoint_path}"
            )
        dims = metadata.get("model_dims") or (metadata.get("model") or {}).get("dims") or {}
        aliases = {"num_pts": ("num_pts", "num_points")}
        for key, expected in expected_dims.items():
            names = aliases.get(key, (key,))
            actual = next((dims[name] for name in names if dims.get(name) is not None), None)
            if actual is not None and int(actual) != int(expected):
                raise RuntimeError(
                    f"oracle pointmesh PointNet checkpoint {key} mismatch: "
                    f"expected {expected}, got {actual}"
                )
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"oracle pointmesh PointNet checkpoint is incompatible: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "oracle pointmesh PointNet checkpoint key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def _load_oracle_pointcloud_pointnet_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
        load_fitted_weights: bool = True,
        checkpoint_adapter: str = "oracle_pointcloud_pointnet_strict",
    ) -> None:
        if expected_dims != {
            "num_pts": 512,
            "patch_size": 32,
            "encoder_channel": 128,
            "num_patches": 16,
        }:
            raise RuntimeError(
                "oracle point-cloud checkpoint requires dimensions "
                "num_points=512, num_patches=16, patch_size=32, encoder_channel=128"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_adapter in {
            "oracle_pointcloud_pointnet_pretrain_strict",
            "oracle_pointcloud_pointnet_normalized_pretrain_strict",
        }:
            if not load_fitted_weights:
                raise RuntimeError(
                    "Native PointNet pretrain checkpoints require encoder "
                    "weight loading"
                )
            if self.encoder.use_rank10_bottleneck:
                raise RuntimeError(
                    "Native PointNet diffusion checkpoints require the "
                    "direct-128 encoder"
                )
            metadata = (
                checkpoint.get("metadata")
                if isinstance(checkpoint, dict)
                else None
            )
            if (
                not isinstance(metadata, dict)
                or metadata.get("schema_version")
                != _PRETRAIN_CHECKPOINT_SCHEMA
            ):
                raise RuntimeError(
                    "Native PointNet checkpoint must use "
                    f"{_PRETRAIN_CHECKPOINT_SCHEMA}: {checkpoint_path}"
                )
            family = (metadata.get("model") or {}).get("family")
            if family != "oracle_pointcloud_pointnet":
                raise RuntimeError(
                    "Native PointNet checkpoint family mismatch: "
                    f"got {family!r} in {checkpoint_path}"
                )
            dims = metadata.get("model_dims") or {}
            expected_metadata = {
                "num_pts": 512,
                "num_patches": 16,
                "patch_size": 32,
                "encoder_channel": 128,
                "feature_dim": 128,
                "pointcloud_feature_mode": "fast11",
                "pointcloud_use_rank10_bottleneck": False,
                "pointcloud_token_mode": "patches",
            }
            if (
                checkpoint_adapter
                == "oracle_pointcloud_pointnet_normalized_pretrain_strict"
            ):
                if self.encoder.input_normalization != "fast11_probe_v1":
                    raise RuntimeError(
                        "Normalized native PointNet checkpoints require "
                        "input_normalization='fast11_probe_v1'"
                    )
                expected_metadata["pointcloud_input_normalization"] = (
                    "fast11_probe_v1"
                )
            elif self.encoder.input_normalization != "identity":
                raise RuntimeError(
                    "Legacy native PointNet checkpoints require "
                    "input_normalization='identity'"
                )
            mismatched = {
                key: (dims.get(key), value)
                for key, value in expected_metadata.items()
                if dims.get(key) != value
            }
            if mismatched:
                raise RuntimeError(
                    "Native PointNet checkpoint metadata mismatch: "
                    f"{mismatched} in {checkpoint_path}"
                )
            raw_state = checkpoint.get("model")
            if not isinstance(raw_state, dict):
                raise RuntimeError(
                    "Native PointNet checkpoint lacks model state: "
                    f"{checkpoint_path}"
                )
            encoder_state = {
                key.removeprefix("encoder."): value
                for key, value in raw_state.items()
                if key.startswith("encoder.")
            }
            expected_keys = set(self.encoder.state_dict())
            if set(encoder_state) != expected_keys:
                raise RuntimeError(
                    "Native PointNet checkpoint key mismatch: "
                    f"missing={sorted(expected_keys - set(encoder_state))}, "
                    f"unexpected={sorted(set(encoder_state) - expected_keys)}"
                )
            self.encoder.load_state_dict(encoder_state, strict=True)
            return
        if checkpoint_adapter == "oracle_pointcloud_pointnet_rl_encoder_strict":
            if not load_fitted_weights:
                raise RuntimeError(
                    "RL encoder checkpoints require load_fitted_weights=True"
                )
            raw_state = checkpoint.get("model_state_dict")
            if not isinstance(raw_state, dict):
                raise RuntimeError(
                    "RL encoder checkpoint lacks model_state_dict: "
                    f"{checkpoint_path}"
                )
            prefix = "encoder."
            encoder_state = {
                key.removeprefix(prefix): value
                for key, value in raw_state.items()
                if key.startswith(prefix)
            }
            expected_keys = set(self.encoder.state_dict())
            if set(encoder_state) != expected_keys:
                raise RuntimeError(
                    "RL encoder checkpoint key mismatch: "
                    f"missing={sorted(expected_keys - set(encoder_state))}, "
                    f"unexpected={sorted(set(encoder_state) - expected_keys)}"
                )
            self.encoder.load_state_dict(encoder_state, strict=True)
            return
        if checkpoint_adapter != "oracle_pointcloud_pointnet_strict":
            raise RuntimeError(
                "Unsupported oracle point-cloud checkpoint adapter "
                f"{checkpoint_adapter!r}"
            )
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != (
            "rank10_fast_pointcloud11_v2"
        ):
            raise RuntimeError(
                "oracle point-cloud PointNet checkpoint must use "
                f"rank10_fast_pointcloud11_v2: {checkpoint_path}"
            )
        state = checkpoint.get("model_state_dict")
        normalization = checkpoint.get("normalization")
        if not isinstance(state, dict) or not isinstance(normalization, dict):
            raise RuntimeError(
                f"oracle point-cloud PointNet checkpoint is incomplete: {checkpoint_path}"
            )
        input_mean = normalization.get("input_mean")
        input_std = normalization.get("input_std")
        token_up_weight = checkpoint.get("token_up_weight")
        token_up_bias = checkpoint.get("token_up_bias")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (input_mean, input_std, token_up_weight, token_up_bias)
        ):
            raise RuntimeError(
                "oracle point-cloud PointNet checkpoint lacks tensor normalization "
                f"or token-up fields: {checkpoint_path}"
            )
        normalized_mean = torch.as_tensor(input_mean, dtype=torch.float32)
        normalized_std = torch.as_tensor(input_std, dtype=torch.float32)
        encoder_state = dict(state)
        encoder_state["input_mean"] = normalized_mean
        encoder_state["input_std"] = normalized_std
        encoder_state["token_up.weight"] = torch.as_tensor(token_up_weight, dtype=torch.float32)
        encoder_state["token_up.bias"] = torch.as_tensor(token_up_bias, dtype=torch.float32)
        expected_shapes = {
            "input_mean": (11,),
            "input_std": (11,),
            "token_up.weight": (128, 10),
            "token_up.bias": (128,),
        }
        for key, shape in expected_shapes.items():
            value = encoder_state.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise RuntimeError(
                    f"oracle point-cloud checkpoint {key} must have shape {shape}: "
                    f"{checkpoint_path}"
                )
        if not bool(torch.isfinite(encoder_state["input_mean"]).all()) or not bool(
            torch.isfinite(encoder_state["input_std"]).all()
        ) or bool((encoder_state["input_std"] <= 0).any()):
            raise RuntimeError(
                f"oracle point-cloud checkpoint normalization is invalid: {checkpoint_path}"
            )
        if not load_fitted_weights:
            # Dataset statistics are not learned token information.  Retaining
            # them keeps input scaling identical while every learned mapping
            # (point_mlp, patch_mlp, and 10->128 token_up) starts randomly.
            with torch.no_grad():
                self.encoder.input_mean.copy_(normalized_mean)
                self.encoder.input_std.copy_(normalized_std)
            return
        try:
            incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"oracle point-cloud PointNet checkpoint is incompatible: {checkpoint_path}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "oracle point-cloud PointNet checkpoint key mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    @classmethod
    def _load_unicorn_contact_hidden_heads(
        cls,
        checkpoint_path: str,
        *,
        expected_input_dim: int,
    ) -> tuple[nn.Sequential, nn.Sequential]:
        """Load both pretrained contact MLPs, excluding their scalar classifiers."""

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_state = cls._checkpoint_state_dict(ckpt, checkpoint_path)
        tool = cls._contact_hidden_head_from_state(
            raw_state,
            head_name="tool_contact_head",
            checkpoint_path=checkpoint_path,
            expected_input_dim=expected_input_dim,
        )
        obj = cls._contact_hidden_head_from_state(
            raw_state,
            head_name="obj_contact_head",
            checkpoint_path=checkpoint_path,
            expected_input_dim=expected_input_dim,
        )
        return tool, obj

    def _load_encoder_token_pca(
        self,
        pca_path: str | None,
        *,
        rank: int,
        expected_dim: int,
    ) -> None:
        if not pca_path:
            raise ValueError("encoder-token PCA projection requires a PCA artifact path")
        payload = torch.load(pca_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("token_stage") != "encoder_pre_mlp":
            raise RuntimeError(f"invalid pre-MLP encoder-token PCA artifact: {pca_path}")
        for prefix in ("tool", "obj"):
            basis = torch.as_tensor(payload.get(f"{prefix}_basis"), dtype=torch.float32)
            mean = torch.as_tensor(payload.get(f"{prefix}_mean"), dtype=torch.float32)
            if basis.shape != (expected_dim, expected_dim):
                raise RuntimeError(
                    f"{prefix}_basis must have shape {(expected_dim, expected_dim)}, "
                    f"got {tuple(basis.shape)}"
                )
            if mean.shape != (expected_dim,):
                raise RuntimeError(
                    f"{prefix}_mean must have shape {(expected_dim,)}, got {tuple(mean.shape)}"
                )
            selected = basis[:rank].contiguous()
            gram = selected @ selected.T
            if not torch.allclose(
                gram,
                torch.eye(rank, dtype=gram.dtype),
                atol=2.0e-4,
                rtol=2.0e-4,
            ):
                raise RuntimeError(f"{prefix}_basis is not orthonormal: {pca_path}")
            self.register_buffer(f"{prefix}_encoder_pca_basis", selected, persistent=True)
            self.register_buffer(f"{prefix}_encoder_pca_mean", mean, persistent=True)

    @staticmethod
    def _pca_reconstruct(
        tokens: torch.Tensor,
        mean: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        centered = tokens - mean
        return mean + torch.matmul(torch.matmul(centered, basis.T), basis)

    def _initialize_encoder_token_bottleneck(
        self,
        pca_path: str | None,
        *,
        rank: int,
        expected_dim: int,
    ) -> None:
        """Initialize a shared trainable linear bottleneck from pooled PCA."""

        if not pca_path:
            raise ValueError("encoder-token bottleneck requires a PCA artifact path")
        payload = torch.load(pca_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("token_stage") != "encoder_pre_mlp":
            raise RuntimeError(f"invalid pre-MLP encoder-token PCA artifact: {pca_path}")

        group_stats: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for prefix in ("tool", "obj"):
            count = int(payload.get(f"{prefix}_count", 0))
            mean = torch.as_tensor(payload.get(f"{prefix}_mean"), dtype=torch.float64)
            basis = torch.as_tensor(payload.get(f"{prefix}_basis"), dtype=torch.float64)
            eigenvalues = torch.as_tensor(
                payload.get(f"{prefix}_eigenvalues"), dtype=torch.float64
            )
            if count < 2:
                raise RuntimeError(f"{prefix}_count must be at least 2 in {pca_path}")
            if mean.shape != (expected_dim,):
                raise RuntimeError(
                    f"{prefix}_mean must have shape {(expected_dim,)}, got {tuple(mean.shape)}"
                )
            if basis.shape != (expected_dim, expected_dim):
                raise RuntimeError(
                    f"{prefix}_basis must have shape {(expected_dim, expected_dim)}, "
                    f"got {tuple(basis.shape)}"
                )
            if eigenvalues.shape != (expected_dim,):
                raise RuntimeError(
                    f"{prefix}_eigenvalues must have shape {(expected_dim,)}, "
                    f"got {tuple(eigenvalues.shape)}"
                )
            covariance = basis.T @ (eigenvalues.clamp_min(0).unsqueeze(1) * basis)
            group_stats.append((count, mean, covariance))

        total_count = sum(count for count, _, _ in group_stats)
        pooled_mean = sum(count * mean for count, mean, _ in group_stats) / total_count
        pooled_scatter = torch.zeros(
            expected_dim, expected_dim, dtype=torch.float64
        )
        for count, mean, covariance in group_stats:
            mean_delta = mean - pooled_mean
            pooled_scatter += (count - 1) * covariance
            pooled_scatter += count * torch.outer(mean_delta, mean_delta)
        pooled_covariance = pooled_scatter / (total_count - 1)
        pooled_covariance = 0.5 * (pooled_covariance + pooled_covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(pooled_covariance)
        order = torch.argsort(eigenvalues, descending=True)
        selected_basis = eigenvectors[:, order[:rank]].T.float().contiguous()
        pooled_mean = pooled_mean.float()

        self.encoder_token_bottleneck_down = nn.Linear(expected_dim, rank, bias=True)
        self.encoder_token_bottleneck_up = nn.Linear(rank, expected_dim, bias=True)
        with torch.no_grad():
            self.encoder_token_bottleneck_down.weight.copy_(selected_basis)
            self.encoder_token_bottleneck_down.bias.copy_(-selected_basis @ pooled_mean)
            self.encoder_token_bottleneck_up.weight.copy_(selected_basis.T)
            self.encoder_token_bottleneck_up.bias.copy_(pooled_mean)

    def _apply_encoder_token_bottleneck(self, tokens: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_token_bottleneck_down(tokens)
        return self.encoder_token_bottleneck_up(latent)

    @staticmethod
    def _contact_hidden_head_from_state(
        state_dict: dict[str, torch.Tensor],
        *,
        head_name: str,
        checkpoint_path: str,
        expected_input_dim: int,
    ) -> nn.Sequential:
        prefix = next(
            (
                candidate
                for candidate in (f"module.{head_name}.", f"{head_name}.")
                if any(key.startswith(candidate) for key in state_dict)
            ),
            None,
        )
        if prefix is None:
            raise RuntimeError(
                f"UniCORN checkpoint lacks pretrained {head_name}: {checkpoint_path}"
            )
        weights: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for key, weight in state_dict.items():
            if not key.startswith(prefix) or not key.endswith(".weight") or weight.ndim != 2:
                continue
            layer_text = key[len(prefix) : -len(".weight")]
            if not layer_text.isdigit():
                continue
            bias_key = f"{prefix}{layer_text}.bias"
            bias = state_dict.get(bias_key)
            if bias is None:
                raise RuntimeError(f"UniCORN contact-head layer lacks bias: {bias_key}")
            weights.append((int(layer_text), weight, bias))
        weights.sort(key=lambda item: item[0])
        if len(weights) < 2 or tuple(weights[-1][1].shape)[0] != 1:
            raise RuntimeError(
                f"UniCORN {head_name} must end in a scalar classifier: {checkpoint_path}"
            )
        hidden = weights[:-1]
        if int(hidden[0][1].shape[1]) != int(expected_input_dim):
            raise RuntimeError(
                f"UniCORN {head_name} input mismatch: expected {expected_input_dim}, "
                f"got {hidden[0][1].shape[1]}"
            )
        if int(hidden[-1][1].shape[0]) != int(expected_input_dim):
            raise RuntimeError(
                f"UniCORN {head_name} last hidden layer must be {expected_input_dim}D, "
                f"got {hidden[-1][1].shape[0]}"
            )
        layers: list[nn.Module] = []
        for _, weight, bias in hidden:
            linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            layers.extend((linear, nn.ReLU()))
        return nn.Sequential(*layers)

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
        if any(key.startswith(("cloud_encoder.", "global_conditioner.")) for key in state_dict):
            return dict(state_dict)
        raise RuntimeError(
            "UniCORN checkpoint does not contain the paired encoder under "
            f"'encoder.'/'module.encoder.': {checkpoint_path}"
        )

    @staticmethod
    def _validate_unicorn_checkpoint_metadata(
        metadata: Any,
        expected_dims: dict[str, int],
        checkpoint_path: str,
    ) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"Paired UniCORN checkpoint missing metadata schema '{_PRETRAIN_CHECKPOINT_SCHEMA}': "
                f"{checkpoint_path}"
            )
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
        aliases = {"num_pts": ("num_pts", "num_points")}
        for key, expected in expected_dims.items():
            names = aliases.get(key, (key,))
            actual = next((dims[name] for name in names if dims.get(name) is not None), None)
            if actual is not None and int(actual) != int(expected):
                raise RuntimeError(
                    f"UniCORN checkpoint dim mismatch for {key}: expected {expected}, "
                    f"got {actual} in {checkpoint_path}"
                )

    def _load_tce_encoder_checkpoint(
        self,
        checkpoint_path: str,
        *,
        expected_dims: dict[str, int],
        expected_vit_attention_mode: str,
        expected_kinematic_conditioning: bool,
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        raw_state = self._checkpoint_state_dict(ckpt, checkpoint_path)
        encoder_state = self._extract_tce_encoder_state_dict(raw_state, checkpoint_path)
        self._validate_tce_checkpoint_metadata(
            metadata,
            expected_dims,
            checkpoint_path,
            expected_vit_attention_mode=expected_vit_attention_mode,
            expected_kinematic_conditioning=expected_kinematic_conditioning,
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
        raise RuntimeError(f"Pretrain checkpoint has no state_dict payload: {checkpoint_path}")

    @staticmethod
    def _validate_tce_checkpoint_metadata(
        metadata: Any,
        expected_dims: dict[str, int],
        checkpoint_path: str,
        *,
        expected_vit_attention_mode: str,
        expected_kinematic_conditioning: bool,
    ) -> None:
        if not isinstance(metadata, dict):
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
        contract = dims.get("vit_attention_contract")
        actual_attention_mode = dims.get("vit_attention_mode")
        legacy_joint_self = (
            contract is None
            and expected_vit_attention_mode == "joint_self"
            and actual_attention_mode == "joint_self"
        )
        if contract != _VIT_ATTENTION_CONTRACT and not legacy_joint_self:
            raise RuntimeError(
                "TCE checkpoint predates explicit attention propagation and "
                "cannot be trusted for this attention mode: expected "
                f"vit_attention_contract="
                f"{_VIT_ATTENTION_CONTRACT!r}, got {contract!r} in {checkpoint_path}"
            )
        if actual_attention_mode != expected_vit_attention_mode:
            raise RuntimeError(
                "TCE checkpoint attention mismatch: expected "
                f"{expected_vit_attention_mode!r}, got "
                f"{actual_attention_mode!r} in {checkpoint_path}"
            )
        actual_kinematic = bool(dims.get("kinematic_conditioning", False))
        if actual_kinematic != bool(expected_kinematic_conditioning):
            raise RuntimeError(
                "TCE checkpoint kinematic-conditioning mismatch: expected "
                f"{bool(expected_kinematic_conditioning)}, got "
                f"{actual_kinematic} in {checkpoint_path}"
            )
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

        encoder_kwargs = {}
        if self.encoder_backend == "oracle_patch":
            oracle_sdf = parts["oracle_mesh_signed_sdf"]
            if oracle_sdf.shape[-1] != 2 * self.num_points:
                raise RuntimeError(
                    "oracle_patch requires exact per-point mesh SDF observation with "
                    f"{2 * self.num_points} values, got {oracle_sdf.shape[-1]}"
                )
            encoder_kwargs = {
                "obj_signed_sdf": oracle_sdf[:, : self.num_points],
                "tool_signed_sdf": oracle_sdf[:, self.num_points :],
            }
        elif self.encoder_backend == "oracle_pointmesh_pointnet":
            oracle_distance = parts["oracle_mesh_unsigned_distance"]
            if oracle_distance.shape[-1] != 2 * self.num_points:
                raise RuntimeError(
                    "oracle_pointmesh_pointnet requires exact per-point unsigned mesh "
                    f"distance with {2 * self.num_points} values, got "
                    f"{oracle_distance.shape[-1]}"
                )
            encoder_kwargs = {
                "obj_unsigned_distance": oracle_distance[:, : self.num_points],
                "tool_unsigned_distance": oracle_distance[:, self.num_points :],
            }
        if self.kinematic_conditioning:
            encoder_kwargs["kinematic_tool_clouds"] = parts[
                "kinematic_gripper_clouds"
            ]
        if self.freeze_encoder:
            with torch.no_grad():
                res = self.encoder.encode(tool_cloud_rel, object_cloud_rel, **encoder_kwargs)
        else:
            res = self.encoder.encode(tool_cloud_rel, object_cloud_rel, **encoder_kwargs)

        all_tokens = res.fused_tokens  # (B, 2P, D)
        if self.encoder_backend == "patch_distance_pointnet":
            patch_centers = torch.cat(
                (res.tool_patch_centers, res.obj_patch_centers), dim=1
            )
            all_tokens = all_tokens + self.patch_distance_position_embedding(
                patch_centers / self.patch_distance_patch_center_scale_m
            )
            tool_body = self.patch_distance_body_embedding[0].view(1, 1, -1)
            object_body = self.patch_distance_body_embedding[1].view(1, 1, -1)
            all_tokens = torch.cat(
                (
                    all_tokens[:, : self.num_patches_per_cloud] + tool_body,
                    all_tokens[:, self.num_patches_per_cloud :] + object_body,
                ),
                dim=1,
            )
        if self.encoder_backend == "tce" and self.encoder_token_pca_rank < self.token_dim:
            tool_tokens = self._pca_reconstruct(
                all_tokens[:, : self.num_patches_per_cloud],
                self.tool_encoder_pca_mean,
                self.tool_encoder_pca_basis,
            )
            obj_tokens = self._pca_reconstruct(
                all_tokens[
                    :,
                    self.num_patches_per_cloud : 2
                    * self.num_patches_per_cloud,
                ],
                self.obj_encoder_pca_mean,
                self.obj_encoder_pca_basis,
            )
            all_tokens = torch.cat(
                (
                    tool_tokens,
                    obj_tokens,
                    all_tokens[:, 2 * self.num_patches_per_cloud :],
                ),
                dim=1,
            )
        if self.encoder_token_bottleneck_rank < self.token_dim:
            all_tokens = self._apply_encoder_token_bottleneck(all_tokens)
        if (
            self.encoder_backend in {"unicorn", "tce"}
            and self.unicorn_token_source == "contact_head_hidden"
        ):
            tool_tokens = self.unicorn_tool_contact_hidden(
                all_tokens[:, : self.num_patches_per_cloud]
            )
            obj_tokens = self.unicorn_obj_contact_hidden(
                all_tokens[
                    :,
                    self.num_patches_per_cloud : 2
                    * self.num_patches_per_cloud,
                ]
            )
            all_tokens = torch.cat(
                (
                    tool_tokens,
                    obj_tokens,
                    all_tokens[:, 2 * self.num_patches_per_cloud :],
                ),
                dim=1,
            )

        # Strict context:
        # [tool_bbox_center-object_bbox_center, object_bbox_center, hand_state,
        #  robot_state, previous_action, relative_goal_pose, physics]
        ctx_vec = build_context_vector(parts)

        return all_tokens, ctx_vec

    def _pointcloud_preprocessing_inputs(self, observations: torch.Tensor):
        """Return canonically centered clouds plus the policy context."""
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
        return tool_cloud_rel, object_cloud_rel, build_context_vector(parts)

    def get_trainable_preprocessing_cache(self, observations: torch.Tensor):
        """Compute fixed search indices once for each rollout observation."""
        if not self.supports_trainable_preprocessing_cache:
            raise RuntimeError("trainable preprocessing cache is unavailable for this encoder")
        tool_cloud_rel, object_cloud_rel, ctx_vec = self._pointcloud_preprocessing_inputs(
            observations
        )
        prepared = self.encoder.prepare_geometry(tool_cloud_rel, object_cloud_rel)
        return prepared.indices, ctx_vec

    def materialize_trainable_preprocessing(
        self,
        observations: torch.Tensor,
        prepared_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the trainable PointNet without repeating FPS or nearest search."""
        if not self.supports_trainable_preprocessing_cache:
            raise RuntimeError("trainable preprocessing cache is unavailable for this encoder")
        tool_cloud_rel, object_cloud_rel, _ = self._pointcloud_preprocessing_inputs(
            observations
        )
        return self.encoder.encode_prepared(
            tool_cloud_rel,
            object_cloud_rel,
            prepared_indices,
        ).fused_tokens

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

    def forward(self):
        raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
            for name in ("unicorn_tool_contact_hidden", "unicorn_obj_contact_hidden"):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

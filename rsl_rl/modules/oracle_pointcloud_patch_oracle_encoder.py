"""RL encoder for the analytic 35D point-cloud patch oracle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from pretrain.patch_oracle_probe import DeepPatchOracleToRankToken
from pretrain.pointcloud_patch_oracle import (
    FAST_POINTCLOUD_PATCH_FEATURE_NAMES,
    build_fast_pointcloud_patch_features,
)
from rsl_rl.modules.oracle_pointcloud_pointnet_encoder import (
    OraclePointCloudEncodeResult,
    OraclePointCloudPointNetEncoder,
    OraclePointCloudPreparedGeometry,
)


class OraclePointCloudPatchOracleEncoder(OraclePointCloudPointNetEncoder):
    """Map analytic point-cloud patch summaries to reconstructed 128D tokens.

    The inherited code is used only for FPS/KNN geometry preparation and its
    compact rollout cache.  PointNet modules are removed at construction.  The
    trainable path is strictly ``35D analytic features -> deep patch MLP -> 10D
    fitted token -> source 10D-to-128D reconstruction``.
    """

    CHECKPOINT_SCHEMA = "rank10_patch_oracle_probe_v1"
    CHECKPOINT_MODEL_NAME = "fast_patch_oracle35"

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        nearest_frame_batch_size: int = 64,
    ) -> None:
        super().__init__(
            num_points=num_points,
            num_patches=num_patches,
            patch_size=patch_size,
            feature_dim=feature_dim,
            nearest_frame_batch_size=nearest_frame_batch_size,
            token_mode="patches",
        )
        # This is a separate analytic encoder.  Do not leave dormant PointNet
        # parameters in DDP or in RL checkpoints.
        del self.point_mlp
        del self.patch_mlp
        del self.token_up
        del self.input_mean
        del self.input_std

        feature_count = len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES)
        self.patch_oracle = DeepPatchOracleToRankToken(input_dim=feature_count)
        self.token_up = nn.Linear(10, feature_dim)
        self.register_buffer("feature_mean", torch.zeros(feature_count), persistent=True)
        self.register_buffer("feature_std", torch.ones(feature_count), persistent=True)
        self.register_buffer("target_mean", torch.zeros(10), persistent=True)
        self.register_buffer("target_std", torch.ones(10), persistent=True)

    @staticmethod
    def _require_vector(
        values: Mapping[str, Any], key: str, width: int, checkpoint_path: Path
    ) -> torch.Tensor:
        value = values.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (width,):
            raise RuntimeError(
                f"analytic point-cloud checkpoint {key} must have shape ({width},): "
                f"{checkpoint_path}"
            )
        result = value.detach().to(dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError(
                f"analytic point-cloud checkpoint {key} contains non-finite values: "
                f"{checkpoint_path}"
            )
        return result

    @staticmethod
    def _source_rank10_reconstruction(
        checkpoint: Mapping[str, Any], checkpoint_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        weight = checkpoint.get("token_up_weight")
        bias = checkpoint.get("token_up_bias")
        source_path = str(checkpoint.get("source_rl_checkpoint") or "")
        if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
            if tuple(weight.shape) != (128, 10) or tuple(bias.shape) != (128,):
                raise RuntimeError(
                    f"embedded rank-10 reconstruction has invalid shape: {checkpoint_path}"
                )
            return weight.float(), bias.float(), source_path

        manifest_path = checkpoint_path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(
                "analytic point-cloud checkpoint lacks an embedded rank-10 reconstruction "
                f"and adjacent manifest: {checkpoint_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = str(manifest.get("source_rl_checkpoint") or "")
        if not source_path:
            raise RuntimeError(
                f"analytic point-cloud manifest lacks source_rl_checkpoint: {manifest_path}"
            )
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"source rank-10 RL checkpoint is missing: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise RuntimeError(f"source rank-10 RL checkpoint lacks model_state_dict: {source}")
        weight = state.get("encoder_token_bottleneck_up.weight")
        bias = state.get("encoder_token_bottleneck_up.bias")
        if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (128, 10):
            raise RuntimeError(f"source rank-10 checkpoint lacks 128x10 token-up weight: {source}")
        if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (128,):
            raise RuntimeError(f"source rank-10 checkpoint lacks 128D token-up bias: {source}")
        return weight.float(), bias.float(), source_path

    def load_fitted_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"analytic point-cloud checkpoint must be a mapping: {path}")
        if checkpoint.get("schema_version") != self.CHECKPOINT_SCHEMA:
            raise RuntimeError(
                f"analytic point-cloud checkpoint schema must be {self.CHECKPOINT_SCHEMA}: {path}"
            )
        if checkpoint.get("model_name") != self.CHECKPOINT_MODEL_NAME:
            raise RuntimeError(
                f"analytic point-cloud checkpoint model_name must be "
                f"{self.CHECKPOINT_MODEL_NAME!r}: {path}"
            )
        if tuple(checkpoint.get("feature_names") or ()) != FAST_POINTCLOUD_PATCH_FEATURE_NAMES:
            raise RuntimeError(f"analytic point-cloud feature contract mismatch: {path}")
        state = checkpoint.get("model_state_dict")
        normalization = checkpoint.get("normalization")
        if not isinstance(state, Mapping) or not isinstance(normalization, Mapping):
            raise RuntimeError(f"analytic point-cloud checkpoint is incomplete: {path}")
        try:
            self.patch_oracle.load_state_dict(dict(state), strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"analytic point-cloud deep MLP is incompatible: {path}") from exc

        feature_count = len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES)
        feature_mean = self._require_vector(normalization, "feature_mean", feature_count, path)
        feature_std = self._require_vector(normalization, "feature_std", feature_count, path)
        target_mean = self._require_vector(normalization, "target_mean", 10, path)
        target_std = self._require_vector(normalization, "target_std", 10, path)
        if bool((feature_std <= 0).any()) or bool((target_std <= 0).any()):
            raise RuntimeError(f"analytic point-cloud normalization std must be positive: {path}")
        token_up_weight, token_up_bias, source_path = self._source_rank10_reconstruction(
            checkpoint, path
        )
        with torch.no_grad():
            self.feature_mean.copy_(feature_mean)
            self.feature_std.copy_(feature_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
            self.token_up.weight.copy_(token_up_weight)
            self.token_up.bias.copy_(token_up_bias)
        return {
            "checkpoint": str(path),
            "source_rl_checkpoint": source_path,
            "metrics": dict(checkpoint.get("metrics") or {}),
        }

    def _patch_tokens(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        distance: torch.Tensor,
        direction: torch.Tensor,
        *,
        is_tool: bool,
    ) -> torch.Tensor:
        point_features = self._raw_point_inputs(
            patches, centers, distance, direction, is_tool=is_tool
        )
        features = build_fast_pointcloud_patch_features(point_features)
        normalized = ((features - self.feature_mean) / self.feature_std).clamp(-12.0, 12.0)
        rank10_normalized = self.patch_oracle(normalized)
        rank10 = rank10_normalized * self.target_std + self.target_mean
        return self.token_up(rank10)

    def encode_prepared(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        prepared: OraclePointCloudPreparedGeometry | torch.Tensor,
    ) -> OraclePointCloudEncodeResult:
        geometry = self._materialize_prepared_geometry(tool_pc, obj_pc, prepared)
        tool_tokens = self._patch_tokens(
            geometry.tool_patches,
            geometry.tool_patch_centers,
            geometry.tool_distance,
            geometry.tool_direction,
            is_tool=True,
        )
        obj_tokens = self._patch_tokens(
            geometry.obj_patches,
            geometry.obj_patch_centers,
            geometry.obj_distance,
            geometry.obj_direction,
            is_tool=False,
        )
        return OraclePointCloudEncodeResult(
            fused_tokens=torch.cat((tool_tokens, obj_tokens), dim=1),
            tool_patch_idx=geometry.tool_patch_idx,
            obj_patch_idx=geometry.obj_patch_idx,
            tool_patch_centers=geometry.tool_patch_centers,
            obj_patch_centers=geometry.obj_patch_centers,
        )

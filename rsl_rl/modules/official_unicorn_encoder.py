"""Adapter for the authors' released UniCORN representation checkpoint.

The upstream representation uses the original ``MLPEncoder`` implementation
under :mod:`rsl_rl.modules.models.cloud.unicorn`.  Its multiresolution model was
trained with a base patch size of 64; level 1 is the 16-patch, 32-point branch
used by the controlled tool-generalist RL comparison.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from rsl_rl.modules.models.cloud.unicorn import MLPEncoder


class OfficialUnicornPairEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class OfficialUnicornPairEncoder(nn.Module):
    """Apply the shared released UniCORN encoder to both point clouds.

    Only the representation encoder is used by RL.  ``query_token`` is retained
    so the released checkpoint is loaded exactly, but it belongs to the
    pretraining contact decoder path and is not appended to the patch tokens.
    """

    _BASE_PATCH_SIZE = 64
    _PATCH_LEVEL = 1

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if (num_points, num_patches, patch_size, feature_dim, num_layers) != (
            512,
            16,
            32,
            128,
            4,
        ):
            raise ValueError(
                "released UniCORN requires num_points=512, num_patches=16, "
                "patch_size=32, feature_dim=128, and num_layers=4"
            )
        self.num_points = int(num_points)
        self._num_patches = int(num_patches)
        self._feature_dim = int(feature_dim)
        self.query_token = nn.Parameter(torch.zeros(feature_dim))
        self.encoder = MLPEncoder(
            MLPEncoder.Config(
                model_dim=feature_dim,
                patch_size=self._BASE_PATCH_SIZE,
                num_layer=num_layers,
                group_type="fps",
                pos_enc_type="mlp",
                patch_type="mlp",
                encoder_type="xfm",
                num_patch_level=4,
                point_dim=3,
                # The released patch MLP consumes flattened XYZ directly.
                pe_dim=None,
            )
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def _encode_cloud(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-2:] != (self.num_points, 3):
            raise ValueError(
                "released UniCORN expects point clouds shaped "
                f"(B, {self.num_points}, 3), got {tuple(points.shape)}"
            )
        aux: dict[str, torch.Tensor] = {}
        tokens = self.encoder(
            points,
            aux=aux,
            patch_level=self._PATCH_LEVEL,
            group_level=self._PATCH_LEVEL,
        )
        patch_idx = aux["patch_index"]
        patch_centers = aux["patch_center"]
        if tokens.shape[-2:] != (self.num_patches, self.feature_dim):
            raise RuntimeError(
                "released UniCORN produced an unexpected token layout: "
                f"{tuple(tokens.shape)}"
            )
        return tokens, patch_idx, patch_centers

    def encode(
        self, tool_pc: torch.Tensor, obj_pc: torch.Tensor
    ) -> OfficialUnicornPairEncodeResult:
        tool_tokens, tool_idx, tool_centers = self._encode_cloud(tool_pc)
        obj_tokens, obj_idx, obj_centers = self._encode_cloud(obj_pc)
        return OfficialUnicornPairEncodeResult(
            fused_tokens=torch.cat((tool_tokens, obj_tokens), dim=1),
            tool_patch_idx=tool_idx,
            obj_patch_idx=obj_idx,
            tool_patch_centers=tool_centers,
            obj_patch_centers=obj_centers,
        )

    def forward(
        self, tool_pc: torch.Tensor, obj_pc: torch.Tensor
    ) -> OfficialUnicornPairEncodeResult:
        return self.encode(tool_pc, obj_pc)

from __future__ import annotations

import torch

from pretrain.patch_oracle_probe import DeepPatchOracleToRankToken
from pretrain.pointcloud_patch_oracle import (
    FAST_POINTCLOUD_PATCH_FEATURE_NAMES,
    build_fast_pointcloud_patch_features,
)


def _inputs(batch: int = 2, patches: int = 5, points: int = 32) -> torch.Tensor:
    values = torch.randn(batch, patches, points, 11)
    values[..., 6] = values[..., 6].abs()
    body = (torch.arange(patches) % 2).float()
    values[..., 10] = body.view(1, patches, 1)
    # Match the source contract: center is repeated within a patch.
    values[..., 3:6] = values[..., :1, 3:6]
    return values


def test_fast_pointcloud_patch_oracle_has_35_finite_features() -> None:
    features = build_fast_pointcloud_patch_features(_inputs())
    assert features.shape == (2, 5, 35)
    assert len(FAST_POINTCLOUD_PATCH_FEATURE_NAMES) == 35
    assert torch.isfinite(features).all()


def test_fast_pointcloud_patch_oracle_is_patch_equivariant() -> None:
    values = _inputs()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    expected = build_fast_pointcloud_patch_features(values)[:, permutation]
    actual = build_fast_pointcloud_patch_features(values[:, permutation])
    assert torch.equal(actual, expected)


def test_deep_probe_accepts_fast_pointcloud_patch_features() -> None:
    features = build_fast_pointcloud_patch_features(_inputs())
    model = DeepPatchOracleToRankToken(input_dim=features.shape[-1])
    assert model(features).shape == (2, 5, 10)

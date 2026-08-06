from __future__ import annotations

import torch

from scripts.train_rank10_pointwise29_pointnet import (
    POINTWISE29_FEATURE_NAMES,
    Pointwise29PointNet,
    build_pointwise29_features,
)


def test_pointwise29_features_match_direct_geometry_formulas():
    source = torch.zeros(1, 1, 11)
    source[..., 0:3] = torch.tensor([0.03, 0.04, 0.0])
    source[..., 6] = 0.01
    source[..., 7:10] = torch.tensor([1.0, 0.0, 0.0])
    source[..., 10] = 1.0

    result = build_pointwise29_features(source)
    soft_5mm = torch.exp(torch.tensor(-2.0))

    assert result.shape == (1, 1, 29)
    assert len(POINTWISE29_FEATURE_NAMES) == 29
    assert torch.allclose(result[..., 11:14], torch.tensor([0.01, 0.0, 0.0]))
    assert torch.allclose(result[..., 17:20], soft_5mm * source[..., 0:3])
    assert torch.allclose(result[..., 20:23], soft_5mm * source[..., 7:10])
    assert torch.allclose(result[..., 23:26], torch.tensor([0.03, 0.0, 0.0]))
    assert torch.allclose(result[..., 26], torch.tensor([[0.05]]))
    assert torch.allclose(
        result[..., 27], torch.tensor([[(0.04**2 + 0.04**2) ** 0.5]])
    )
    assert torch.allclose(result[..., 28], torch.tensor([[0.04]]))


def test_pointwise29_is_point_equivariant_and_model_has_requested_architecture():
    torch.manual_seed(7)
    source = torch.randn(3, 32, 11)
    source[..., 6] = source[..., 6].abs()
    source[..., 7:10] = torch.nn.functional.normalize(source[..., 7:10], dim=-1)
    permutation = torch.randperm(32)

    direct = build_pointwise29_features(source)
    permuted = build_pointwise29_features(source[:, permutation])
    model = Pointwise29PointNet(hidden_dim=16, patch_hidden_dim=32)

    assert torch.allclose(permuted, direct[:, permutation])
    assert model.point_linear.in_features == 29
    assert model.point_linear.out_features == 16
    assert model(direct).shape == (3, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1722

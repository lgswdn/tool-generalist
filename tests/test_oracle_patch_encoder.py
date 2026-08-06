from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/oracle_patch_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_patch_encoder_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.OraclePatchDistanceEncoder


def _enhanced_encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/oracle_patch_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_patch_enhanced_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.OraclePatchDistanceEncoder


def test_oracle_patch_encoder_has_canonical_token_shape_and_finite_values():
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=16,
    )
    tool = torch.randn(2, 32, 3) * 0.02
    obj = torch.randn(2, 32, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])
    tool_sdf = torch.randn(2, 32) * 0.01
    obj_sdf = torch.randn(2, 32) * 0.01

    result = encoder.encode(
        tool,
        obj,
        tool_signed_sdf=tool_sdf,
        obj_signed_sdf=obj_sdf,
    )

    assert result.fused_tokens.shape == (2, 8, 16)
    assert result.tool_patch_idx.shape == (2, 4, 8)
    assert result.obj_patch_idx.shape == (2, 4, 8)
    assert torch.isfinite(result.fused_tokens).all()


def test_oracle_raw_features_preserve_real_mesh_sdf_contact_and_body_type():
    encoder = _encoder_class()(
        num_points=4,
        num_patches=1,
        patch_size=4,
        feature_dim=8,
        contact_eps=0.002,
        center_scale_m=1.0,
        distance_scale_m=1.0,
    )
    patch = torch.tensor(
        [[[[0.000, 0.0, 0.0], [0.001, 0.0, 0.0], [0.010, 0.0, 0.0], [0.020, 0.0, 0.0]]]]
    )
    center = patch.mean(dim=2)
    patch_sdf = torch.tensor([[[-0.003, 0.001, 0.010, 0.020]]])

    raw = encoder._argmax_raw_features(patch, center, patch_sdf, type_id=1)

    assert raw.shape == (1, 1, 12)
    assert raw[0, 0, 5].item() == 1.0
    assert raw[0, 0, 6:8].tolist() == [0.0, 1.0]
    assert torch.isclose(raw[0, 0, 3], torch.tensor(-0.003))


def test_oracle_pretrain_features_exclude_binary_contact_label():
    encoder = _encoder_class()(
        num_points=4,
        num_patches=1,
        patch_size=4,
        feature_dim=8,
        include_contact_feature=False,
        contact_eps=0.002,
        center_scale_m=1.0,
        distance_scale_m=1.0,
    )
    patch = torch.tensor(
        [[[[0.000, 0.0, 0.0], [0.001, 0.0, 0.0], [0.010, 0.0, 0.0], [0.020, 0.0, 0.0]]]]
    )
    center = patch.mean(dim=2)
    patch_sdf = torch.tensor([[[-0.003, 0.001, 0.010, 0.020]]])

    raw = encoder._argmax_raw_features(patch, center, patch_sdf, type_id=1)

    assert raw.shape == (1, 1, 11)
    # After min/mean SDF, the next two coordinates are directly the body type;
    # no thresholded contact-label coordinate is present between them.
    assert raw[0, 0, 5:7].tolist() == [0.0, 1.0]
    assert torch.isclose(raw[0, 0, 3], torch.tensor(-0.003))


def test_oracle_argmax_explicitly_selects_closest_point_location():
    encoder = _enhanced_encoder_class()(
        num_points=4,
        num_patches=1,
        patch_size=4,
        feature_dim=8,
        contact_eps=0.002,
        center_scale_m=1.0,
        distance_scale_m=1.0,
        patch_relative_scale_m=1.0,
    )
    patch = torch.tensor(
        [[[[0.000, 0.0, 0.0], [0.010, 0.0, 0.0], [0.020, 0.0, 0.0], [0.030, 0.0, 0.0]]]]
    )
    center = patch.mean(dim=2)
    patch_sdf = torch.tensor([[[0.03, 0.02, 0.001, 0.01]]])
    raw = encoder._argmax_raw_features(patch, center, patch_sdf, type_id=0)

    expected = patch[0, 0, 2] - center[0, 0]
    assert raw.shape == (1, 1, 12)
    assert torch.allclose(raw[0, 0, 8:11], expected)


def test_oracle_forbids_point_cloud_distance_fallback():
    encoder = _encoder_class()(num_points=8, num_patches=2, patch_size=4, feature_dim=8)
    cloud = torch.zeros(1, 8, 3)
    try:
        encoder.encode(cloud, cloud)
    except RuntimeError as exc:
        assert "point-cloud distance fallback is forbidden" in str(exc)
    else:
        raise AssertionError("oracle encoder accepted missing real mesh SDF")


def test_oracle_signed_log_distance_emphasizes_near_contact_changes():
    encoder = _enhanced_encoder_class()(
        num_points=4,
        num_patches=1,
        patch_size=4,
        feature_dim=8,
        log_distance_resolution_m=0.005,
        log_distance_cap_m=0.05,
    )
    values = torch.tensor([0.0, 0.005, 0.100, 0.110])
    denom = torch.log1p(torch.tensor(encoder.log_distance_cap_m / encoder.log_distance_resolution_m))
    transformed = torch.log1p(
        values.clamp_max(encoder.log_distance_cap_m) / encoder.log_distance_resolution_m
    ) / denom
    assert transformed[1] - transformed[0] > transformed[3] - transformed[2]
    assert transformed[2] == transformed[3] == 1.0


def test_oracle_experiment_contracts():
    from configs.experiments.panda_general_oracle_patch_full_yes_5k import EXP_CFG as full
    from configs.experiments.panda_general_oracle_patch_gg_from_full_yes_5k import EXP_CFG as gg

    full.validate()
    gg.validate()
    assert full.model.encoder_backend == "oracle_patch"
    assert full.pretrain.enabled is False
    assert full.rl.freeze_encoder is False
    assert full.rl.ppo.max_iterations == 5000
    assert gg.rl.ppo.max_iterations == 15000
    assert full.rl.action.scale == gg.rl.action.scale == 0.06
    assert "/no-contact/oracle_patch/" in gg.rl.init_checkpoint


def test_oracle_contact_pretrain_is_short_full_dataset_control():
    from configs.experiments.oracle_patch_pretrain import EXP_CFG

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_patch"
    assert EXP_CFG.model.oracle_patch.include_contact_feature is False
    assert EXP_CFG.pretrain.mode == "oracle_contact"
    assert EXP_CFG.pretrain.enabled_heads == ["contact"]
    assert EXP_CFG.pretrain.epochs == 3
    assert EXP_CFG.pretrain.dataset_manifest
    assert EXP_CFG.contact_gen.enabled is False
    assert EXP_CFG.rl.enabled is False

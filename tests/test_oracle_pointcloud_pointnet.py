from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import pytest


def _encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/oracle_pointcloud_pointnet_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_pointcloud_test_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointCloudPointNetEncoder


def _patch_oracle_encoder_class():
    root = Path(__file__).parents[1]
    parent_name = "rsl_rl.modules.oracle_pointcloud_pointnet_encoder"
    parent_path = root / "rsl_rl/modules/oracle_pointcloud_pointnet_encoder.py"
    parent_spec = importlib.util.spec_from_file_location(parent_name, parent_path)
    assert parent_spec is not None and parent_spec.loader is not None
    parent = importlib.util.module_from_spec(parent_spec)
    sys.modules[parent_name] = parent
    parent_spec.loader.exec_module(parent)

    path = root / "rsl_rl/modules/oracle_pointcloud_patch_oracle_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_patch_test_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointCloudPatchOracleEncoder


def test_fast_pointcloud_encoder_produces_128d_patch_tokens_without_mesh_input():
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=128,
        nearest_frame_batch_size=2,
    )
    tool = torch.randn(2, 32, 3) * 0.02
    obj = torch.randn(2, 32, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])
    result = encoder.encode(tool, obj)

    assert result.fused_tokens.shape == (2, 8, 128)
    assert result.tool_patch_idx.shape == (2, 4, 8)
    assert result.obj_patch_idx.shape == (2, 4, 8)
    assert torch.isfinite(result.fused_tokens).all()
    assert not any(isinstance(module, torch.nn.TransformerEncoder) for module in encoder.modules())


def test_fast11_probe_normalization_is_explicit_and_nonidentity():
    encoder = _encoder_class()(
        feature_mode="fast11",
        use_rank10_bottleneck=False,
        input_normalization="fast11_probe_v1",
    )

    assert encoder.input_normalization == "fast11_probe_v1"
    assert encoder.input_mean.shape == (11,)
    assert encoder.input_std.shape == (11,)
    assert torch.isfinite(encoder.input_mean).all()
    assert torch.all(encoder.input_std > 0)
    assert not torch.equal(encoder.input_mean, torch.zeros_like(encoder.input_mean))
    assert not torch.equal(encoder.input_std, torch.ones_like(encoder.input_std))


def test_fast11_probe_normalization_rejects_incompatible_feature_mode():
    with pytest.raises(ValueError, match="requires feature_mode='fast11'"):
        _encoder_class()(
            feature_mode="rich21",
            input_normalization="fast11_probe_v1",
        )


def test_cached_geometry_is_compact_and_exactly_reproduces_encoder_tokens():
    torch.manual_seed(7)
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=128,
        nearest_frame_batch_size=2,
    )
    tool = torch.randn(2, 32, 3) * 0.02
    obj = torch.randn(2, 32, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])

    prepared = encoder.prepare_geometry(tool, obj)
    cached = encoder.encode_prepared(tool, obj, prepared)
    direct = encoder.encode(tool, obj)

    assert prepared.indices.dtype == torch.int16
    assert prepared.indices.shape == (2, 4 * 4 * 8 + 2 * 4)
    assert torch.equal(cached.tool_patch_idx, direct.tool_patch_idx)
    assert torch.equal(cached.obj_patch_idx, direct.obj_patch_idx)
    assert torch.allclose(cached.fused_tokens, direct.fused_tokens, atol=1e-7, rtol=1e-7)


def test_batched_tool_object_pointnet_matches_separate_reference_calls():
    torch.manual_seed(17)
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=128,
        nearest_frame_batch_size=2,
    )
    tool = torch.randn(3, 32, 3) * 0.02
    obj = torch.randn(3, 32, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])
    prepared = encoder.prepare_geometry(tool, obj)
    geometry = encoder._materialize_prepared_geometry(tool, obj, prepared)

    expected_tool = encoder._pointnet_tokens(
        geometry.tool_patches,
        geometry.tool_patch_centers,
        geometry.tool_distance,
        geometry.tool_direction,
        is_tool=True,
    )
    expected_object = encoder._pointnet_tokens(
        geometry.obj_patches,
        geometry.obj_patch_centers,
        geometry.obj_distance,
        geometry.obj_direction,
        is_tool=False,
    )
    actual = encoder.encode_prepared(tool, obj, prepared).fused_tokens

    assert torch.allclose(
        actual,
        torch.cat((expected_tool, expected_object), dim=1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_point_token_mode_emits_every_patch_member_without_pooling():
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=128,
        nearest_frame_batch_size=2,
        token_mode="points",
    )
    tool = torch.randn(2, 32, 3) * 0.02
    obj = torch.randn(2, 32, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])
    result = encoder.encode(tool, obj)

    assert encoder.num_patches == 32
    assert result.fused_tokens.shape == (2, 64, 128)
    assert all(parameter.requires_grad for parameter in encoder.point_mlp.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.patch_mlp.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.token_up.parameters())


def test_fast_pointcloud_full_yes_experiment_contract():
    from configs.experiments.panda_general_oracle_pointcloud_pointnet_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_pointcloud_pointnet"
    assert EXP_CFG.model.pretrained_encoder.adapter == "oracle_pointcloud_pointnet_strict"
    assert EXP_CFG.pretrain.enabled is False
    assert EXP_CFG.rl.isaac_task_id == "generated-gripper-v0"
    assert EXP_CFG.rl.observation.include_oracle_mesh_sdf is False
    assert EXP_CFG.rl.observation.include_oracle_mesh_unsigned_distance is False
    assert EXP_CFG.rl.freeze_encoder is False
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06


def test_general_frozen_ggbest_pointnet_experiment_contract():
    from configs.experiments.ce_general_oracle_pointcloud_pointnet_ggbest_frozen_dgn_5k import (
        EXP_CFG,
        GG_BEST_CHECKPOINT,
    )
    from scripts.train import _build_policy_params

    EXP_CFG.validate()
    policy = _build_policy_params(EXP_CFG, GG_BEST_CHECKPOINT)
    assert EXP_CFG.paths_yaml == "configs/paths/ce_general_contact_pretrain.yaml"
    assert EXP_CFG.num_gpus == 8
    assert EXP_CFG.contact_gen.enabled is False
    assert EXP_CFG.pretrain.enabled is False
    assert EXP_CFG.rl.env.robot_mode == "cross_embodiment_gripper"
    assert EXP_CFG.rl.freeze_encoder is True
    assert EXP_CFG.rl.ppo.max_iterations == 5_000
    assert EXP_CFG.model.pretrained_encoder.checkpoint_path == GG_BEST_CHECKPOINT
    assert EXP_CFG.model.pretrained_encoder.adapter == (
        "oracle_pointcloud_pointnet_rl_encoder_strict"
    )
    assert policy["oracle_pointcloud_checkpoint_adapter"] == (
        "oracle_pointcloud_pointnet_rl_encoder_strict"
    )


def test_general_frozen_ggbest_pointnet_gg_contract():
    from configs.experiments.ce_general_oracle_pointcloud_pointnet_ggbest_frozen_gg_15k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.num_gpus == 8
    assert EXP_CFG.rl.env.robot_mode == "cross_embodiment_gripper"
    assert EXP_CFG.rl.freeze_encoder is True
    assert EXP_CFG.rl.ppo.max_iterations == 15_000
    assert EXP_CFG.rl.init_checkpoint.endswith("/model_last.pt")


def test_fast_pointcloud_scratch_full_yes_experiment_contract():
    from configs.experiments.panda_general_oracle_pointcloud_pointnet_scratch_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_pointcloud_pointnet"
    assert EXP_CFG.model.oracle_pointcloud_pointnet.load_fitted_weights is False
    assert EXP_CFG.rl.freeze_encoder is False
    assert EXP_CFG.rl.curriculum.enabled is False
    assert EXP_CFG.rl.curriculum.start_stable_pose_probability == 0.0
    assert EXP_CFG.rl.curriculum.end_stable_pose_probability == 0.0
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06


def test_rich_pointcloud_scratch_full_yes_has_no_checkpoint_or_rank10_bottleneck():
    from configs.experiments.panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    pointcloud = EXP_CFG.model.oracle_pointcloud_pointnet
    assert pointcloud.feature_mode == "rich21"
    assert pointcloud.load_fitted_weights is False
    assert pointcloud.use_rank10_bottleneck is False
    assert EXP_CFG.model.pretrained_encoder.adapter == "oracle_none"
    assert EXP_CFG.model.pretrained_encoder.checkpoint_path is None
    assert EXP_CFG.rl.freeze_encoder is False
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06


def test_point_token_full_yes_experiment_contract():
    from configs.experiments.panda_general_oracle_pointcloud_pointtokens_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_pointcloud_pointnet"
    assert EXP_CFG.model.oracle_pointcloud_pointnet.token_mode == "points"
    assert EXP_CFG.rl.freeze_encoder is False
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06


def test_point_token_mode_is_forwarded_by_isaac_policy_config():
    config_path = (
        Path(__file__).parents[1]
        / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/agents/config/rsl_rl_ppo_cfg.py"
    )
    source = config_path.read_text()
    assert "oracle_pointcloud_token_mode: str" in source
    assert '_policy("oracle_pointcloud_token_mode", "patches")' in source


def test_point_token_v2_uses_a_clean_artifact_name():
    from configs.experiments.panda_general_oracle_pointcloud_pointtokens_v2_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.name == "panda_general_oracle_pointcloud_pointtokens_v2_full_yes_5k"
    assert EXP_CFG.model.oracle_pointcloud_pointnet.token_mode == "points"


def test_analytic_patch_oracle_loads_best_checkpoint_and_reuses_search_cache():
    checkpoint = (
        Path(__file__).parents[1]
        / "artifacts/probes/rank10_patch_pointnet/fast_patch_oracle35/"
        "fast_patch_oracle35_best.pt"
    )
    if not checkpoint.is_file():
        return
    encoder = _patch_oracle_encoder_class()(
        num_points=512,
        num_patches=16,
        patch_size=32,
        feature_dim=128,
        nearest_frame_batch_size=1,
    )
    metadata = encoder.load_fitted_checkpoint(checkpoint)
    assert metadata["metrics"]["epoch"] == 30
    assert not any("point_mlp" in name for name, _ in encoder.named_modules())

    torch.manual_seed(11)
    tool = torch.randn(1, 512, 3) * 0.02
    obj = torch.randn(1, 512, 3) * 0.02 + torch.tensor([0.04, 0.0, 0.0])
    prepared = encoder.prepare_geometry(tool, obj)
    cached = encoder.encode_prepared(tool, obj, prepared)
    direct = encoder.encode(tool, obj)
    assert cached.fused_tokens.shape == (1, 32, 128)
    assert torch.equal(cached.fused_tokens, direct.fused_tokens)


def test_analytic_patch_oracle_full_yes_experiment_contract():
    from configs.experiments.panda_general_oracle_pointcloud_patch_oracle_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_pointcloud_patch_oracle"
    assert (
        EXP_CFG.model.pretrained_encoder.adapter
        == "oracle_pointcloud_patch_oracle_strict"
    )
    assert EXP_CFG.pretrain.enabled is False
    assert EXP_CFG.rl.freeze_encoder is False
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06
    assert EXP_CFG.rl.launch.wandb_project == "ungrasp"

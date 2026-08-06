from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/oracle_pointmesh_pointnet_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_pointmesh_test_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointMeshPointNetEncoder


def test_pointmesh_pointnet_produces_independent_128d_patch_tokens():
    encoder = _encoder_class()(
        num_points=32,
        num_patches=4,
        patch_size=8,
        feature_dim=128,
    )
    tool = torch.randn(2, 32, 3) * 0.02
    obj = torch.randn(2, 32, 3) * 0.02
    tool_distance = torch.rand(2, 32) * 0.1
    obj_distance = torch.rand(2, 32) * 0.1
    result = encoder.encode(
        tool,
        obj,
        tool_unsigned_distance=tool_distance,
        obj_unsigned_distance=obj_distance,
    )
    assert result.fused_tokens.shape == (2, 8, 128)
    assert result.tool_patch_idx.shape == (2, 4, 8)
    assert result.obj_patch_idx.shape == (2, 4, 8)
    assert not any(isinstance(module, torch.nn.TransformerEncoder) for module in encoder.modules())


def test_pointmesh_pointnet_rejects_signed_input():
    encoder = _encoder_class()(
        num_points=8,
        num_patches=2,
        patch_size=4,
        feature_dim=16,
    )
    points = torch.zeros(1, 8, 3)
    distance = torch.zeros(1, 8)
    distance[0, 0] = -0.001
    try:
        encoder.encode(
            points,
            points,
            tool_unsigned_distance=distance,
            obj_unsigned_distance=distance.abs(),
        )
    except RuntimeError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("pointmesh PointNet accepted a signed-distance input")


def test_pointmesh_full_yes_experiment_contract():
    from configs.experiments.panda_general_oracle_pointmesh_pointnet_full_yes_5k import (
        EXP_CFG,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "oracle_pointmesh_pointnet"
    assert EXP_CFG.pretrain.mode == "oracle_pointmesh_contact"
    assert EXP_CFG.pretrain.enabled is True
    assert EXP_CFG.pretrain.epochs == 3
    assert EXP_CFG.rl.isaac_task_id == "generated-gripper-oracle-pointmesh-v0"
    assert EXP_CFG.rl.observation.layout[2] == "oracle_mesh_unsigned_distance"
    assert EXP_CFG.rl.observation.include_oracle_mesh_unsigned_distance is True
    assert EXP_CFG.rl.freeze_encoder is True
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06

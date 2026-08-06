from __future__ import annotations

import importlib
import runpy

import pytest

from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    GENERATED_GRIPPER_PATHS_YAML,
)


POST_SOURCE = "panda_general_native_pointnet_post_original400_dgn_5k"
ORIGINAL_CHILDREN = (
    "panda_general_native_pointnet_post_original400_unfrozen_gg_15k",
    "panda_general_native_pointnet_post_original400_frozen_gg_15k",
)
CE_PAIRS = (
    (
        "ce_general_native_pointnet_post_current_velocity_unfrozen_dgn_5k",
        "ce_general_native_pointnet_post_current_velocity_unfrozen_gg_15k",
        CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    ),
    (
        "ce_prl_native_pointnet_post_current_velocity_unfrozen_dgn_5k",
        "ce_prl_native_pointnet_post_current_velocity_unfrozen_gg_15k",
        GENERATED_GRIPPER_NEW_PATHS_YAML,
    ),
)
NORMALIZED_PRETRAIN_SOURCE = "ce_prl_native_pointnet_normalized_post_pretrain.py"
NORMALIZED_PAIRS = (
    (
        "ce_prl_native_pointnet_normalized_post_frozen_dgn_5k",
        "ce_prl_native_pointnet_normalized_post_frozen_gg_15k",
        True,
    ),
    (
        "ce_prl_native_pointnet_normalized_post_unfrozen_dgn_5k",
        "ce_prl_native_pointnet_normalized_post_unfrozen_gg_15k",
        False,
    ),
)
SAFE_VELOCITY_PAIRS = (
    (
        "panda_general_native_pointnet_post_original400_safe_velocity_frozen_dgn_5k",
        "panda_general_native_pointnet_post_original400_safe_velocity_frozen_gg_15k",
        True,
    ),
    (
        "panda_general_native_pointnet_post_original400_safe_velocity_unfrozen_dgn_5k",
        "panda_general_native_pointnet_post_original400_safe_velocity_unfrozen_gg_15k",
        False,
    ),
)


def _cfg(name: str):
    return importlib.import_module(f"configs.experiments.{name}").EXP_CFG


def test_original400_gg_pair_uses_same_completed_last_parent() -> None:
    unfrozen, frozen = (_cfg(name) for name in ORIGINAL_CHILDREN)

    for cfg in (unfrozen, frozen):
        assert cfg.pretrain_reuse == f"{POST_SOURCE}.py"
        assert cfg.paths_yaml == GENERATED_GRIPPER_PATHS_YAML
        assert cfg.rl.init_checkpoint.endswith("/model_last.pt")
        assert f"/RL/{POST_SOURCE}/" in cfg.rl.init_checkpoint
        assert cfg.rl.resume_checkpoint is None
        assert cfg.rl.ppo.max_iterations == 15_000
        assert cfg.rl.env.generated_parallel_finger_velocity_limit_m_s == 2.61

    assert unfrozen.rl.init_checkpoint == frozen.rl.init_checkpoint
    assert unfrozen.rl.freeze_encoder is False
    assert frozen.rl.freeze_encoder is True


@pytest.mark.parametrize(("parent", "child", "paths_yaml"), CE_PAIRS)
def test_ce_post_pipeline_contract(parent: str, child: str, paths_yaml: str) -> None:
    parent_cfg = _cfg(parent)
    child_cfg = _cfg(child)

    for cfg in (parent_cfg, child_cfg):
        assert cfg.pretrain_reuse == f"{POST_SOURCE}.py"
        assert cfg.paths_yaml == paths_yaml
        assert cfg.model.encoder_backend == "oracle_pointcloud_pointnet"
        assert cfg.model.pretrained_encoder.adapter == (
            "oracle_pointcloud_pointnet_pretrain_strict"
        )
        assert cfg.model.oracle_pointcloud_pointnet.use_rank10_bottleneck is False
        assert cfg.rl.freeze_encoder is False
        assert cfg.rl.env.generated_parallel_finger_velocity_limit_m_s == 2.61

    assert parent_cfg.rl.ppo.max_iterations == 5_000
    assert parent_cfg.rl.launch.wandb_project == "dgn_set"
    assert child_cfg.rl.ppo.max_iterations == 15_000
    assert child_cfg.rl.launch.wandb_project == "ungraspable_set"
    assert child_cfg.rl.init_checkpoint.endswith("/model_last.pt")
    assert child_cfg.rl.resume_checkpoint is None


def test_frozen_dgn_diagnostic_contract() -> None:
    cfg = _cfg("panda_general_native_pointnet_post_original400_frozen_dgn_5k")

    assert cfg.pretrain_reuse == f"{POST_SOURCE}.py"
    assert cfg.rl.freeze_encoder is True
    assert cfg.rl.ppo.max_iterations == 5_000
    assert cfg.rl.ppo.save_interval == 100


@pytest.mark.parametrize(("parent", "child", "frozen"), NORMALIZED_PAIRS)
def test_normalized_ce_prl_post_pipeline_contract(
    parent: str, child: str, frozen: bool
) -> None:
    parent_cfg = _cfg(parent)
    child_cfg = _cfg(child)

    for cfg in (parent_cfg, child_cfg):
        pointnet = cfg.model.oracle_pointcloud_pointnet
        assert cfg.pretrain_reuse == NORMALIZED_PRETRAIN_SOURCE
        assert cfg.paths_yaml == GENERATED_GRIPPER_NEW_PATHS_YAML
        assert cfg.pretrain.mode == "oracle_pointcloud_postcontact"
        assert cfg.pretrain.enabled_heads == ["postcontact"]
        assert pointnet.use_rank10_bottleneck is False
        assert pointnet.input_normalization == "fast11_probe_v1"
        assert cfg.model.pretrained_encoder.adapter == (
            "oracle_pointcloud_pointnet_normalized_pretrain_strict"
        )
        assert cfg.rl.freeze_encoder is frozen
        assert cfg.rl.env.generated_parallel_finger_velocity_limit_m_s == 0.05

    assert parent_cfg.rl.ppo.max_iterations == 5_000
    assert child_cfg.rl.ppo.max_iterations == 15_000
    assert child_cfg.rl.init_checkpoint.endswith("/model_last.pt")


def test_waiter_maps_all_native_post_gg_children() -> None:
    variants = runpy.run_path("scripts/wait_unicorn_full_yes_then_gg.py")[
        "VARIANTS"
    ]
    mapped_children = {entry[1] for entry in variants.values()}

    assert set(ORIGINAL_CHILDREN).issubset(mapped_children)
    assert {child for _, child, _ in CE_PAIRS}.issubset(mapped_children)
    assert {child for _, child, _ in NORMALIZED_PAIRS}.issubset(mapped_children)
    assert {child for _, child, _ in SAFE_VELOCITY_PAIRS}.issubset(mapped_children)


@pytest.mark.parametrize(("parent", "child", "frozen"), SAFE_VELOCITY_PAIRS)
def test_original400_safe_velocity_pipeline_contract(
    parent: str, child: str, frozen: bool
) -> None:
    parent_cfg = _cfg(parent)
    child_cfg = _cfg(child)

    for cfg in (parent_cfg, child_cfg):
        assert cfg.paths_yaml == GENERATED_GRIPPER_PATHS_YAML
        assert cfg.pretrain.enabled is False
        assert cfg.pretrain_reuse is None
        assert cfg.contact_gen.enabled is False
        assert cfg.model.oracle_pointcloud_pointnet.use_rank10_bottleneck is False
        assert cfg.model.oracle_pointcloud_pointnet.input_normalization == "identity"
        assert cfg.rl.freeze_encoder is frozen
        assert cfg.rl.env.generated_parallel_finger_velocity_limit_m_s == 0.05

    assert parent_cfg.rl.init_checkpoint is None
    assert parent_cfg.rl.ppo.max_iterations == 5_000
    assert child_cfg.rl.init_checkpoint.endswith("/model_last.pt")
    assert child_cfg.rl.resume_checkpoint is None
    assert child_cfg.rl.ppo.max_iterations == 15_000

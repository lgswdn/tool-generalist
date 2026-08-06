from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg
from scripts.train import _build_policy_params, build_rl_runtime_spec
from utils.config.loader import load_exp_cfg
from utils.artifacts.resolver import resolve_artifacts
from utils.config.paths import load_project_paths
from utils.experiment.rl_runtime_spec import validate_runtime_spec
from utils.geometry.generated_gripper_kinematics import (
    CachedGripperPointKinematics,
)
from utils.geometry.gripper_cloud_cache import GripperCloudCache


def test_kinematic_encoder_shares_patch_pointnet_and_adds_three_tokens():
    encoder = TCEPointCloudEncoder(
        TCEPointCloudEncoderCfg(
            num_pts=32,
            patch_size=8,
            encoder_channel=32,
            vit_depth=1,
            vit_heads=4,
            freeze=False,
            vit_attention_mode="joint_self",
            kinematic_conditioning=True,
            kinematic_attention_layers=1,
        )
    )
    tool = torch.randn(2, 32, 3)
    obj = torch.randn(2, 32, 3)
    states = torch.randn(2, 3, 32, 3)
    result = encoder.encode(tool, obj, kinematic_tool_clouds=states)
    assert result.fused_tokens.shape == (2, 11, 32)
    assert sum(1 for module in encoder.modules() if module is encoder.patch_enc) == 1


def test_parallel_motion_uses_nearest_canonical_cache_bin():
    states = torch.zeros(128, 4, 3)
    states[:, 1, 1] = torch.arange(128) / 127.0
    states[:, 2, 1] = -torch.arange(128) / 127.0
    cache = GripperCloudCache(
        gripper_id="test",
        source_manifest="/tmp/test.json",
        source_asset_root="/tmp/test",
        opening_fractions=torch.linspace(0.0, 1.0, 128),
        body_names=("palm", "left", "right"),
        point_body_index=torch.tensor([0, 1, 2, 0]),
        points_body=torch.zeros(4, 3),
        state_clouds_palm=states,
    )
    points = states[32] - torch.tensor([0.1, 0.0, 0.0])
    kinematics = CachedGripperPointKinematics(
        opening_fraction=0.25,
        bbox_center=torch.tensor([0.1, 0.0, 0.0]),
        cache=cache,
    )
    moved = kinematics.cloud_at_fraction(
        points, 0.75, canonical_local=False
    )
    assert torch.equal(moved, states[95] - kinematics.bbox_center)
    assert kinematics.static_state_clouds(points).shape == (3, 4, 3)


def test_new_experiment_is_isolated_and_observation_layout_matches():
    original = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_concavity_biased_dgn_10k.py"
    )
    kinematic = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_"
        "concavity_biased_kinematic_dgn_10k.py"
    )
    original_refs = {ref.stage: ref.config_hash for ref in resolve_artifacts(original).stages}
    kinematic_refs = {ref.stage: ref.config_hash for ref in resolve_artifacts(kinematic).stages}
    assert original_refs["contact_gen"] == kinematic_refs["contact_gen"]
    assert original_refs["pretrain"] != kinematic_refs["pretrain"]
    assert original_refs["rl"] != kinematic_refs["rl"]

    assert (
        kinematic.rl.effective_observation_dim
        - original.rl.effective_observation_dim
        == 3 * 512 * 3
    )


def test_four_layer_kinematic_experiment_uses_5k_dgn():
    cfg = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d4_full_nonpenetrating_contact_"
        "concavity_biased_kinematic_dgn_5k.py"
    )
    cfg.validate()
    assert cfg.model.tce.vit_depth == 4
    assert cfg.model.tce.kinematic_conditioning.attention_layers == 4
    assert cfg.rl.ppo.max_iterations == 5_000
    policy = _build_policy_params(cfg, "unused-test-checkpoint.pt")
    assert policy["kinematic_conditioning"] is True
    assert policy["kinematic_attention_layers"] == 4
    assert cfg.rl.effective_observation_dim == 7_737


def test_general_d4_matrix_uses_combined_grippers_and_shared_contact_data():
    for contact_quality in ("paper", "concavity_global"):
        configs = {}
        hashes = {}
        for architecture in ("raw", "kinematic"):
            name = (
                f"ce_general_d4_full_{contact_quality}_{architecture}_dgn_5k"
            )
            cfg = load_exp_cfg(f"configs/experiments/{name}.py")
            configs[architecture] = cfg
            hashes[architecture] = {
                ref.stage: ref.config_hash for ref in resolve_artifacts(cfg).stages
            }
            assert cfg.paths_yaml == "configs/paths/ce_general_contact_pretrain.yaml"
            assert cfg.num_gpus == 8
            assert cfg.rl.env.robot_mode == "cross_embodiment_gripper"
            assert cfg.model.tce.vit_depth == 4
            assert cfg.rl.ppo.max_iterations == 5_000

        assert (
            hashes["raw"]["contact_gen"]
            == hashes["kinematic"]["contact_gen"]
        )
        assert hashes["raw"]["pretrain"] != hashes["kinematic"]["pretrain"]
        assert configs["raw"].model.tce.kinematic_conditioning.enabled is False
        assert (
            configs["kinematic"].model.tce.kinematic_conditioning.enabled
            is True
        )
        assert configs["raw"].rl.effective_observation_dim == 3_129
        assert configs["kinematic"].rl.effective_observation_dim == 7_737


def test_general_kinematic_runtime_accepts_cross_embodiment_mode(tmp_path):
    cfg = load_exp_cfg(
        "configs/experiments/"
        "ce_general_d4_full_concavity_global_kinematic_dgn_5k.py"
    )
    spec = build_rl_runtime_spec(
        cfg,
        load_project_paths(cfg.paths_yaml),
        tmp_path,
        encoder_checkpoint_override="/tmp/test-kinematic-encoder.pt",
    )

    validate_runtime_spec(asdict(spec))


def test_paper_contact_kinematic_experiment_reuses_original_contact_data():
    original = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d4_full_paper_contact_dgn_5k.py"
    )
    kinematic = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d4_full_paper_contact_kinematic_dgn_5k.py"
    )
    original_refs = {
        ref.stage: ref.config_hash for ref in resolve_artifacts(original).stages
    }
    kinematic_refs = {
        ref.stage: ref.config_hash for ref in resolve_artifacts(kinematic).stages
    }

    assert original_refs["contact_gen"] == kinematic_refs["contact_gen"]
    assert original_refs["pretrain"] != kinematic_refs["pretrain"]
    assert original.contact_gen.require_tool_tip_anchor is False
    assert kinematic.contact_gen.require_tool_tip_anchor is False
    assert kinematic.contact_gen.name == "contact_gen_prl_paper_contact_500k"
    assert kinematic.model.tce.vit_depth == 4
    assert kinematic.model.tce.kinematic_conditioning.enabled is True
    assert kinematic.model.tce.kinematic_conditioning.attention_layers == 4
    assert kinematic.rl.effective_observation_dim == 7_737
    assert kinematic.rl.ppo.max_iterations == 5_000


def test_paper_head_kinematic_experiment_reuses_tip_contact_data():
    original = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d4_full_paper_head_dgn_5k.py"
    )
    kinematic = load_exp_cfg(
        "configs/experiments/"
        "ce_prl_unicorn_d4_full_paper_head_kinematic_dgn_5k.py"
    )
    original_refs = {
        ref.stage: ref.config_hash for ref in resolve_artifacts(original).stages
    }
    kinematic_refs = {
        ref.stage: ref.config_hash for ref in resolve_artifacts(kinematic).stages
    }

    assert original_refs["contact_gen"] == kinematic_refs["contact_gen"]
    assert original_refs["pretrain"] != kinematic_refs["pretrain"]
    assert original.contact_gen.require_tool_tip_anchor is True
    assert kinematic.contact_gen.require_tool_tip_anchor is True
    assert kinematic.contact_gen.name == "contact_gen_prl_paper_head_500k"
    assert kinematic.model.tce.vit_depth == 4
    assert kinematic.model.tce.kinematic_conditioning.enabled is True
    assert kinematic.model.tce.kinematic_conditioning.attention_layers == 4
    assert kinematic.rl.effective_observation_dim == 7_737
    assert kinematic.rl.ppo.max_iterations == 5_000


def test_all_d4_ce_prl_contact_quality_experiments_use_5k_dgn():
    paths = sorted(
        Path("configs/experiments").glob(
            "ce_prl_unicorn_d4_full_*_dgn_5k.py"
        )
    )
    assert len(paths) == 9
    for path in paths:
        cfg = load_exp_cfg(path)
        cfg.validate()
        assert cfg.model.tce.vit_depth == 4
        assert cfg.rl.ppo.max_iterations == 5_000

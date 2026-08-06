from __future__ import annotations

import json
from pathlib import Path

from configs.experiments.ce_unicorn_ours_dgn_10k import (
    EXP_CFG,
    UNICORN_OURS_PRETRAIN_CHECKPOINT,
)
from configs.experiments.ce_unicorn_ours_gg_15k import (
    EXP_CFG as CE_GG_EXP_CFG,
)
from configs.experiments.ce_unicorn_ours_nonpenetrating_contact_gg_15k import (
    EXP_CFG as CE_NONPENETRATING_GG_EXP_CFG,
)
from configs.experiments.ce_unicorn_ours_nonpenetrating_contact_dgn_10k import (
    EXP_CFG as CE_NONPENETRATING_RL_EXP_CFG,
    NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT,
)
from configs.experiments.ce_unicorn_ours_raw_contact_dgn_10k import (
    EXP_CFG as CE_RAW_RL_EXP_CFG,
    RAW_CONTACT_PRETRAIN_CHECKPOINT,
)
from configs.experiments.ce_unicorn_ours_raw_contact_gg_15k import (
    EXP_CFG as CE_RAW_GG_EXP_CFG,
)
from configs.experiments.ce_rev_unicorn_ours_dgn_10k import (
    EXP_CFG as CE_REV_DGN_EXP_CFG,
)
from configs.experiments.ce_rev_unicorn_ours_gg_15k import (
    EXP_CFG as CE_REV_GG_EXP_CFG,
)
from configs.experiments.ce_rev_unicorn_ours_raw_contact_dgn_10k import (
    EXP_CFG as CE_REV_RAW_DGN_EXP_CFG,
)
from configs.experiments.ce_rev_unicorn_ours_raw_contact_gg_15k import (
    EXP_CFG as CE_REV_RAW_GG_EXP_CFG,
)
from configs.experiments.ce_prl_diff_dgn_10k import (
    EXP_CFG as CE_PRL_DIFF_EXP_CFG,
)
from configs.experiments.ce_prl_diff_gg_15k import (
    EXP_CFG as CE_PRL_DIFF_GG_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_raw_dgn_10k import (
    EXP_CFG as CE_PRL_RAW_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_raw_cross_dgn_10k import (
    EXP_CFG as CE_PRL_RAW_CROSS_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_raw_gg_15k import (
    EXP_CFG as CE_PRL_RAW_GG_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_nonpenetrating_dgn_10k import (
    EXP_CFG as CE_PRL_NONPENETRATING_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_nonpenetrating_gg_15k import (
    EXP_CFG as CE_PRL_NONPENETRATING_GG_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_stable_dgn_10k import (
    EXP_CFG as CE_PRL_STABLE_EXP_CFG,
)
from configs.experiments.ce_prl_unicorn_ours_stable_gg_15k import (
    EXP_CFG as CE_PRL_STABLE_GG_EXP_CFG,
)
from configs.experiments.panda_general_unicorn_ours_intersecting_depth1_full_yes_5k import (
    EXP_CFG as INTERSECTING_DEPTH1_FULL_YES_EXP_CFG,
)
from configs.experiments.panda_general_unicorn_ours_intersecting_depth1_gg_from_full_yes_5k import (
    EXP_CFG as INTERSECTING_DEPTH1_GG_EXP_CFG,
)
from configs.experiments.panda_general_unicorn_ours_full_yes_5k import (
    EXP_CFG as ORIGINAL_UNICORN_OURS_EXP_CFG,
)
from configs.panda_experiment_common import INTERSECTING_GEOMETRY_CONTACT_DATASET
from configs.panda_experiment_common import (
    ce_unicorn_ours_nonpenetrating_contact_pretrain_cfg,
    ce_unicorn_ours_raw_contact_pretrain_cfg,
)
from utils.config.paths import load_project_paths


ROOT = Path(__file__).resolve().parents[1]


def test_ce_dgn_10k_contract_uses_200_parallel_and_revolute_assets():
    EXP_CFG.validate()
    paths = load_project_paths(EXP_CFG.paths_yaml)
    generated = json.loads(
        paths.get("generated_grippers.manifest").read_text(encoding="utf-8")
    )

    assert EXP_CFG.name == "ce_unicorn_ours_dgn_10k"
    assert EXP_CFG.num_gpus == 8
    assert EXP_CFG.rl.launch.distributed is True
    assert EXP_CFG.rl.ppo.max_iterations == 10000
    assert EXP_CFG.rl.env.robot_mode == "cross_embodiment_gripper"
    assert EXP_CFG.rl.isaac_task_id == "cross-embodiment-gripper-v0"
    assert EXP_CFG.rl.observation.tool_cloud_source == "gripper_cloud_cache_v1"
    assert EXP_CFG.model.pretrained_encoder.checkpoint_path == UNICORN_OURS_PRETRAIN_CHECKPOINT
    assert len(generated["grippers"]) == 200
    assert paths.get("generated_grippers.root") == Path(
        "/mnt/project/world_model/tool_generalist/gripper_new"
    )
    assert paths.get("one_dof_grippers.manifest") == (
        ROOT / "gripper/generated_graspgenx/two_finger_revolute.json"
    )
    revolute = json.loads(
        paths.get("one_dof_grippers.manifest").read_text(encoding="utf-8")
    )
    assert len(revolute["grippers"]) == 100
    assert {entry["category"] for entry in revolute["grippers"]} == {
        "two_finger_revolute"
    }


def test_every_ce_experiment_is_explicitly_eight_gpu():
    from utils.config.loader import load_exp_cfg

    ce_configs = sorted((ROOT / "configs/experiments").glob("ce_*.py"))
    assert ce_configs
    for config_path in ce_configs:
        cfg = load_exp_cfg(config_path)
        assert cfg.num_gpus == 8, config_path.name
        assert cfg.rl.launch.distributed is True, config_path.name


def test_ce_recording_defaults_to_two_family_ranks_and_skips_optimizer_load():
    record_wrapper = (ROOT / "record.bash").read_text(encoding="utf-8")
    record_script = (
        ROOT / "scripts/record_failure_videos.py"
    ).read_text(encoding="utf-8")

    assert 'NUM_GPUS="${RECORD_NUM_GPUS:-2}"' in record_wrapper
    assert "ppo_runner.load(resume_path, load_optimizer=False)" in record_script


def test_cross_embodiment_runtime_keeps_action_and_state_semantics_aligned():
    action_source = (
        ROOT
        / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/actions/symmetric_generated_gripper_action.py"
    ).read_text(encoding="utf-8")
    observation_source = (
        ROOT
        / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/observations.py"
    ).read_text(encoding="utf-8")

    assert "semantic_closure: bool = False" in action_source
    assert "_generated_gripper_commanded_closure" in action_source
    assert "_cross_embodiment_generated_robot_state" in observation_source
    assert 'requested_robot_mode", "") == "cross_embodiment_gripper"' in observation_source


def test_ce_contact_pretrain_catalog_contains_both_generated_families():
    paths = load_project_paths("configs/paths/ce_contact_pretrain.yaml")
    selected = json.loads(
        paths.get("tools.tools_selected_json").read_text(encoding="utf-8")
    )
    adjusted = json.loads(
        paths.get("tools.tools_adjusted_json").read_text(encoding="utf-8")
    )

    assert len(selected) == 300
    assert len(set(selected)) == 300
    assert len(adjusted) == 300
    assert sum("source_generated_gripper_id" in entry for entry in adjusted) == 200
    assert sum("source_one_dof_gripper_id" in entry for entry in adjusted) == 100


def test_ce_contact_ablation_is_matched_except_for_rejection_search():
    raw = ce_unicorn_ours_raw_contact_pretrain_cfg()
    nonpenetrating = ce_unicorn_ours_nonpenetrating_contact_pretrain_cfg()
    raw.validate()
    nonpenetrating.validate()

    assert raw.paths_yaml == nonpenetrating.paths_yaml
    assert raw.num_gpus == nonpenetrating.num_gpus == 2
    assert raw.general.seed == nonpenetrating.general.seed
    assert raw.general.contact_objects_manifest == nonpenetrating.general.contact_objects_manifest
    assert raw.model.tce == nonpenetrating.model.tce
    assert raw.pretrain.epochs == nonpenetrating.pretrain.epochs
    assert raw.pretrain.batch_size == nonpenetrating.pretrain.batch_size
    assert raw.pretrain.optimizer == nonpenetrating.pretrain.optimizer
    assert raw.pretrain.max_files == nonpenetrating.pretrain.max_files == 2048
    assert (
        raw.pretrain.max_contacts_per_file
        == nonpenetrating.pretrain.max_contacts_per_file
        == 524
    )
    assert raw.contact_gen.geometry_only is True
    assert nonpenetrating.contact_gen.geometry_only is True
    assert raw.contact_gen.require_tool_tip_anchor is True
    assert nonpenetrating.contact_gen.require_tool_tip_anchor is True
    assert raw.contact_gen.B == nonpenetrating.contact_gen.B == 524
    assert raw.contact_gen.contact_geometry_mode == "intersecting_anchor_pairs"
    assert nonpenetrating.contact_gen.contact_geometry_mode == "anchor_pair_rejection"
    assert raw.contact_gen.M == 1
    assert nonpenetrating.contact_gen.M == 256
    assert raw.contact_gen.rejection_refill is False
    assert nonpenetrating.contact_gen.rejection_refill is True
    assert nonpenetrating.contact_gen.rejection_max_rounds == 64


def test_ce_contact_ablation_rl_uses_matched_cross_embodiment_setting():
    for cfg in (CE_RAW_RL_EXP_CFG, CE_NONPENETRATING_RL_EXP_CFG):
        cfg.validate()
        assert cfg.rl.env.robot_mode == "cross_embodiment_gripper"
        assert cfg.num_gpus == 8
        assert cfg.rl.isaac_task_id == "cross-embodiment-gripper-v0"
        assert cfg.rl.observation.tool_cloud_source == "gripper_cloud_cache_v1"
        assert cfg.rl.ppo.max_iterations == 10000
        assert cfg.rl.launch.wandb_project == "dgn_set"
        assert cfg.contact_gen.enabled is False
        assert cfg.pretrain.enabled is False
        assert cfg.pretrain_reuse is None

    assert CE_RAW_RL_EXP_CFG.paths_yaml == "configs/paths/ce_contact_pretrain.yaml"
    assert CE_NONPENETRATING_RL_EXP_CFG.paths_yaml == (
        "configs/paths/ce_contact_pretrain.yaml"
    )
    assert (
        CE_RAW_RL_EXP_CFG.model.pretrained_encoder.checkpoint_path
        == RAW_CONTACT_PRETRAIN_CHECKPOINT
    )
    assert (
        CE_NONPENETRATING_RL_EXP_CFG.model.pretrained_encoder.checkpoint_path
        == NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT
    )


def test_ce_gg_15k_transfers_matching_completed_dgn_10k_policy():
    pairs = (
        (CE_GG_EXP_CFG, EXP_CFG),
        (CE_RAW_GG_EXP_CFG, CE_RAW_RL_EXP_CFG),
        (CE_NONPENETRATING_GG_EXP_CFG, CE_NONPENETRATING_RL_EXP_CFG),
    )
    for gg_cfg, parent_cfg in pairs:
        gg_cfg.validate()
        assert gg_cfg.num_gpus == 8
        assert gg_cfg.rl.ppo.max_iterations == 15000
        assert gg_cfg.rl.launch.wandb_project == "ungraspable_set"
        assert gg_cfg.general.rl_objects_manifest.endswith(
            "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_"
            "conf_gt_0p9_listed_scales.json"
        )
        assert gg_cfg.rl.resume_checkpoint is None
        assert Path(gg_cfg.rl.init_checkpoint).is_file()
        assert Path(gg_cfg.rl.init_checkpoint).name == "model_best.pt"
        assert parent_cfg.name in gg_cfg.rl.init_checkpoint
        assert (
            gg_cfg.model.pretrained_encoder.checkpoint_path
            == parent_cfg.model.pretrained_encoder.checkpoint_path
        )
        assert gg_cfg.paths_yaml == parent_cfg.paths_yaml


def test_ce_rev_experiments_use_only_generated_revolute_grippers():
    revolute_paths = load_project_paths(
        "configs/paths/generated_two_finger_revolute.yaml"
    )
    revolute_manifest = json.loads(
        revolute_paths.get("one_dof_grippers.manifest").read_text(encoding="utf-8")
    )
    assert len(revolute_manifest["grippers"]) == 100
    assert {entry["category"] for entry in revolute_manifest["grippers"]} == {
        "two_finger_revolute"
    }

    dgn_pairs = (
        (CE_REV_DGN_EXP_CFG, UNICORN_OURS_PRETRAIN_CHECKPOINT),
        (CE_REV_RAW_DGN_EXP_CFG, RAW_CONTACT_PRETRAIN_CHECKPOINT),
    )
    for cfg, encoder_checkpoint in dgn_pairs:
        cfg.validate()
        assert cfg.num_gpus == 8
        assert cfg.paths_yaml == "configs/paths/generated_two_finger_revolute.yaml"
        assert cfg.rl.env.robot_mode == "one_dof_gripper"
        assert cfg.rl.isaac_task_id == "one-dof-gripper-v0"
        assert cfg.rl.observation.tool_cloud_source == "gripper_cloud_cache_v1"
        assert cfg.rl.ppo.max_iterations == 10000
        assert cfg.rl.launch.wandb_project == "dgn_set"
        assert cfg.model.pretrained_encoder.checkpoint_path == encoder_checkpoint

    gg_pairs = (
        (CE_REV_GG_EXP_CFG, CE_REV_DGN_EXP_CFG),
        (CE_REV_RAW_GG_EXP_CFG, CE_REV_RAW_DGN_EXP_CFG),
    )
    for cfg, parent_cfg in gg_pairs:
        cfg.validate()
        assert cfg.num_gpus == 8
        assert cfg.paths_yaml == parent_cfg.paths_yaml
        assert cfg.rl.env.robot_mode == "one_dof_gripper"
        assert cfg.rl.ppo.max_iterations == 15000
        assert cfg.rl.launch.wandb_project == "ungraspable_set"
        assert cfg.rl.resume_checkpoint is None
        assert parent_cfg.name in cfg.rl.init_checkpoint


def test_ce_prl_depth1_experiments_share_parallel_full_attention_structure():
    configs = (
        CE_PRL_RAW_EXP_CFG,
        CE_PRL_NONPENETRATING_EXP_CFG,
        CE_PRL_STABLE_EXP_CFG,
        CE_PRL_DIFF_EXP_CFG,
    )
    for cfg in configs:
        cfg.validate()
        paths = load_project_paths(cfg.paths_yaml)
        parallel = json.loads(
            paths.get("generated_grippers.manifest").read_text(encoding="utf-8")
        )

        assert cfg.num_gpus == 8
        assert cfg.paths_yaml == "configs/paths/generated_gripper_contact.yaml"
        assert len(parallel["grippers"]) == 400
        assert cfg.rl.env.robot_mode == "generated_gripper"
        assert cfg.rl.isaac_task_id == "generated-gripper-v0"
        assert cfg.rl.observation.tool_cloud_source == (
            "gripper_cloud_cache_v1"
        )
        assert cfg.model.encoder_backend == "tce"
        assert cfg.model.tce.vit_depth == 1
        assert cfg.model.tce.vit_attention_mode == "joint_self"
        assert cfg.model.pretrained_encoder.checkpoint_path is None
        assert cfg.contact_gen.enabled is False
        assert cfg.contact_gen.regenerate is False
        assert cfg.pretrain.enabled is True
        assert cfg.pretrain.retrain is True
        assert cfg.pretrain_reuse is None
        assert cfg.rl.ppo.max_iterations == 10000
        assert cfg.rl.launch.wandb_project == "dgn_set"

    assert CE_PRL_RAW_EXP_CFG.pretrain.use_geometry_candidates is True
    assert CE_PRL_RAW_EXP_CFG.pretrain.max_files == 2048
    assert CE_PRL_RAW_EXP_CFG.pretrain.max_contacts_per_file == 512
    assert CE_PRL_RAW_EXP_CFG.pretrain.tasks.contact is True
    assert CE_PRL_RAW_EXP_CFG.pretrain.tasks.diffusion is False
    assert CE_PRL_RAW_EXP_CFG.pretrain.tasks.postcontact is False

    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.use_geometry_candidates is True
    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.max_files == 2048
    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.max_contacts_per_file == 524
    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.tasks.contact is True
    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.tasks.diffusion is False
    assert CE_PRL_NONPENETRATING_EXP_CFG.pretrain.tasks.postcontact is False
    assert (
        CE_PRL_NONPENETRATING_EXP_CFG.pretrain.dataset_manifest
        == CE_PRL_STABLE_EXP_CFG.pretrain.dataset_manifest
    )

    assert CE_PRL_STABLE_EXP_CFG.pretrain.use_geometry_candidates is False
    assert CE_PRL_STABLE_EXP_CFG.pretrain.max_files == 0
    assert CE_PRL_STABLE_EXP_CFG.pretrain.max_contacts_per_file == 0
    assert CE_PRL_STABLE_EXP_CFG.pretrain.tasks.contact is True
    assert CE_PRL_STABLE_EXP_CFG.pretrain.tasks.diffusion is False
    assert CE_PRL_STABLE_EXP_CFG.pretrain.tasks.postcontact is False

    assert CE_PRL_DIFF_EXP_CFG.pretrain.use_geometry_candidates is False
    assert CE_PRL_DIFF_EXP_CFG.pretrain.max_files == 0
    assert CE_PRL_DIFF_EXP_CFG.pretrain.max_contacts_per_file == 0
    assert CE_PRL_DIFF_EXP_CFG.pretrain.tasks.contact is False
    assert CE_PRL_DIFF_EXP_CFG.pretrain.tasks.diffusion is True
    assert CE_PRL_DIFF_EXP_CFG.pretrain.tasks.postcontact is False
    assert CE_PRL_DIFF_EXP_CFG.pretrain.enabled_heads == ["diff"]
    assert (
        CE_PRL_STABLE_EXP_CFG.pretrain.dataset_manifest
        == CE_PRL_DIFF_EXP_CFG.pretrain.dataset_manifest
    )


def test_parallel_nonpenetrating_gg_transfers_matching_dgn_policy():
    cfg = CE_PRL_NONPENETRATING_GG_EXP_CFG
    parent = CE_PRL_NONPENETRATING_EXP_CFG

    cfg.validate()
    assert cfg.num_gpus == 8
    assert cfg.paths_yaml == parent.paths_yaml
    assert cfg.rl.env.robot_mode == "generated_gripper"
    assert cfg.rl.ppo.max_iterations == 15000
    assert cfg.rl.launch.wandb_project == "ungraspable_set"
    assert cfg.rl.resume_checkpoint is None
    assert parent.name in cfg.rl.init_checkpoint
    assert cfg.pretrain.enabled is True
    assert cfg.pretrain.retrain is False
    assert cfg.pretrain_reuse == (
        "ce_prl_unicorn_ours_nonpenetrating_dgn_10k.py"
    )
    assert cfg.model.pretrained_encoder.checkpoint_path is None
    assert cfg.model.tce.vit_depth == parent.model.tce.vit_depth == 1
    assert (
        cfg.model.tce.vit_attention_mode
        == parent.model.tce.vit_attention_mode
        == "joint_self"
    )


def test_parallel_raw_stable_and_diff_gg_transfer_matching_dgn_policies():
    pairs = (
        (CE_PRL_RAW_GG_EXP_CFG, CE_PRL_RAW_EXP_CFG),
        (CE_PRL_STABLE_GG_EXP_CFG, CE_PRL_STABLE_EXP_CFG),
        (CE_PRL_DIFF_GG_EXP_CFG, CE_PRL_DIFF_EXP_CFG),
    )
    for cfg, parent in pairs:
        cfg.validate()
        assert cfg.num_gpus == parent.num_gpus == 8
        assert cfg.paths_yaml == parent.paths_yaml
        assert cfg.rl.env.robot_mode == "generated_gripper"
        assert cfg.rl.ppo.max_iterations == 15000
        assert cfg.rl.launch.wandb_project == "ungraspable_set"
        assert cfg.rl.resume_checkpoint is None
        assert parent.name in cfg.rl.init_checkpoint
        assert cfg.pretrain.enabled is True
        assert cfg.pretrain.retrain is False
        assert cfg.pretrain_reuse == f"{parent.name}.py"
        assert cfg.model.pretrained_encoder.checkpoint_path is None
        assert cfg.model.tce.vit_depth == parent.model.tce.vit_depth == 1
        assert (
            cfg.model.tce.vit_attention_mode
            == parent.model.tce.vit_attention_mode
            == "joint_self"
        )


def test_parallel_raw_cross_uses_real_cross_only_pretrain_and_rl():
    cfg = CE_PRL_RAW_CROSS_EXP_CFG
    full = CE_PRL_RAW_EXP_CFG

    cfg.validate()
    assert cfg.name == "ce_prl_unicorn_ours_raw_cross_dgn_10k"
    assert cfg.num_gpus == 8
    assert cfg.paths_yaml == full.paths_yaml
    assert cfg.model.tce.vit_depth == full.model.tce.vit_depth == 1
    assert cfg.model.tce.vit_attention_mode == "cross_only"
    assert full.model.tce.vit_attention_mode == "joint_self"
    assert cfg.model.pretrained_encoder.checkpoint_path is None
    assert cfg.pretrain.enabled is True
    assert cfg.pretrain.retrain is True
    assert cfg.pretrain_reuse is None
    assert cfg.contact_gen.enabled is False
    assert cfg.pretrain.dataset_manifest == full.pretrain.dataset_manifest
    assert cfg.pretrain.use_geometry_candidates is True
    assert cfg.pretrain.max_files == full.pretrain.max_files == 2048
    assert (
        cfg.pretrain.max_contacts_per_file
        == full.pretrain.max_contacts_per_file
        == 512
    )
    assert cfg.rl.ppo.max_iterations == 10000
    assert cfg.rl.launch.wandb_project == "dgn_set"


def test_intersecting_depth1_experiments_use_expected_encoder_and_grippers():
    for cfg in (
        INTERSECTING_DEPTH1_FULL_YES_EXP_CFG,
        INTERSECTING_DEPTH1_GG_EXP_CFG,
    ):
        cfg.validate()
        paths = load_project_paths(cfg.paths_yaml)
        generated = json.loads(
            paths.get("generated_grippers.manifest").read_text(encoding="utf-8")
        )

        assert len(generated["grippers"]) == 200
        assert cfg.rl.env.robot_mode == "generated_gripper"
        assert cfg.rl.isaac_task_id == "generated-gripper-v0"
        assert cfg.rl.observation.tool_cloud_source == (
            "gripper_cloud_cache_v1"
        )
        assert cfg.model.pretrained_encoder.checkpoint_path is None
        assert cfg.model.tce.vit_depth == 1
        assert cfg.model.tce.vit_attention_mode == "cross_only"
        assert paths.get("one_dof_grippers.manifest") is None

    assert INTERSECTING_DEPTH1_FULL_YES_EXP_CFG.contact_gen.enabled is False
    assert INTERSECTING_DEPTH1_FULL_YES_EXP_CFG.pretrain.enabled is True
    assert INTERSECTING_DEPTH1_FULL_YES_EXP_CFG.pretrain_reuse is None
    assert INTERSECTING_DEPTH1_FULL_YES_EXP_CFG.pretrain.dataset_manifest == (
        INTERSECTING_GEOMETRY_CONTACT_DATASET
    )
    assert INTERSECTING_DEPTH1_GG_EXP_CFG.pretrain_reuse == (
        "panda_general_unicorn_ours_intersecting_depth1_full_yes_5k.py"
    )


def test_unicorn_ours_baseline_and_comparison_wandb_projects():
    assert ORIGINAL_UNICORN_OURS_EXP_CFG.model.tce.vit_depth == 12
    assert ORIGINAL_UNICORN_OURS_EXP_CFG.model.tce.vit_attention_mode == "joint_self"
    assert INTERSECTING_DEPTH1_FULL_YES_EXP_CFG.rl.launch.wandb_project == "dgn_set"
    assert INTERSECTING_DEPTH1_GG_EXP_CFG.rl.launch.wandb_project == (
        "ungraspable_set"
    )


def test_real_speed_gg_finetune_initializes_prior_policy_with_fresh_optimizer():
    from configs.experiments.panda_general_unicorn_ours_gg_real_speed import (
        ENCODER_CHECKPOINT,
        EXP_CFG as REAL_SPEED_EXP_CFG,
        PARENT_POLICY_CHECKPOINT,
    )

    REAL_SPEED_EXP_CFG.validate()
    assert REAL_SPEED_EXP_CFG.name == "panda_general_unicorn_ours_gg_real_speed"
    assert REAL_SPEED_EXP_CFG.rl.env.robot_mode == "generated_gripper"
    assert REAL_SPEED_EXP_CFG.rl.init_checkpoint == PARENT_POLICY_CHECKPOINT
    assert REAL_SPEED_EXP_CFG.rl.resume_checkpoint is None
    assert (
        REAL_SPEED_EXP_CFG.model.pretrained_encoder.checkpoint_path
        == ENCODER_CHECKPOINT
    )
    assert REAL_SPEED_EXP_CFG.rl.ppo.max_iterations == 15000

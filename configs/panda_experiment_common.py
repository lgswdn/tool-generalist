"""Shared builders for Panda/generated-gripper experiments.

This is intentionally not an experiment module: it defines no EXP_CFG.  Actual
experiment files call these builders instead of importing another full RL
experiment.
"""

from __future__ import annotations

from pathlib import Path

from configs.config_contact_gen import (
    CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
    CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
    CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
    PENETRATION_CHECK_BIDIRECTIONAL,
    ROTATION_SELECTION_MOST_CAVITY_CENTERED,
    ROTATION_SELECTION_RANDOM_LEGAL,
    TOOL_SOURCE_SELECTED_TOOLS,
)
from configs.config_exp import ExpCfg
from configs.config_pretrain import DIFF_CFG, clone_cfg


FULL_YES_MANIFEST = "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"
GENERATED_GRIPPER_PATHS_YAML = "configs/paths/generated_gripper_contact.yaml"
GENERATED_GRIPPER_NEW_PATHS_YAML = (
    "configs/paths/generated_gripper_contact_new.yaml"
)
ROBOTIQ_2F140_PATHS_YAML = "configs/paths/robotiq_2f140.yaml"
ONROBOT_RG2_PATHS_YAML = "configs/paths/onrobot_rg2.yaml"
ROBOTIQ_3F_PATHS_YAML = "configs/paths/robotiq_3f.yaml"
GENERATED_REVOLUTE_PATHS_YAML = "configs/paths/generated_two_finger_revolute.yaml"
GENERATED_THREE_FINGER_PATHS_YAML = "configs/paths/generated_three_finger_high_dof.yaml"
CROSS_EMBODIMENT_GRIPPER_PATHS_YAML = (
    "configs/paths/cross_embodiment_generated_revolute.yaml"
)
CE_CONTACT_PRETRAIN_PATHS_YAML = "configs/paths/ce_contact_pretrain.yaml"
CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML = (
    "configs/paths/ce_general_contact_pretrain.yaml"
)
UNICORN_COMPARISON_CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_generated_gripper/"
    "fdc5885d5d2a55727c19a6d984557275d2a7f5e48e70f6ef32e01a5bbc03daa3"
)
PARALLEL_NEW_POSTCONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_gripper_new/"
    "2f47448cf8e73179fddc2ff6c6bee40fc70691ba69eb4d0e15bafc53a82f366c"
)
PARALLEL_PAPER_1M_CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_prl_paper_contact_1m/"
    "c8f09bf58fa867999226d41b08b5c84879803edb4f5a603cfe123e2f82556f41"
)
PARALLEL_NONPENETRATING_1M_CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_prl_nonpenetrating_contact_1m/"
    "7927c6dce4068c890d29bc195a814d6211c35411e8476a02d8dcb666070839c7"
)
PROVEN_PARALLEL_NONPENETRATING_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "ce_prl_unicorn_ours_nonpenetrating_dgn_10k/"
    "contact_gen_generated_gripper/"
    "ce_prl_unicorn_ours_nonpenetrating_dgn_10k_"
    "ce_prl_unicorn_ours_nonpenetrating_dgn_10k/"
    "5e5e699c0b4209f7b137525644a5f145a8b77b343e28ebcf57722a8218ee6a48/"
    "best.pt"
)
INTERSECTING_GEOMETRY_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_intersecting_geometry/contact_gen_intersecting_geometry/"
    "unicorn_ours_intersecting_geometry_unicorn_ours_intersecting_geometry/"
    "f6117a7cd0bf6725e3eb43d5636c9731cb5d481682c4854c326aadbb090c85b2/best.pt"
)
INTERSECTING_GEOMETRY_CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_intersecting_geometry/"
    "7a6514929e5af1ce62563c1761fea2a3f9c96476fad1b790ce7de8e16cf21a92"
)
PARALLEL_RAW_CONTACT_MAX_FILES = 2048
PARALLEL_RAW_CONTACTS_PER_FILE = 512
OFFICIAL_UNICORN_REPRESENTATION_CHECKPOINT = str(
    Path(__file__).resolve().parents[1]
    / ".pretrained_checkpoints"
    / "hamnet"
    / "sam-multires-scaledr-zaug-3way-000000"
)
GG_BEST_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/"
    "no-contact/oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/"
    "20260719T202622Z/model_best.pt"
)
FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)
ORIGINAL_ORACLE_POINTCLOUD_POINTNET_DGN5K_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "no-contact/oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "20260719T092442Z/model_best.pt"
)


def generated_gripper_diff_post_pretrain_cfg() -> ExpCfg:
    cfg = ExpCfg(name="generated_gripper_diff_post_pretrain")
    cfg.general.name = "generated_gripper_diff_post_pretrain"
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.num_gpus = 8
    cfg.general.tool_mount.scale_xyz = [1.0, 1.0, 1.0]
    cfg.general.contact_objects_manifest = FULL_YES_MANIFEST
    cfg.general.rl_objects_manifest = FULL_YES_MANIFEST

    cfg.contact_gen.name = "contact_gen_generated_gripper"
    cfg.contact_gen.enabled = True
    cfg.contact_gen.regenerate = False
    cfg.contact_gen.rotation_selection = ROTATION_SELECTION_RANDOM_LEGAL
    cfg.contact_gen.tool_source = TOOL_SOURCE_SELECTED_TOOLS
    cfg.contact_gen.require_tool_tip_anchor = True
    cfg.contact_gen.object_tool_manifest = None
    cfg.contact_gen.allow_self_object_tool_pairs = False
    cfg.contact_gen.num_pairs = 20000
    cfg.contact_gen.num_object_poses = 1
    cfg.contact_gen.object_scale_range = (0.1, 0.3)
    cfg.contact_gen.B = 4096
    cfg.contact_gen.M = 4096
    cfg.contact_gen.chunk_B = 256
    cfg.contact_gen.physics.t_stabilize = 30
    cfg.contact_gen.shard_count = 1
    cfg.contact_gen.shard_index = 0

    cfg.model.name = "generated_gripper_diff_post"

    cfg.pretrain = clone_cfg(DIFF_CFG)
    cfg.pretrain.name = "diff_post_generated_gripper"
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = True
    cfg.pretrain.enabled_heads = ["diff", "postcontact"]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = True
    cfg.pretrain.tasks.postcontact = True
    cfg.pretrain.tasks.contact = False
    cfg.pretrain.encoder_input_centering = "object_center"
    cfg.pretrain.condition_normalization = True
    cfg.pretrain.condition_norm_sample_files = 64
    cfg.pretrain.logger = "wandb"
    cfg.pretrain.wandb_project = "generated_gripper_pretrain"
    cfg.pretrain.wandb_run_name = "diff_post_generated_gripper"
    cfg.pretrain.optimizer.min_learning_rate = 3e-5

    cfg.rl.enabled = False
    cfg.rl.launch.distributed = False
    cfg.rl.observation.model_input_centering = "object_center"
    return cfg


def generated_gripper_diff_post_rl_cfg(name: str) -> ExpCfg:
    cfg = generated_gripper_diff_post_pretrain_cfg()
    cfg.name = name
    cfg.general.name = name
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.rl.enabled = True
    cfg.rl.name = name
    cfg.rl.isaac_task_id = "generated-gripper-v0"
    cfg.rl.launch.run_name = name
    cfg.rl.launch.distributed = True
    cfg.rl.launch.logger = "wandb"
    cfg.rl.launch.wandb_project = "panda_generated_gripper"
    cfg.rl.env.robot_mode = "generated_gripper"
    cfg.rl.env.num_envs = 1024
    cfg.rl.action.action_dim = 8
    cfg.rl.action.joint_names = ["panda_joint.*"]
    cfg.rl.observation.previous_action_dim = 8
    cfg.rl.observation.robot_state_dim = 18
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.table.enabled = True
    cfg.rl.table.pose_xyz = [0.5, 0.0, -0.02]
    cfg.rl.domain_randomization.ground.material.enabled = False
    cfg.rl.ppo.entropy_coef = 0.006
    cfg.rl.separate_actor_critic_fusion = True
    cfg.rl.reward.object_goal_tracking_term_weight = 3
    cfg.rl.reward.object_goal_tracking_fine_term_weight = 6
    return cfg


def generated_gripper_post_pretrain_cfg() -> ExpCfg:
    """Generated-gripper postcontact-only pretrain with object centering."""

    cfg = generated_gripper_diff_post_pretrain_cfg()
    cfg.name = "generated_gripper_post_pretrain"
    cfg.general.name = cfg.name
    cfg.model.name = "generated_gripper_post"
    cfg.pretrain.name = "post_generated_gripper"
    cfg.pretrain.enabled_heads = ["postcontact"]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = False
    cfg.pretrain.tasks.postcontact = True
    cfg.pretrain.tasks.contact = False
    cfg.pretrain.num_precontact_steps = 0
    cfg.pretrain.encoder_input_centering = "object_center"
    cfg.pretrain.wandb_run_name = "post_generated_gripper"
    cfg.rl.observation.model_input_centering = "object_center"
    return cfg


def generated_gripper_post_rl_cfg(name: str) -> ExpCfg:
    """Generated-gripper RL backed by the postcontact-only encoder."""

    post_cfg = generated_gripper_post_pretrain_cfg()
    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.contact_gen = clone_cfg(post_cfg.contact_gen)
    cfg.pretrain = clone_cfg(post_cfg.pretrain)
    cfg.pretrain.retrain = False
    cfg.model.name = post_cfg.model.name
    cfg.pretrain_reuse = None
    cfg.rl.observation.model_input_centering = "object_center"
    return cfg


def official_panda_diff_post_rl_cfg(name: str) -> ExpCfg:
    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.rl.isaac_task_id = "panda-gripper-v0"
    cfg.rl.env.robot_mode = "official_panda_gripper"
    cfg.rl.observation.tool_cloud_source = "official_panda_gripper_kinematic_mesh"
    cfg.rl.launch.wandb_project = "panda_gripper"
    return cfg


def one_dof_gripper_diff_post_rl_cfg(name: str, *, paths_yaml: str) -> ExpCfg:
    """Official one-command gripper using the generated-gripper representation encoder."""
    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.paths_yaml = paths_yaml
    cfg.rl.isaac_task_id = "one-dof-gripper-v0"
    cfg.rl.env.robot_mode = "one_dof_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "panda_one_dof_gripper"
    return cfg


def robotiq_2f140_diff_post_rl_cfg(name: str) -> ExpCfg:
    return one_dof_gripper_diff_post_rl_cfg(name, paths_yaml=ROBOTIQ_2F140_PATHS_YAML)


def onrobot_rg2_diff_post_rl_cfg(name: str) -> ExpCfg:
    return one_dof_gripper_diff_post_rl_cfg(name, paths_yaml=ONROBOT_RG2_PATHS_YAML)


def robotiq_3f_diff_post_rl_cfg(name: str) -> ExpCfg:
    return one_dof_gripper_diff_post_rl_cfg(name, paths_yaml=ROBOTIQ_3F_PATHS_YAML)


def generated_revolute_diff_post_rl_cfg(name: str) -> ExpCfg:
    return one_dof_gripper_diff_post_rl_cfg(
        name,
        paths_yaml=GENERATED_REVOLUTE_PATHS_YAML,
    )


def generated_three_finger_diff_post_rl_cfg(name: str) -> ExpCfg:
    return one_dof_gripper_diff_post_rl_cfg(
        name,
        paths_yaml=GENERATED_THREE_FINGER_PATHS_YAML,
    )


def unicorn_pretrain_generated_gripper_cfg(*, ours_tce: bool = False) -> ExpCfg:
    cfg = generated_gripper_diff_post_pretrain_cfg()

    # Keep data, objective, augmentation, optimizer, schedule, and model size
    # identical. The selected encoder backend is the only semantic difference.
    cfg.pretrain.enabled_heads = ["contact"]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = False
    cfg.pretrain.tasks.postcontact = False
    cfg.pretrain.tasks.contact = True
    cfg.pretrain.num_precontact_steps = 0
    cfg.pretrain.encoder_input_centering = "object_center"

    if ours_tce:
        cfg.name = "unicorn_pretrain_ours_generated_gripper"
        cfg.general.name = "unicorn_pretrain_ours_generated_gripper"
        cfg.model.name = "unicorn_contact_ours_generated_gripper"
        cfg.model.encoder_backend = "tce"
        cfg.model.pretrained_encoder.name = "tce"
        cfg.model.pretrained_encoder.adapter = "tce_strict"
        cfg.pretrain.name = "unicorn_contact_ours_generated_gripper"
        cfg.pretrain.mode = "tce_multitask"
        cfg.pretrain.wandb_run_name = "generated_gripper_ours_tce_contact"
        # Keep the original UniCORN-ours baseline at its full-attention,
        # twelve-block architecture. Specialized ablations override these.
        cfg.model.tce.vit_depth = 12
        cfg.model.tce.vit_attention_mode = "joint_self"
    else:
        cfg.name = "unicorn_pretrain_generated_gripper"
        cfg.general.name = "unicorn_pretrain_generated_gripper"
        cfg.model.name = "unicorn_contact_generated_gripper"
        cfg.model.encoder_backend = "unicorn"
        cfg.model.pretrained_encoder.name = "unicorn"
        cfg.model.pretrained_encoder.adapter = "unicorn_strict"
        cfg.model.unicorn.num_points = cfg.model.tce.num_points
        cfg.model.unicorn.num_patches = cfg.model.tce.num_points // cfg.model.tce.patch_size
        cfg.model.unicorn.patch_size = cfg.model.tce.patch_size
        cfg.model.unicorn.encoder_channel = cfg.model.tce.encoder_channel
        cfg.model.unicorn.vit_depth = cfg.model.tce.vit_depth
        cfg.model.unicorn.vit_heads = cfg.model.tce.vit_heads
        cfg.pretrain.name = "unicorn_contact_generated_gripper"
        cfg.pretrain.mode = "unicorn_contact"
        cfg.pretrain.wandb_run_name = "generated_gripper_unicorn_contact"

    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = False
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.dataset_manifest = UNICORN_COMPARISON_CONTACT_DATASET
    cfg.pretrain.logger = "wandb"
    cfg.pretrain.wandb_project = "generated_gripper_pretrain"
    cfg.rl.enabled = False
    cfg.rl.launch.distributed = False
    return cfg


def unicorn_ours_intersecting_geometry_pretrain_cfg() -> ExpCfg:
    """UniCORN contact pretrain on unconstrained, geometry-only contacts."""

    cfg = unicorn_pretrain_generated_gripper_cfg(ours_tce=True)
    cfg.name = "unicorn_pretrain_ours_intersecting_geometry"
    cfg.general.name = cfg.name
    cfg.model.name = "unicorn_ours_intersecting_geometry"

    cfg.contact_gen.name = "contact_gen_intersecting_geometry"
    cfg.contact_gen.enabled = True
    cfg.contact_gen.regenerate = False
    cfg.contact_gen.geometry_only = True
    cfg.contact_gen.contact_geometry_mode = CONTACT_GEOMETRY_INTERSECTING_ANCHORS
    cfg.contact_gen.require_tool_tip_anchor = True
    # The original UniCORN-ours checkpoint saw 965,043 training cases after
    # its 90/10 split. Pack 524 poses into each file so this split produces
    # 966,256 training cases (about 944 distributed batches versus 943).
    cfg.contact_gen.num_pairs = 2048
    cfg.contact_gen.num_object_poses = 1
    cfg.contact_gen.B = 524
    cfg.contact_gen.M = 1
    cfg.contact_gen.chunk_B = 64
    cfg.contact_gen.object_scale_range = (0.1, 0.3)
    cfg.contact_gen.epsilon = 0.002

    cfg.pretrain.name = "unicorn_ours_intersecting_geometry"
    cfg.pretrain.dataset_manifest = None
    cfg.pretrain.use_geometry_candidates = True
    cfg.pretrain.max_files = 2048
    cfg.pretrain.max_contacts_per_file = 524
    cfg.pretrain.retrain = False
    cfg.pretrain.unicorn.label.contact_eps = 0.002
    cfg.pretrain.wandb_run_name = "unicorn_ours_intersecting_geometry"
    return cfg


def unicorn_ours_intersecting_depth1_pretrain_cfg() -> ExpCfg:
    """Intersecting-geometry pretrain with a one-block cross-only ViT."""

    cfg = unicorn_ours_intersecting_geometry_pretrain_cfg()
    cfg.name = "unicorn_pretrain_ours_intersecting_depth1"
    cfg.general.name = cfg.name
    cfg.paths_yaml = "configs/paths/generated_gripper_contact_new.yaml"
    # Depth changes the model, not its training examples. Reuse the completed
    # dataset consumed by the original intersecting-geometry encoder.
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.model.name = "unicorn_ours_intersecting_depth1"
    cfg.model.tce.vit_depth = 1
    cfg.model.tce.vit_attention_mode = "cross_only"
    cfg.pretrain.name = "unicorn_ours_intersecting_depth1"
    cfg.pretrain.dataset_manifest = INTERSECTING_GEOMETRY_CONTACT_DATASET
    cfg.pretrain.max_contacts_per_file = 0
    cfg.pretrain.wandb_run_name = "unicorn_ours_intersecting_depth1"
    return cfg


def unicorn_ours_nonpenetrating_geometry_pretrain_cfg() -> ExpCfg:
    """Matched control using the existing rejection-sampled candidates."""

    cfg = unicorn_pretrain_generated_gripper_cfg(ours_tce=True)
    cfg.name = "unicorn_pretrain_ours_nonpenetrating_geometry"
    cfg.general.name = cfg.name
    cfg.model.name = "unicorn_ours_nonpenetrating_geometry"
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False

    cfg.pretrain.name = "unicorn_ours_nonpenetrating_geometry"
    cfg.pretrain.dataset_manifest = UNICORN_COMPARISON_CONTACT_DATASET
    cfg.pretrain.use_geometry_candidates = True
    cfg.pretrain.max_files = 2048
    # Candidate files vary in size and average roughly 950 cases in this
    # selection. Cap each at 524 to match the penetrating and original runs.
    cfg.pretrain.max_contacts_per_file = 524
    cfg.pretrain.retrain = False
    cfg.pretrain.unicorn.label.contact_eps = 0.002
    cfg.pretrain.wandb_run_name = "unicorn_ours_nonpenetrating_geometry"
    return cfg


def ce_unicorn_ours_contact_pretrain_cfg(*, allow_penetration: bool) -> ExpCfg:
    """Matched contact pretrain over parallel and revolute gripper proxies.

    Both variants use the same 300-tool catalog, sampled object/tool pairs,
    encoder, objective, and training budget.  The only data-construction
    distinction is whether anchor-aligned poses are accepted directly or must
    pass the SDF floor/penetration rejection filter.
    """

    variant = "raw_contact" if allow_penetration else "nonpenetrating_contact"
    cfg = unicorn_ours_intersecting_geometry_pretrain_cfg()
    cfg.name = f"unicorn_pretrain_ce_{variant}"
    cfg.general.name = cfg.name
    cfg.paths_yaml = CE_CONTACT_PRETRAIN_PATHS_YAML
    cfg.num_gpus = 2
    cfg.model.name = f"unicorn_ce_{variant}"

    cfg.contact_gen.name = f"contact_gen_ce_{variant}"
    cfg.contact_gen.contact_geometry_mode = (
        CONTACT_GEOMETRY_INTERSECTING_ANCHORS
        if allow_penetration
        else CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION
    )
    # Raw alignment needs one pose per anchor pair.  Rejection sampling tries
    # several orientations for that same pair so it can still return at most
    # the same B=524 examples per file after applying the non-penetration test.
    cfg.contact_gen.M = 1 if allow_penetration else 256
    cfg.contact_gen.penetration_eps = 5e-4
    cfg.contact_gen.geometry_only = True
    cfg.contact_gen.require_tool_tip_anchor = True
    cfg.contact_gen.rejection_refill = not allow_penetration
    cfg.contact_gen.rejection_max_rounds = 64 if not allow_penetration else 1

    cfg.pretrain.name = f"unicorn_ce_{variant}"
    cfg.pretrain.dataset_manifest = None
    cfg.pretrain.wandb_project = "cross_embodiment_gripper_pretrain"
    cfg.pretrain.wandb_run_name = f"unicorn_ce_{variant}"
    return cfg


def ce_unicorn_ours_raw_contact_pretrain_cfg() -> ExpCfg:
    """Raw anchor-aligned contact data; object/tool intersection is allowed."""

    return ce_unicorn_ours_contact_pretrain_cfg(allow_penetration=True)


def ce_unicorn_ours_nonpenetrating_contact_pretrain_cfg() -> ExpCfg:
    """Anchor-aligned contact data after floor and SDF penetration rejection."""

    return ce_unicorn_ours_contact_pretrain_cfg(allow_penetration=False)


def ce_unicorn_ours_contact_rl_cfg(
    name: str,
    *,
    allow_penetration: bool,
) -> ExpCfg:
    """Cross-embodiment RL initialized from one matched contact pretrain."""

    pretrain_cfg = ce_unicorn_ours_contact_pretrain_cfg(
        allow_penetration=allow_penetration
    )
    cfg = cross_embodiment_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.num_gpus = 8
    # Contains both the exact static contact meshes and live RL asset manifests, so
    # contact generation, pretrain, and RL can execute in one inline run.
    cfg.paths_yaml = CE_CONTACT_PRETRAIN_PATHS_YAML
    cfg.contact_gen = clone_cfg(pretrain_cfg.contact_gen)
    cfg.contact_gen.enabled = True
    cfg.contact_gen.regenerate = False
    cfg.pretrain = clone_cfg(pretrain_cfg.pretrain)
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.model = clone_cfg(pretrain_cfg.model)
    cfg.model.pretrained_encoder.checkpoint_path = None
    return cfg


def unicorn_ours_cross_only_depth1_pretrain_cfg() -> ExpCfg:
    """Original UniCORN-ours data/objective with a one-block cross-only ViT."""

    cfg = unicorn_pretrain_generated_gripper_cfg(ours_tce=True)
    cfg.name = "unicorn_pretrain_ours_cross_only_depth1"
    cfg.general.name = cfg.name
    cfg.model.name = "unicorn_ours_cross_only_depth1"
    cfg.model.tce.vit_depth = 1
    cfg.model.tce.vit_attention_mode = "cross_only"
    cfg.pretrain.name = "unicorn_ours_cross_only_depth1"
    cfg.pretrain.retrain = False
    cfg.pretrain.wandb_run_name = "unicorn_ours_cross_only_depth1"
    return cfg


def oracle_contact_pretrain_generated_gripper_cfg() -> ExpCfg:
    """Short full-dataset control using oracle tokens on the UniCORN objective."""

    cfg = unicorn_pretrain_generated_gripper_cfg(ours_tce=False)
    cfg.name = "oracle_patch_pretrain_generated_gripper"
    cfg.general.name = cfg.name
    cfg.model.name = "oracle_patch"
    cfg.model.encoder_backend = "oracle_patch"
    cfg.model.pretrained_encoder.name = "oracle_patch"
    cfg.model.pretrained_encoder.adapter = "oracle_none"
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.model.oracle_patch.num_points = cfg.model.unicorn.num_points
    cfg.model.oracle_patch.num_patches = cfg.model.unicorn.num_patches
    cfg.model.oracle_patch.patch_size = cfg.model.unicorn.patch_size
    cfg.model.oracle_patch.encoder_channel = cfg.model.unicorn.encoder_channel
    cfg.model.oracle_patch.include_contact_feature = False
    cfg.pretrain.name = "oracle_patch_contact_generated_gripper"
    cfg.pretrain.mode = "oracle_contact"
    cfg.pretrain.epochs = 3
    cfg.pretrain.retrain = True
    cfg.pretrain.wandb_run_name = "generated_gripper_oracle_patch_contact"
    return cfg


def official_panda_unicorn_rl_cfg(name: str, *, ours_tce: bool = False) -> ExpCfg:
    pretrain_cfg = unicorn_pretrain_generated_gripper_cfg(ours_tce=ours_tce)
    cfg = official_panda_diff_post_rl_cfg(name)
    cfg.contact_gen = clone_cfg(pretrain_cfg.contact_gen)
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain = clone_cfg(pretrain_cfg.pretrain)
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.model.name = pretrain_cfg.model.name
    cfg.model.encoder_backend = pretrain_cfg.model.encoder_backend
    cfg.model.pretrained_encoder = clone_cfg(pretrain_cfg.model.pretrained_encoder)
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.model.tce = clone_cfg(pretrain_cfg.model.tce)
    cfg.model.unicorn = clone_cfg(pretrain_cfg.model.unicorn)
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    cfg.rl.actor_critic_class = "ActorCriticTG"
    cfg.rl.observation.model_input_centering = "object_center"
    if not ours_tce:
        # The authors' released UniCORN representation is already pretrained.
        # Keep the controlled ``unicorn_ours`` pretraining path unchanged.
        cfg.pretrain.enabled = False
        cfg.pretrain.retrain = False
        cfg.pretrain_reuse = None
        cfg.model.unicorn.vit_depth = 4
        cfg.model.pretrained_encoder.checkpoint_path = (
            OFFICIAL_UNICORN_REPRESENTATION_CHECKPOINT
        )
    return cfg


def generated_gripper_unicorn_rl_cfg(name: str, *, ours_tce: bool = False) -> ExpCfg:
    cfg = official_panda_unicorn_rl_cfg(name, ours_tce=ours_tce)
    cfg.rl.isaac_task_id = "generated-gripper-v0"
    cfg.rl.env.robot_mode = "generated_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "panda_generated_gripper"
    return cfg


def parallel_depth1_full_attention_unicorn_rl_cfg(
    name: str,
    *,
    raw_contact: bool,
) -> ExpCfg:
    """Parallel-only UniCORN-ours pretrain/RL with a one-block joint ViT."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.paths_yaml = GENERATED_GRIPPER_PATHS_YAML
    cfg.num_gpus = 8
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = True
    cfg.pretrain_reuse = None
    cfg.model.name = name
    cfg.model.tce.vit_depth = 1
    cfg.model.tce.vit_attention_mode = "joint_self"
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.pretrain.name = name
    cfg.pretrain.dataset_manifest = (
        INTERSECTING_GEOMETRY_CONTACT_DATASET
        if raw_contact
        else UNICORN_COMPARISON_CONTACT_DATASET
    )
    cfg.pretrain.use_geometry_candidates = raw_contact
    # Raw files pack 512 anchor-aligned examples each. Stable files are much
    # smaller, so consume the complete original stable dataset rather than
    # matching by file count. Both selections contain roughly 1.05M examples.
    cfg.pretrain.max_files = PARALLEL_RAW_CONTACT_MAX_FILES if raw_contact else 0
    cfg.pretrain.max_contacts_per_file = (
        PARALLEL_RAW_CONTACTS_PER_FILE if raw_contact else 0
    )
    cfg.pretrain.wandb_project = "parallel_gripper_pretrain"
    cfg.pretrain.wandb_run_name = name
    return cfg


PAPER_CONTACT_VARIANTS = (
    "paper_contact",
    "paper_head",
    "raw_contact",
    "nonpenetrating_contact",
    "nonpenetrating_contact_perturbed",
    "nonpenetrating_contact_concavity_biased",
)


def parallel_paper_contact_quality_rl_cfg(
    name: str,
    *,
    contact_variant: str,
    transformer_depth: int = 1,
    point_jitter_std: float = 0.0,
    contact_eps: float = 0.0,
    dgn_iterations: int = 10_000,
    perturb_nonpenetrating: bool = True,
    nonpenetrating_penetration_eps: float = 0.002,
) -> ExpCfg:
    """Controlled full-attention UniCORN contact-quality experiment.

    Every variant uses the same 200 parallel grippers, balanced object/tool
    pair plan, exactly 500,000 training cases before the deterministic split,
    paper pretraining hyperparameters, encoder, and RL settings. Only the
    geometry procedure that constructs a near-contact pose changes.
    """

    if contact_variant not in PAPER_CONTACT_VARIANTS:
        raise ValueError(
            f"contact_variant must be one of {PAPER_CONTACT_VARIANTS}, "
            f"got {contact_variant!r}"
        )
    if int(transformer_depth) < 1:
        raise ValueError("transformer_depth must be >= 1")

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.num_gpus = 8
    cfg.general.contact_objects_manifest = FULL_YES_MANIFEST
    cfg.general.rl_objects_manifest = FULL_YES_MANIFEST

    cfg.model.name = (
        f"ce_prl_unicorn_d{int(transformer_depth)}_full_{contact_variant}"
    )
    cfg.model.tce.vit_depth = int(transformer_depth)
    cfg.model.tce.vit_attention_mode = "joint_self"
    cfg.model.pretrained_encoder.checkpoint_path = None

    contact = cfg.contact_gen
    contact.name = f"contact_gen_prl_{contact_variant}_500k"
    contact.enabled = True
    contact.regenerate = False
    contact.geometry_only = True
    contact.require_complete = True
    contact.precompute_convex_union_labels = False
    contact.precompute_mesh_sdf = True
    contact.balanced_tool_pairs = True
    contact.require_tool_tip_anchor = contact_variant != "paper_contact"
    contact.rotation_selection = ROTATION_SELECTION_RANDOM_LEGAL
    contact.num_pairs = 1000
    contact.num_object_poses = 1
    contact.B = 500
    contact.chunk_B = 64
    contact.num_surface_pts = 512
    contact.object_scale_range = (0.1, 0.3)
    contact.epsilon = 0.0
    # Disable the table/upright bias in the non-penetrating sampler. All
    # contact-quality variants use unrestricted random SE(3) configurations.
    contact.floor_eps = 10.0
    contact.upright_threshold = 1.0
    contact.tangent_translation_noise_std = 0.002
    contact.tangent_rotation_noise_std_rad = 0.01
    nonpenetrating = contact_variant.startswith("nonpenetrating")
    contact.penetration_eps = (
        float(nonpenetrating_penetration_eps) if nonpenetrating else 5e-4
    )
    contact.rejection_refill = nonpenetrating
    contact.rejection_max_rounds = 64 if nonpenetrating else 1
    contact.rejection_apply_tangent_gaussian = (
        nonpenetrating and bool(perturb_nonpenetrating)
    )
    if (
        contact_variant == "nonpenetrating_contact"
        and contact.rejection_apply_tangent_gaussian
    ):
        contact.name = "contact_gen_prl_nonpenetrating_contact_perturbed_500k"
    if contact_variant in {"paper_contact", "paper_head"}:
        contact.contact_geometry_mode = CONTACT_GEOMETRY_TANGENT_GAUSSIAN
        contact.M = 1
    elif contact_variant == "raw_contact":
        contact.contact_geometry_mode = CONTACT_GEOMETRY_INTERSECTING_ANCHORS
        contact.M = 1
    else:
        contact.contact_geometry_mode = CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION
        contact.M = 256
        contact.penetration_check_mode = PENETRATION_CHECK_BIDIRECTIONAL
        if (
            contact_variant
            == "nonpenetrating_contact_concavity_biased"
        ):
            contact.name = (
                "contact_gen_prl_nonpenetrating_contact_"
                "concavity_global_ranked_500k"
            )
            contact.rotation_selection = (
                ROTATION_SELECTION_MOST_CAVITY_CENTERED
            )

    pretrain = cfg.pretrain
    pretrain.name = (
        f"ce_prl_unicorn_d{int(transformer_depth)}_full_{contact_variant}"
    )
    pretrain.enabled = True
    pretrain.retrain = False
    pretrain.mode = "tce_multitask"
    pretrain.dataset_manifest = None
    pretrain.use_geometry_candidates = True
    pretrain.max_files = 0
    pretrain.max_contacts_per_file = 0
    pretrain.enabled_heads = ["contact"]
    pretrain.tasks.sdf = False
    pretrain.tasks.diffusion = False
    pretrain.tasks.postcontact = False
    pretrain.tasks.contact = True
    pretrain.num_precontact_steps = 0
    pretrain.encoder_input_centering = "object_center"
    pretrain.condition_normalization = False
    pretrain.augment = True
    pretrain.epochs = 50
    # The configured batch is per rank: 128 x 8 GPUs = paper batch 1024.
    pretrain.batch.batch_size = 128
    pretrain.optimizer.name = "sam"
    pretrain.optimizer.learning_rate = 2e-4
    pretrain.optimizer.min_learning_rate = 1e-6
    pretrain.optimizer.scheduler = "cosine"
    pretrain.optimizer.weight_decay = 0.001
    pretrain.optimizer.max_gradient_norm = 1000.0
    pretrain.optimizer.sam_rho = 0.05
    pretrain.unicorn.num_patches = 16
    pretrain.unicorn.decoder_type = "paper_cmlp_cbn"
    pretrain.unicorn.decoder_hidden_dims = [128, 128]
    pretrain.unicorn.positive_patch_fraction = 0.5
    # Contact quality is the only controlled difference. Every active variant
    # uses cached exact-mesh SDF with the same explicit distance threshold.
    pretrain.unicorn.label.contact_eps = float(contact_eps)
    pretrain.unicorn.label.source = "precomputed_mesh_sdf"
    pretrain.unicorn.augment.paper_pair_augmentation = True
    pretrain.unicorn.augment.rotation_range = (-3.141592653589793, 3.141592653589793)
    pretrain.unicorn.augment.translation_range = (-0.1, 0.1)
    pretrain.unicorn.augment.log_scale_range = (0.0, 0.0)
    pretrain.unicorn.augment.noise_std = float(point_jitter_std)
    pretrain.wandb_project = "ce_prl_contact_quality_pretrain"
    pretrain.wandb_run_name = name
    cfg.pretrain_reuse = None

    cfg.rl.launch.wandb_project = "dgn_set"
    cfg.rl.ppo.max_iterations = int(dgn_iterations)
    return cfg


def parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg(
    name: str,
    *,
    contact_variant: str,
) -> ExpCfg:
    """Matched 1M-case contact-quality experiment with 1 mm point jitter."""

    cfg = parallel_paper_contact_quality_rl_cfg(
        name,
        contact_variant=contact_variant,
        point_jitter_std=0.001,
        contact_eps=0.002,
        dgn_iterations=5_000,
        perturb_nonpenetrating=False,
        nonpenetrating_penetration_eps=5e-4,
    )
    # Keep B fixed so the per-file contact distribution is unchanged. Doubling
    # the balanced pair plan gives more object/gripper diversity and exactly
    # 2,000 * 500 = 1,000,000 cases before the deterministic dataset split.
    cfg.contact_gen.name = f"contact_gen_prl_{contact_variant}_1m"
    cfg.contact_gen.num_pairs = 2000
    cfg.contact_gen.B = 500
    cfg.pretrain.epochs = 50
    cfg.pretrain.unicorn.augment.log_scale_range = (0.0, 0.0)
    cfg.pretrain.unicorn.augment.noise_std = 0.001
    return cfg


def parallel_concavity_sdf_regression_rl_cfg(
    name: str,
    *,
    transformer_depth: int = 1,
    dgn_iterations: int = 10_000,
) -> ExpCfg:
    """Concavity-biased contacts with per-patch minimum-SDF regression."""

    cfg = parallel_paper_contact_quality_rl_cfg(
        name,
        contact_variant="nonpenetrating_contact_concavity_biased",
        transformer_depth=transformer_depth,
        dgn_iterations=dgn_iterations,
    )
    objective_name = (
        f"ce_prl_unicorn_d{int(transformer_depth)}_full_nonpenetrating_contact_"
        "concavity_biased_sdf"
    )
    cfg.model.name = objective_name
    cfg.pretrain.name = objective_name
    cfg.pretrain.enabled_heads = ["sdf"]
    cfg.pretrain.tasks.sdf = True
    cfg.pretrain.tasks.diffusion = False
    cfg.pretrain.tasks.postcontact = False
    cfg.pretrain.tasks.contact = False
    cfg.pretrain.sdf_head_mode = "patch"
    cfg.pretrain.decoder_pooling = "min"
    cfg.pretrain.loss.sdf_relative_loss = False
    # Reuse the exact mutual signed distances cached by contact generation.
    # Missing arrays are fatal; the model never falls back to mesh queries.
    cfg.pretrain.unicorn.label.source = "precomputed_mesh_sdf"
    return cfg


def parallel_kinematic_conditioning_rl_cfg(
    name: str,
    *,
    contact_variant: str = "nonpenetrating_contact_concavity_biased",
    transformer_depth: int = 1,
    dgn_iterations: int = 10_000,
) -> ExpCfg:
    """Selected contact recipe with shared-PointNet three-state kinematics."""

    if int(transformer_depth) < 1:
        raise ValueError("transformer_depth must be >= 1")
    if int(dgn_iterations) < 1:
        raise ValueError("dgn_iterations must be >= 1")
    cfg = parallel_paper_contact_quality_rl_cfg(
        name,
        contact_variant=contact_variant,
        transformer_depth=transformer_depth,
        dgn_iterations=dgn_iterations,
    )
    architecture_name = (
        f"ce_prl_unicorn_d{int(transformer_depth)}_full_{contact_variant}_kinematic"
    )
    cfg.model.name = architecture_name
    cfg.model.tce.vit_depth = int(transformer_depth)
    cfg.model.tce.kinematic_conditioning.enabled = True
    cfg.model.tce.kinematic_conditioning.state_fractions = (0.0, 0.5, 1.0)
    cfg.model.tce.kinematic_conditioning.attention_layers = int(
        transformer_depth
    )
    cfg.model.tce.kinematic_conditioning.delta_std = 0.15
    cfg.pretrain.name = architecture_name
    cfg.pretrain.wandb_run_name = name

    cfg.rl.observation.include_kinematic_gripper_clouds = True
    cfg.rl.observation.point_cloud_noise_enabled = False
    tool_index = cfg.rl.observation.layout.index("tool_cloud_flat")
    cfg.rl.observation.layout.insert(
        tool_index + 1, "kinematic_gripper_clouds_flat"
    )
    cfg.rl.ppo.max_iterations = int(dgn_iterations)
    return cfg


def general_d4_full_rl_cfg(
    name: str,
    *,
    contact_quality: str,
    architecture: str,
) -> ExpCfg:
    """Strict 200-parallel/200-revolute D4 contact-quality comparison."""

    variants = {
        "paper": "paper_contact",
        "concavity_global": "nonpenetrating_contact_concavity_biased",
    }
    if contact_quality not in variants:
        raise ValueError(
            f"contact_quality must be one of {tuple(variants)}, got {contact_quality!r}"
        )
    if architecture not in {"raw", "kinematic"}:
        raise ValueError("architecture must be 'raw' or 'kinematic'")

    contact_variant = variants[contact_quality]
    if architecture == "kinematic":
        cfg = parallel_kinematic_conditioning_rl_cfg(
            name,
            contact_variant=contact_variant,
            transformer_depth=4,
            dgn_iterations=5_000,
        )
    else:
        cfg = parallel_paper_contact_quality_rl_cfg(
            name,
            contact_variant=contact_variant,
            transformer_depth=4,
            dgn_iterations=5_000,
        )

    architecture_name = (
        f"ce_general_d4_full_{contact_quality}_{architecture}"
    )
    cfg.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
    cfg.num_gpus = 8
    cfg.model.name = architecture_name
    cfg.pretrain.name = architecture_name
    cfg.pretrain.wandb_project = "ce_general_contact_quality_pretrain"
    cfg.pretrain.wandb_run_name = name
    cfg.contact_gen.name = (
        f"contact_gen_general_{contact_quality}_400tools_128bin_500k"
    )
    # 2,000 pairs x 250 cases keeps the 500k budget while assigning exactly
    # five distinct object pairs to every one of the 400 grippers.
    cfg.contact_gen.num_pairs = 2_000
    cfg.contact_gen.B = 250
    cfg.rl.isaac_task_id = "cross-embodiment-gripper-v0"
    cfg.rl.env.robot_mode = "cross_embodiment_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    return cfg


def general_d4_hamnet_rl_cfg(name: str) -> ExpCfg:
    """Combined-gripper concavity/raw D4 encoder with a HAMNet policy trunk."""

    cfg = general_d4_full_rl_cfg(
        name,
        contact_quality="concavity_global",
        architecture="raw",
    )
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = (
        "ce_general_d4_full_concavity_global_raw_dgn_5k.py"
    )
    cfg.rl.actor_critic_class = "ActorCriticTGHAMNet"
    cfg.rl.hamnet_num_modules = 4
    cfg.rl.hamnet_hidden_dims = (256, 128, 128, 64)
    cfg.rl.hamnet_router_hidden_dims = (256, 256)
    return cfg


def parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(
    name: str,
) -> ExpCfg:
    """Parallel-only full attention over existing non-penetrating candidates."""

    cfg = parallel_depth1_full_attention_unicorn_rl_cfg(
        name,
        raw_contact=False,
    )
    cfg.pretrain.dataset_manifest = UNICORN_COMPARISON_CONTACT_DATASET
    cfg.pretrain.use_geometry_candidates = True
    cfg.pretrain.max_files = 2048
    cfg.pretrain.max_contacts_per_file = 524
    return cfg


def parallel_proven_nonpenetrating_recipe_paper_dataset_rl_cfg(
    name: str,
) -> ExpCfg:
    """Proven nonpenetrating recipe with only its contact dataset swapped."""

    cfg = parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(name)
    cfg.pretrain.dataset_manifest = PARALLEL_PAPER_1M_CONTACT_DATASET
    cfg.pretrain.wandb_run_name = name
    return cfg


def parallel_new200_proven_nonpenetrating_recipe_rl_cfg(
    name: str,
) -> ExpCfg:
    """Legacy nonpenetrating pretrain recipe and RL on the newest 200 tools."""

    cfg = parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(name)
    # Keep the pretraining geometry population and the RL embodiment population
    # identical: both are generated_gripper_000000 through 000199 from
    # gripper_new.
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.pretrain.dataset_manifest = PARALLEL_NONPENETRATING_1M_CONTACT_DATASET
    cfg.pretrain.wandb_run_name = name
    return cfg


def parallel_new200_proven_nonpenetrating_encoder_rl_cfg(
    name: str,
) -> ExpCfg:
    """RL-only transfer of the proven nonpenetrating encoder to new 200 tools."""

    cfg = parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(name)
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.model.pretrained_encoder.checkpoint_path = (
        PROVEN_PARALLEL_NONPENETRATING_ENCODER_CHECKPOINT
    )
    return cfg


def parallel_depth1_full_attention_diff_rl_cfg(name: str) -> ExpCfg:
    """Parallel-only diffusion-only pretrain and RL."""

    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.paths_yaml = GENERATED_GRIPPER_PATHS_YAML
    cfg.num_gpus = 8
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = True
    cfg.pretrain_reuse = None
    cfg.model.name = name
    cfg.model.tce.vit_depth = 1
    cfg.model.tce.vit_attention_mode = "joint_self"
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.pretrain.name = name
    cfg.pretrain.dataset_manifest = UNICORN_COMPARISON_CONTACT_DATASET
    cfg.pretrain.use_geometry_candidates = False
    cfg.pretrain.max_files = 0
    cfg.pretrain.max_contacts_per_file = 0
    cfg.pretrain.enabled_heads = ["diff"]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = True
    cfg.pretrain.tasks.postcontact = False
    cfg.pretrain.tasks.contact = False
    cfg.pretrain.wandb_project = "parallel_gripper_pretrain"
    cfg.pretrain.wandb_run_name = name
    return cfg


def parallel_depth1_full_attention_post_rl_cfg(name: str) -> ExpCfg:
    """Newest-200 parallel grippers with postcontact-only pretrain and RL."""

    cfg = generated_gripper_post_rl_cfg(name)
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.num_gpus = 8
    # This completed dataset was generated with the same newest 200-gripper
    # manifest and contains stabilization plus achieved postcontact targets.
    cfg.contact_gen.name = "contact_gen_gripper_new"
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = True
    cfg.pretrain_reuse = None
    cfg.model.name = name
    cfg.model.tce.vit_depth = 1
    cfg.model.tce.vit_attention_mode = "joint_self"
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.pretrain.name = name
    cfg.pretrain.dataset_manifest = PARALLEL_NEW_POSTCONTACT_DATASET
    cfg.pretrain.use_geometry_candidates = False
    cfg.pretrain.max_files = 0
    cfg.pretrain.max_contacts_per_file = 0
    cfg.pretrain.enabled_heads = ["postcontact"]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = False
    cfg.pretrain.tasks.postcontact = True
    cfg.pretrain.tasks.contact = False
    cfg.pretrain.num_precontact_steps = 0
    cfg.pretrain.wandb_project = "parallel_gripper_pretrain"
    cfg.pretrain.wandb_run_name = name
    return cfg


def generated_revolute_unicorn_rl_cfg(
    name: str, *, ours_tce: bool = False
) -> ExpCfg:
    """UniCORN policy trained exclusively with generated revolute grippers."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=ours_tce)
    cfg.paths_yaml = GENERATED_REVOLUTE_PATHS_YAML
    cfg.rl.isaac_task_id = "one-dof-gripper-v0"
    cfg.rl.env.robot_mode = "one_dof_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "generated_revolute_gripper"
    return cfg


def generated_gripper_intersecting_depth1_unicorn_rl_cfg(name: str) -> ExpCfg:
    """RL config that runs depth-1 intersecting contact/pretrain stages inline."""

    pretrain_cfg = unicorn_ours_intersecting_depth1_pretrain_cfg()
    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.paths_yaml = pretrain_cfg.paths_yaml
    cfg.contact_gen = clone_cfg(pretrain_cfg.contact_gen)
    cfg.pretrain = clone_cfg(pretrain_cfg.pretrain)
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.model = clone_cfg(pretrain_cfg.model)
    cfg.model.pretrained_encoder.checkpoint_path = None
    return cfg


def cross_embodiment_gripper_unicorn_rl_cfg(
    name: str, *, ours_tce: bool = False
) -> ExpCfg:
    """One policy with a 50/50 generated-parallel/revolute rank split."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=ours_tce)
    cfg.paths_yaml = CROSS_EMBODIMENT_GRIPPER_PATHS_YAML
    cfg.rl.isaac_task_id = "cross-embodiment-gripper-v0"
    cfg.rl.env.robot_mode = "cross_embodiment_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "cross_embodiment_gripper"
    return cfg


def generated_gripper_patch_distance_pointnet_rl_cfg(name: str) -> ExpCfg:
    """Frozen XYZ-only patch PointNet loaded from a standalone checkpoint."""

    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None

    cfg.model.name = "patch_distance_pointnet"
    cfg.model.encoder_backend = "patch_distance_pointnet"
    cfg.model.pretrained_encoder.name = "patch_distance_pointnet"
    cfg.model.pretrained_encoder.adapter = "patch_distance_pointnet_strict"
    cfg.model.pretrained_encoder.checkpoint_path = str(
        Path(__file__).resolve().parents[1]
        / ".pretrained_checkpoints"
        / "patch_distance_pointnet"
        / "best.pt"
    )
    cfg.rl.actor_critic_class = "ActorCriticTG"
    cfg.rl.freeze_encoder = True
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg


def generated_gripper_oracle_patch_rl_cfg(name: str) -> ExpCfg:
    """Generated-gripper RL using only explicit privileged patch distances."""

    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.rl.isaac_task_id = "generated-gripper-oracle-v0"
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.enabled = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.model.name = "oracle_patch_distance"
    cfg.model.encoder_backend = "oracle_patch"
    cfg.model.pretrained_encoder.name = "oracle_patch"
    cfg.model.pretrained_encoder.adapter = "oracle_none"
    cfg.model.pretrained_encoder.checkpoint_path = None
    cfg.model.oracle_patch.num_points = cfg.model.tce.num_points
    cfg.model.oracle_patch.num_patches = cfg.model.tce.num_points // cfg.model.tce.patch_size
    cfg.model.oracle_patch.patch_size = cfg.model.tce.patch_size
    cfg.model.oracle_patch.encoder_channel = cfg.model.tce.encoder_channel
    cfg.rl.actor_critic_class = "ActorCriticTG"
    # The 8D -> 128D oracle embedding is learned jointly with PPO.
    cfg.rl.freeze_encoder = False
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.rl.observation.include_oracle_mesh_sdf = True
    if "oracle_mesh_signed_sdf" not in cfg.rl.observation.layout:
        cfg.rl.observation.layout.insert(2, "oracle_mesh_signed_sdf")
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg


def generated_gripper_oracle_pointmesh_pointnet_rl_cfg(name: str) -> ExpCfg:
    """Pretrain and run RL with unsigned point-to-mesh patchwise PointNet tokens."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=False)
    cfg.rl.isaac_task_id = "generated-gripper-oracle-pointmesh-v0"
    cfg.model.name = "oracle_pointmesh_pointnet"
    cfg.model.encoder_backend = "oracle_pointmesh_pointnet"
    cfg.model.pretrained_encoder.name = "oracle_pointmesh_pointnet"
    cfg.model.pretrained_encoder.adapter = "oracle_pointmesh_pointnet_strict"
    cfg.model.pretrained_encoder.checkpoint_path = None
    pointmesh = cfg.model.oracle_pointmesh_pointnet
    pointmesh.num_points = cfg.model.unicorn.num_points
    pointmesh.num_patches = cfg.model.unicorn.num_patches
    pointmesh.patch_size = cfg.model.unicorn.patch_size
    pointmesh.encoder_channel = cfg.model.unicorn.encoder_channel
    cfg.pretrain.name = "oracle_pointmesh_pointnet_contact_generated_gripper"
    cfg.pretrain.mode = "oracle_pointmesh_contact"
    cfg.pretrain.epochs = 3
    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = False
    cfg.pretrain.wandb_run_name = "generated_gripper_oracle_pointmesh_pointnet_contact"
    cfg.pretrain_reuse = None
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.rl.actor_critic_class = "ActorCriticTG"
    cfg.rl.freeze_encoder = True
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.rl.observation.include_oracle_mesh_unsigned_distance = True
    if "oracle_mesh_unsigned_distance" not in cfg.rl.observation.layout:
        cfg.rl.observation.layout.insert(2, "oracle_mesh_unsigned_distance")
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg


def generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    name: str,
    *,
    checkpoint_path: str | None,
) -> ExpCfg:
    """RL with the fitted fast nearest-point-cloud patchwise PointNet."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.model.name = "oracle_pointcloud_pointnet"
    cfg.model.encoder_backend = "oracle_pointcloud_pointnet"
    cfg.model.pretrained_encoder.name = "oracle_pointcloud_pointnet"
    cfg.model.pretrained_encoder.adapter = (
        "oracle_pointcloud_pointnet_strict" if checkpoint_path else "oracle_none"
    )
    cfg.model.pretrained_encoder.checkpoint_path = checkpoint_path
    pointcloud = cfg.model.oracle_pointcloud_pointnet
    pointcloud.num_points = cfg.model.tce.num_points
    pointcloud.num_patches = cfg.model.tce.num_points // cfg.model.tce.patch_size
    pointcloud.patch_size = cfg.model.tce.patch_size
    pointcloud.encoder_channel = cfg.model.tce.encoder_channel
    cfg.pretrain.enabled = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.rl.actor_critic_class = "ActorCriticTG"
    # This checkpoint is an initialization: PPO may adapt the small PointNet
    # and its learned rank-10 reconstruction to control.
    cfg.rl.freeze_encoder = False
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg


def _generated_gripper_native_pointnet_pretrain_rl_cfg(
    name: str,
    *,
    objective: str,
) -> ExpCfg:
    """Pretrain the exact direct-128 PointNet used unfrozen in RL."""

    if objective not in {"diff", "postcontact"}:
        raise ValueError(
            "Native PointNet pretraining objective must be diff or postcontact"
        )

    cfg = generated_gripper_diff_post_rl_cfg(name)
    cfg.paths_yaml = GENERATED_GRIPPER_PATHS_YAML
    cfg.num_gpus = 8
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False

    cfg.model.name = "oracle_pointcloud_pointnet"
    cfg.model.encoder_backend = "oracle_pointcloud_pointnet"
    cfg.model.pretrained_encoder.name = "oracle_pointcloud_pointnet"
    cfg.model.pretrained_encoder.adapter = (
        "oracle_pointcloud_pointnet_pretrain_strict"
    )
    cfg.model.pretrained_encoder.checkpoint_path = None
    pointcloud = cfg.model.oracle_pointcloud_pointnet
    pointcloud.num_points = cfg.model.tce.num_points
    pointcloud.num_patches = cfg.model.tce.num_points // cfg.model.tce.patch_size
    pointcloud.patch_size = cfg.model.tce.patch_size
    pointcloud.encoder_channel = cfg.model.tce.encoder_channel
    pointcloud.feature_mode = "fast11"
    pointcloud.load_fitted_weights = True
    pointcloud.use_rank10_bottleneck = False
    pointcloud.token_mode = "patches"

    cfg.pretrain.enabled = True
    cfg.pretrain.retrain = True
    cfg.pretrain_reuse = None
    cfg.pretrain.mode = (
        "oracle_pointcloud_diffusion"
        if objective == "diff"
        else "oracle_pointcloud_postcontact"
    )
    cfg.pretrain.name = name
    cfg.pretrain.dataset_manifest = UNICORN_COMPARISON_CONTACT_DATASET
    cfg.pretrain.use_geometry_candidates = False
    cfg.pretrain.max_files = 0
    cfg.pretrain.max_contacts_per_file = 0
    cfg.pretrain.enabled_heads = [objective]
    cfg.pretrain.tasks.sdf = False
    cfg.pretrain.tasks.diffusion = objective == "diff"
    cfg.pretrain.tasks.postcontact = objective == "postcontact"
    cfg.pretrain.tasks.contact = False
    if objective == "postcontact":
        cfg.pretrain.num_precontact_steps = 0
    cfg.pretrain.encoder_input_centering = "object_center"
    cfg.pretrain.wandb_project = "parallel_gripper_pretrain"
    cfg.pretrain.wandb_run_name = name

    cfg.rl.actor_critic_class = "ActorCriticTG"
    cfg.rl.freeze_encoder = False
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg


def generated_gripper_native_pointnet_diffusion_rl_cfg(name: str) -> ExpCfg:
    """Diffusion-pretrain the exact direct-128 PointNet used unfrozen in RL."""

    return _generated_gripper_native_pointnet_pretrain_rl_cfg(
        name,
        objective="diff",
    )


def generated_gripper_native_pointnet_postcontact_rl_cfg(name: str) -> ExpCfg:
    """Post-pretrain the exact direct-128 PointNet used unfrozen in RL."""

    return _generated_gripper_native_pointnet_pretrain_rl_cfg(
        name,
        objective="postcontact",
    )


def parallel_native_pointnet_normalized_postcontact_rl_cfg(name: str) -> ExpCfg:
    """Newest-200 direct-128 PointNet with normalized post-only pretraining."""

    cfg = generated_gripper_native_pointnet_postcontact_rl_cfg(name)
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.contact_gen.name = "contact_gen_gripper_new"
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.dataset_manifest = PARALLEL_NEW_POSTCONTACT_DATASET
    cfg.model.oracle_pointcloud_pointnet.input_normalization = (
        "fast11_probe_v1"
    )
    cfg.model.pretrained_encoder.adapter = (
        "oracle_pointcloud_pointnet_normalized_pretrain_strict"
    )
    cfg.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
    return cfg


def general_frozen_gg_oracle_pointcloud_pointnet_rl_cfg(
    name: str,
    *,
    checkpoint_path: str,
) -> ExpCfg:
    """Combined-gripper RL with only the GG-adapted PointNet frozen."""

    cfg = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=checkpoint_path,
    )
    cfg.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
    cfg.num_gpus = 8
    cfg.model.pretrained_encoder.schema = "rsl_rl_checkpoint_v1"
    cfg.model.pretrained_encoder.adapter = (
        "oracle_pointcloud_pointnet_rl_encoder_strict"
    )
    cfg.rl.freeze_encoder = True
    cfg.rl.isaac_task_id = "cross-embodiment-gripper-v0"
    cfg.rl.env.robot_mode = "cross_embodiment_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "ce_general_frozen_oracle_pointnet"
    return cfg


def general_fitted_oracle_pointcloud_pointnet_rl_cfg(name: str) -> ExpCfg:
    """Combined grippers with the original fitted PointNet trainable in PPO."""

    cfg = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
    )
    cfg.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
    cfg.num_gpus = 8
    cfg.rl.isaac_task_id = "cross-embodiment-gripper-v0"
    cfg.rl.env.robot_mode = "cross_embodiment_gripper"
    cfg.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
    cfg.rl.launch.wandb_project = "ce_general_fitted_oracle_pointnet"
    return cfg


def general_unfrozen_gg_oracle_pointcloud_pointnet_rl_cfg(
    name: str,
    *,
    checkpoint_path: str,
) -> ExpCfg:
    """Combined-gripper GG-adapted PointNet transfer with PPO adaptation."""

    cfg = general_frozen_gg_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=checkpoint_path,
    )
    cfg.rl.freeze_encoder = False
    cfg.rl.launch.wandb_project = "ce_general_unfrozen_oracle_pointnet"
    return cfg


def parallel_frozen_gg_oracle_pointcloud_pointnet_rl_cfg(
    name: str,
    *,
    checkpoint_path: str,
) -> ExpCfg:
    """Newest-200 parallel-gripper RL with the GG-adapted PointNet frozen."""

    cfg = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=checkpoint_path,
    )
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.num_gpus = 8
    cfg.model.pretrained_encoder.schema = "rsl_rl_checkpoint_v1"
    cfg.model.pretrained_encoder.adapter = (
        "oracle_pointcloud_pointnet_rl_encoder_strict"
    )
    cfg.rl.freeze_encoder = True
    cfg.rl.launch.wandb_project = "ce_prl_frozen_oracle_pointnet"
    return cfg


def parallel_fitted_oracle_pointcloud_pointnet_rl_cfg(name: str) -> ExpCfg:
    """Newest-200 parallel grippers with the original fitted PointNet trainable."""

    cfg = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
    )
    cfg.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
    cfg.num_gpus = 8
    cfg.rl.launch.wandb_project = "ce_prl_fitted_oracle_pointnet"
    return cfg


def parallel_unfrozen_gg_oracle_pointcloud_pointnet_rl_cfg(
    name: str,
    *,
    checkpoint_path: str,
) -> ExpCfg:
    """Parallel GG-adapted PointNet transfer with PPO encoder adaptation."""

    cfg = parallel_frozen_gg_oracle_pointcloud_pointnet_rl_cfg(
        name,
        checkpoint_path=checkpoint_path,
    )
    cfg.rl.freeze_encoder = False
    cfg.rl.launch.wandb_project = "ce_prl_unfrozen_oracle_pointnet"
    return cfg


def generated_gripper_oracle_pointcloud_patch_oracle_rl_cfg(
    name: str,
    *,
    checkpoint_path: str,
) -> ExpCfg:
    """RL initialized from the deep analytic 35D point-cloud patch probe."""

    cfg = generated_gripper_unicorn_rl_cfg(name, ours_tce=True)
    cfg.model.name = "oracle_pointcloud_patch_oracle"
    cfg.model.encoder_backend = "oracle_pointcloud_patch_oracle"
    cfg.model.pretrained_encoder.name = "oracle_pointcloud_patch_oracle"
    cfg.model.pretrained_encoder.adapter = "oracle_pointcloud_patch_oracle_strict"
    cfg.model.pretrained_encoder.checkpoint_path = checkpoint_path
    oracle = cfg.model.oracle_pointcloud_patch_oracle
    oracle.num_points = cfg.model.tce.num_points
    oracle.num_patches = cfg.model.tce.num_points // cfg.model.tce.patch_size
    oracle.patch_size = cfg.model.tce.patch_size
    oracle.encoder_channel = cfg.model.tce.encoder_channel
    cfg.pretrain.enabled = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = None
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.rl.actor_critic_class = "ActorCriticTG"
    # The fitted deep patch MLP is an initialization and remains adaptable by PPO.
    cfg.rl.freeze_encoder = False
    cfg.rl.observation.model_input_centering = "object_center"
    cfg.model.policy_fusion.reuse_pretrain_pose_cross_attn = False
    return cfg

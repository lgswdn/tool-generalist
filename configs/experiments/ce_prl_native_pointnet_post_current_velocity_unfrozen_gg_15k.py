"""GG15k continuation of current-velocity CE-PRL native-PointNet DGN."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


POST_PRETRAIN_SOURCE = "panda_general_native_pointnet_post_original400_dgn_5k"
PARENT = "ce_prl_native_pointnet_post_current_velocity_unfrozen_dgn_5k"
NAME = "ce_prl_native_pointnet_post_current_velocity_unfrozen_gg_15k"

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain_reuse = f"{POST_PRETRAIN_SOURCE}.py"
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = False
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_gg_comparison(EXP_CFG)

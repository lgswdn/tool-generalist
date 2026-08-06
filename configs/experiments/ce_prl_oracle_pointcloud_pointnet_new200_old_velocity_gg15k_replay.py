"""Replay the original oracle GG run with new 200 grippers and old dynamics."""

from configs.panda_comparison_common import configure_gg_comparison
from configs.panda_experiment_common import (
    FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    ORIGINAL_ORACLE_POINTCLOUD_POINTNET_DGN5K_CHECKPOINT,
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "ce_prl_oracle_pointcloud_pointnet_new200_old_velocity_gg15k_replay",
    checkpoint_path=FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
EXP_CFG.rl.init_checkpoint = (
    ORIGINAL_ORACLE_POINTCLOUD_POINTNET_DGN5K_CHECKPOINT
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.launch.wandb_project = "ungrasp"

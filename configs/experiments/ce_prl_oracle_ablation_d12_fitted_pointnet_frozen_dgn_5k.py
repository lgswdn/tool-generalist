"""Test whether the offline-fitted depth-12 PointNet needs PPO adaptation."""

from configs.oracle_pointnet_rebuild_common import POINTNET_CHECKPOINT
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


NAME = "ce_prl_oracle_ablation_d12_fitted_pointnet_frozen_dgn_5k"
if not POINTNET_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Fitted PointNet is missing: {POINTNET_CHECKPOINT}")

EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    NAME,
    checkpoint_path=str(POINTNET_CHECKPOINT),
)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"


"""Rich 21D direct-128 PointNet learned from scratch by full-YES RL for 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k",
    checkpoint_path=None,
)
EXP_CFG.model.oracle_pointcloud_pointnet.feature_mode = "rich21"
EXP_CFG.model.oracle_pointcloud_pointnet.load_fitted_weights = False
EXP_CFG.model.oracle_pointcloud_pointnet.use_rank10_bottleneck = False
configure_full_yes_comparison(EXP_CFG)

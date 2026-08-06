"""Full-YES RL with an XYZ-only, patch-distance-pretrained PointNet."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_patch_distance_pointnet_rl_cfg,
)


EXP_CFG = generated_gripper_patch_distance_pointnet_rl_cfg(
    "panda_general_patch_distance_pointnet_full_yes_5k"
)
configure_full_yes_comparison(EXP_CFG)

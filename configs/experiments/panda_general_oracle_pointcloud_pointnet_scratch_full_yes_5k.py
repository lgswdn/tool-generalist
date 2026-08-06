"""Full-YES 5k PointNet RL with all learned encoder weights trained from scratch."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


# Used only for the fixed 11D input mean/std.  No fitted PointNet, rank-10, or
# 10->128 projection weights are loaded when load_fitted_weights is false.
NORMALIZATION_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_scratch_full_yes_5k",
    checkpoint_path=NORMALIZATION_CHECKPOINT,
)
EXP_CFG.model.oracle_pointcloud_pointnet.load_fitted_weights = False
configure_full_yes_comparison(EXP_CFG)

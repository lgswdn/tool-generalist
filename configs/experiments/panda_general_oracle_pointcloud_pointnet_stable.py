"""Fast nearest-point-cloud PointNet RL with stable-pose goals only."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


FAST_POINTNET_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_stable",
    checkpoint_path=FAST_POINTNET_CHECKPOINT,
)
configure_full_yes_comparison(EXP_CFG)

# Restrict every sampled target to a physically stable object pose.  With the
# curriculum disabled, these probabilities remain fixed for the entire run.
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 1.0

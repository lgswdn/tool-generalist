"""Corrected unpooled 1,024-point-token oracle policy on full-YES for 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


FAST_POINTNET_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointtokens_v2_full_yes_5k",
    checkpoint_path=FAST_POINTNET_CHECKPOINT,
)
EXP_CFG.model.oracle_pointcloud_pointnet.token_mode = "points"
configure_full_yes_comparison(EXP_CFG)

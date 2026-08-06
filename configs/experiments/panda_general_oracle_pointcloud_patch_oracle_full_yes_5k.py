"""Deep analytic point-cloud patch oracle initialized RL on full-YES for 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_patch_oracle_rl_cfg,
)


FAST_PATCH_ORACLE_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_patch_oracle35/fast_patch_oracle35_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_patch_oracle_rl_cfg(
    "panda_general_oracle_pointcloud_patch_oracle_full_yes_5k",
    checkpoint_path=FAST_PATCH_ORACLE_CHECKPOINT,
)
configure_full_yes_comparison(EXP_CFG)

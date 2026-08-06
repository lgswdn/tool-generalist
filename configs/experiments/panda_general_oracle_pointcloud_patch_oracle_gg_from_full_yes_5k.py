"""Analytic point-cloud patch oracle GG 15k transfer from full-YES."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_patch_oracle_rl_cfg,
)


PARENT_EXPERIMENT = "panda_general_oracle_pointcloud_patch_oracle_full_yes_5k"
FAST_PATCH_ORACLE_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_patch_oracle35/fast_patch_oracle35_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_patch_oracle_rl_cfg(
    "panda_general_oracle_pointcloud_patch_oracle_gg_from_full_yes_5k",
    checkpoint_path=FAST_PATCH_ORACLE_CHECKPOINT,
)
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_patch_oracle",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

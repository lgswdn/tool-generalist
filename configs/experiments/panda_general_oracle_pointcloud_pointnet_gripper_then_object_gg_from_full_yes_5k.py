"""GG 15k transfer for state -> gripper -> object sequential fusion."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


PARENT_EXPERIMENT = (
    "panda_general_oracle_pointcloud_pointnet_gripper_then_object_full_yes_5k"
)
FAST_POINTNET_CHECKPOINT = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_gripper_then_object_gg_from_full_yes_5k",
    checkpoint_path=FAST_POINTNET_CHECKPOINT,
)
EXP_CFG.model.policy_fusion.cross_attn_layers = 2
EXP_CFG.model.policy_fusion.cross_attn_token_order = "tool_then_object"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

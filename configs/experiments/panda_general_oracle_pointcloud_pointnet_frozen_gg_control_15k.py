"""Freeze the fitted PointNet in the original-setting GG transfer control."""

from pathlib import Path

from configs.panda_comparison_common import configure_gg_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_PATHS_YAML,
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


NAME = "panda_general_oracle_pointcloud_pointnet_frozen_gg_control_15k"
POINTNET_CHECKPOINT = Path(
    "/mnt/home/zhengyixin/tool-generalist/artifacts/probes/"
    "rank10_patch_pointnet/fast_pointcloud11/fast_pointcloud11_best.pt"
)
PARENT_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "20260719T092442Z/model_best.pt"
)
for required in (POINTNET_CHECKPOINT, PARENT_CHECKPOINT):
    if not required.is_file():
        raise FileNotFoundError(f"Required control checkpoint is missing: {required}")


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    NAME,
    checkpoint_path=str(POINTNET_CHECKPOINT),
)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_PATHS_YAML
EXP_CFG.rl.init_checkpoint = str(PARENT_CHECKPOINT)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.launch.wandb_project = "ungrasp"

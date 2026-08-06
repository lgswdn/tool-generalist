"""Train the rebuilt combined-gripper PointNet on DGN for 5k."""

from configs.oracle_pointnet_general_rebuild_common import POINTNET_CHECKPOINT
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


NAME = "ce_general_oracle_rebuild_d12_pointnet_dgn_5k"
if not POINTNET_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Fitted PointNet is missing: {POINTNET_CHECKPOINT}")

EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    NAME,
    checkpoint_path=str(POINTNET_CHECKPOINT),
)
EXP_CFG.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.rl.isaac_task_id = "cross-embodiment-gripper-v0"
EXP_CFG.rl.env.robot_mode = "cross_embodiment_gripper"
EXP_CFG.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
EXP_CFG.rl.freeze_encoder = False
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

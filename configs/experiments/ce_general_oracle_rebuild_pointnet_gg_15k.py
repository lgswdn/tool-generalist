"""Transfer the rebuilt combined-gripper PointNet DGN policy to GG for 15k."""

from configs.oracle_pointnet_general_rebuild_common import POINTNET_CHECKPOINT
from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


PARENT = "ce_general_oracle_rebuild_d12_pointnet_dgn_5k"
NAME = "ce_general_oracle_rebuild_d12_pointnet_gg_15k"

EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    NAME,
    checkpoint_path=str(POINTNET_CHECKPOINT),
)
EXP_CFG.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.rl.isaac_task_id = "cross-embodiment-gripper-v0"
EXP_CFG.rl.env.robot_mode = "cross_embodiment_gripper"
EXP_CFG.rl.observation.tool_cloud_source = "gripper_cloud_cache_v1"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
    expected_pretrained_encoder_checkpoint=str(POINTNET_CHECKPOINT),
    expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_best.pt",
)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = False
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.launch.wandb_project = "ungrasp"

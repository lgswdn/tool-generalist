"""Original400 native post-PointNet DGN5k at safe velocity, encoder frozen."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


NAME = "panda_general_native_pointnet_post_original400_safe_velocity_frozen_dgn_5k"
PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "panda_general_native_pointnet_post_original400_dgn_5k/"
    "contact_gen_generated_gripper/"
    "panda_general_native_pointnet_post_original400_dgn_5k_oracle_pointcloud_pointnet/"
    "f41517a2990854a1d27510e02ff09e83eee009aa95a667315362523f2ffb3d46/"
    "best.pt"
)

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = PRETRAIN_CHECKPOINT
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

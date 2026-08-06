"""Temporary-resource diagnostic with the post-pretrained PointNet frozen."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


POST_PRETRAIN_SOURCE = "panda_general_native_pointnet_post_original400_dgn_5k"
NAME = "panda_general_native_pointnet_post_original400_frozen_dgn_5k"

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain_reuse = f"{POST_PRETRAIN_SOURCE}.py"
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
# This run may be preempted, so keep denser recoverable progress checkpoints.
EXP_CFG.rl.ppo.save_interval = 100
EXP_CFG.rl.launch.wandb_project = "ungrasp"

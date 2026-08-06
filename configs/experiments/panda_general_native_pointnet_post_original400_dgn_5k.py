"""Native direct-128 PointNet post-only pretraining and original-setting DGN."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


NAME = "panda_general_native_pointnet_post_original400_dgn_5k"

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

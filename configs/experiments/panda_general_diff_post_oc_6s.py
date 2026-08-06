"""Generated-gripper object-centered diff-post RL with a 6s episode limit."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_diff_post_oc_6s")
EXP_CFG.rl.env.episode_length_s = 6.0

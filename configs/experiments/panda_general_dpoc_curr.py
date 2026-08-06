"""Generated-gripper object-centered diff-post RL with stable-pose curriculum."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_curr")
EXP_CFG.rl.curriculum.enabled = True
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 10000
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3

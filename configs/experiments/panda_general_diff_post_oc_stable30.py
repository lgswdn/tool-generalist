"""Generated-gripper object-centered diff-post RL with a fixed 30/70 stable/random goal mix."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_diff_post_oc_stable30")
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.3
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3

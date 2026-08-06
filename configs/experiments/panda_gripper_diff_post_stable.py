"""Official Panda-gripper diff-post RL with stable-pose target goals only."""

from configs.panda_experiment_common import official_panda_diff_post_rl_cfg


EXP_CFG = official_panda_diff_post_rl_cfg("panda_gripper_diff_post_stable")
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 1.0

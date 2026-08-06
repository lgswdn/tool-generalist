"""Generated-gripper object-centered diff-post RL with stable targets and grasp-lift targets."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_grasp")
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.3
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3
EXP_CFG.rl.object_pose_sampling.secondary_task = "grasp_lift"
EXP_CFG.rl.object_pose_sampling.grasp_lift_height = 0.10

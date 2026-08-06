"""Generated-gripper object-centered diff-post RL with SM stable-pose curriculum."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_curr_sm")
EXP_CFG.rl.actor_critic_class = "ActorCriticTGSM"
EXP_CFG.rl.separate_actor_critic_fusion = True
EXP_CFG.rl.observation.task_embedding_dim = 2
EXP_CFG.rl.observation.layout = [
    "object_cloud_flat",
    "tool_cloud_flat",
    "object_bbox_center",
    "tool_bbox_center",
    "hand_state",
    "robot_state",
    "previous_action",
    "relative_goal_pose",
    "task_embedding",
    "physics",
]
EXP_CFG.rl.curriculum.enabled = True
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 10000
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3

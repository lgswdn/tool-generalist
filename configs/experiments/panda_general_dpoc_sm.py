"""Generated-gripper object-centered diff-post RL with Soft-Module stable/random routing."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_sm")
EXP_CFG.rl.actor_critic_class = "ActorCriticTGSM"
EXP_CFG.rl.separate_actor_critic_fusion = True
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.3
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3
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

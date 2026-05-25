"""Fork-only RL on arbitrary target poses using the multitools diff-post encoder."""

from copy import deepcopy

from configs.experiments.multitools_diff_post import EXP_CFG as _MULTITOOLS_DIFF_POST_CFG


EXP_CFG = deepcopy(_MULTITOOLS_DIFF_POST_CFG)

EXP_CFG.name = "fork_unstable_diff_post"
EXP_CFG.num_gpus = 8
EXP_CFG.general.name = "fork_unstable_diff_post"
EXP_CFG.general.tools_selected_json = "configs/tool_selections/fork_only.json"
EXP_CFG.pretrain_reuse = "multitools_diff_post.py"

EXP_CFG.rl.name = "fork_unstable_diff_post"
EXP_CFG.rl.isaac_task_id = "tool-unstable-v0"
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.launch.run_name = "fork_unstable_diff_post"
EXP_CFG.rl.launch.wandb_project = "fork"

EXP_CFG.rl.reward.rotation_threshold = 0.2
EXP_CFG.rl.separate_actor_critic_fusion = True
EXP_CFG.rl.reward.object_goal_tracking_term_weight = 2.5
EXP_CFG.rl.reward.object_goal_tracking_fine_term_weight = 4.5
EXP_CFG.rl.reward.object_stillness_at_goal_term_weight = 250.0
EXP_CFG.rl.reward.stable_success_linear_velocity_threshold = 0.03
EXP_CFG.rl.reward.stable_success_angular_velocity_threshold = 0.15
EXP_CFG.rl.reward.stable_success_dwell_steps = 3

EXP_CFG.rl.curriculum.enabled = True
EXP_CFG.rl.curriculum.start_step = 1000 * 8
EXP_CFG.rl.curriculum.end_step = 9000 * 8
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.0

EXP_CFG.rl.observation.object_velocity_dim = 6
EXP_CFG.rl.observation.layout = [
    "object_cloud_flat",
    "tool_cloud_flat",
    "object_bbox_center",
    "tool_bbox_center",
    "hand_state",
    "robot_state",
    "previous_action",
    "relative_goal_pose",
    "object_velocity",
    "physics",
]

"""Bimanual Franka-tool RL on arbitrary target poses using diff-post encoder."""

from copy import deepcopy

from configs.experiments.fork_unstable_diff_post import EXP_CFG as _UNSTABLE_CFG


EXP_CFG = deepcopy(_UNSTABLE_CFG)

EXP_CFG.name = "bimanual_unstable_diff_post"
EXP_CFG.general.name = "bimanual_unstable_diff_post"
EXP_CFG.pretrain_reuse = "multitools_diff_post.py"

EXP_CFG.rl.name = "bimanual_unstable_diff_post"
EXP_CFG.rl.isaac_task_id = "tool-bimanual-unstable-v0"
EXP_CFG.rl.actor_critic_class = "ActorCriticTGBimanual"
EXP_CFG.rl.action.action_dim = 14
EXP_CFG.rl.launch.run_name = "bimanual_unstable_diff_post"
EXP_CFG.rl.launch.wandb_project = "bimanual"
EXP_CFG.rl.curriculum.enabled = True

EXP_CFG.model.policy_fusion.reuse_pretrain_pose_cross_attn = True
EXP_CFG.model.policy_fusion.cross_attn_heads = EXP_CFG.pretrain.cross_attn_heads
EXP_CFG.model.policy_fusion.cross_attn_layers = EXP_CFG.pretrain.cross_attn_layers
EXP_CFG.model.policy_fusion.sd_num_query = 16
EXP_CFG.model.policy_fusion.fusion_hidden_dims = [1024, 512, 256]
EXP_CFG.model.policy_fusion.actor_hidden_dims = [256, 128]
EXP_CFG.model.policy_fusion.critic_hidden_dims = [256, 128]

EXP_CFG.rl.observation.hand_state_dim = 18
EXP_CFG.rl.observation.robot_state_dim = 28
EXP_CFG.rl.observation.previous_action_dim = 14
EXP_CFG.rl.observation.bbox_center_dim = 9
EXP_CFG.rl.observation.object_velocity_dim = 6
EXP_CFG.rl.observation.layout = [
    "object_cloud_flat",
    "tool1_cloud_flat",
    "tool2_cloud_flat",
    "object_bbox_center",
    "tool1_bbox_center",
    "tool2_bbox_center",
    "hand1_state",
    "hand2_state",
    "robot1_state",
    "robot2_state",
    "previous_action",
    "relative_goal_pose",
    "object_velocity",
    "physics",
]

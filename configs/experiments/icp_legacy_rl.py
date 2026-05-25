"""RL-only legacy ICP policy on the bare Franka environment."""

from configs.config_exp import ExpCfg


EXP_CFG = ExpCfg(name="icp_legacy_rl")
EXP_CFG.general.name = "icp_legacy_rl"

EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False

EXP_CFG.model.name = "icp_legacy"
EXP_CFG.model.encoder_backend = "icp"
EXP_CFG.model.pretrained_encoder.name = "icp_legacy"
EXP_CFG.model.pretrained_encoder.schema = "icp_legacy"
EXP_CFG.model.pretrained_encoder.adapter = "icp_legacy"

EXP_CFG.rl.enabled = True
EXP_CFG.rl.actor_critic_class = "ActorCriticICP"
EXP_CFG.rl.encoder_checkpoint = "/mnt/home/zhengyixin/np_reproduce/512-32-balanced-SAM-wd-5e-05-920"
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.robot_mode = "bare_franka"
EXP_CFG.rl.isaac_task_id = "tool-sdf-v0"

EXP_CFG.rl.observation.include_object_cloud = True
EXP_CFG.rl.observation.include_tool_cloud = False
EXP_CFG.rl.observation.include_bbox_centers = False
EXP_CFG.rl.observation.layout = [
    "object_cloud_flat",
    "hand_state",
    "robot_state",
    "previous_action",
    "relative_goal_pose",
    "physics",
]

EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "icp_legacy"
EXP_CFG.rl.launch.run_name = "bare_franka"

EXP_CFG.num_gpus = 4
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024

EXP_CFG.rl.separate_actor_critic_fusion = True
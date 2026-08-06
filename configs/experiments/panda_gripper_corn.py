"""Official Panda-gripper RL using the legacy CORN/ICP encoder."""

from configs.panda_experiment_common import official_panda_diff_post_rl_cfg


CORN_ENCODER_CHECKPOINT = "/mnt/home/zhengyixin/np_reproduce/512-32-balanced-SAM-wd-5e-05-920"


EXP_CFG = official_panda_diff_post_rl_cfg("panda_gripper_corn")
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.name = "corn_legacy"
EXP_CFG.model.encoder_backend = "corn"
EXP_CFG.model.pretrained_encoder.name = "corn_legacy"
EXP_CFG.model.pretrained_encoder.schema = "icp_legacy"
EXP_CFG.model.pretrained_encoder.adapter = "icp_legacy"
EXP_CFG.model.pretrained_encoder.checkpoint_path = None
EXP_CFG.model.icp.checkpoint_path = None
EXP_CFG.rl.actor_critic_class = "ActorCriticICP"
EXP_CFG.rl.encoder_checkpoint = CORN_ENCODER_CHECKPOINT
EXP_CFG.rl.freeze_encoder = True
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
EXP_CFG.rl.separate_actor_critic_fusion = True

"""RL experiment using the published Point2Vec encoder checkpoint."""

from configs.config_exp import ExpCfg


POINT2VEC_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/model/"
    "pre_point2vec-epoch.799-step.64800.ckpt"
)

EXP_CFG = ExpCfg(name="point2vec")
EXP_CFG.general.name = "point2vec"
EXP_CFG.general.randomize_tool_assignment = True
EXP_CFG.general.randomize_object_assignment = True
EXP_CFG.model.name = "point2vec"

EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.rl.enabled = True

EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500

EXP_CFG.model.encoder_backend = "point2vec"
EXP_CFG.model.pretrained_encoder.name = "point2vec"
EXP_CFG.model.pretrained_encoder.checkpoint_path = POINT2VEC_CHECKPOINT
EXP_CFG.model.pretrained_encoder.schema = "point2vec"
EXP_CFG.model.pretrained_encoder.adapter = "point2vec_native"
EXP_CFG.rl.actor_critic_class = "ActorCriticPoint2Vec"
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.ppo.save_interval = 500
EXP_CFG.rl.ppo.entropy_coef = 0.006
EXP_CFG.num_gpus = 2
EXP_CFG.rl.launch.distributed = True

EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.logger = "wandb"

"""Default contact generation followed by SDF-only pretraining."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import SDF_CFG, clone_cfg


EXP_CFG = ExpCfg(name="fork_former")
EXP_CFG.general.name = "fork_former"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef/tools_fork.json"
EXP_CFG.model.name = "fork_former"

EXP_CFG.num_gpus = 4

EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False

EXP_CFG.model.pretrained_encoder.checkpoint_path = "/mnt/project/world_model/tool_generalist/model/encoder/tool_sdf_patch/best.pt"
EXP_CFG.model.tce.vit_depth = 4
EXP_CFG.rl.observation.model_input_centering = "object_center"
EXP_CFG.rl.actor_critic_class = "ActorCriticTG"

EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500

EXP_CFG.rl.enabled = True
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "fork"
EXP_CFG.rl.launch.run_name = "former"

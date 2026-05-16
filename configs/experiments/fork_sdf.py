"""Default contact generation followed by SDF-only pretraining."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import SDF_CFG, clone_cfg


EXP_CFG = ExpCfg(name="fork_sdf")
EXP_CFG.general.name = "fork_sdf"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef/tools_fork.json"
EXP_CFG.model.name = "tce_sdf_only"

EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.name = "contact_gen_default"
EXP_CFG.contact_gen.enabled = True
#EXP_CFG.contact_gen.regenerate = True

EXP_CFG.contact_gen.M = 4096
EXP_CFG.contact_gen.chunk_B = 256
EXP_CFG.contact_gen.B = 4096
EXP_CFG.contact_gen.visualization.enabled = False


EXP_CFG.pretrain = clone_cfg(SDF_CFG)
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "fork_pretrain"
EXP_CFG.pretrain.wandb_run_name = "sdf_only"

EXP_CFG.model.policy_fusion.reuse_pretrain_pose_cross_attn = True

EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500

EXP_CFG.rl.enabled = True
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 512
EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "fork"
EXP_CFG.rl.launch.run_name = "sdf_only"
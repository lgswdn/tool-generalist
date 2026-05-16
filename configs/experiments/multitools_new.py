"""Default contact generation followed by SDF-only pretraining."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import SDF_CFG, clone_cfg


EXP_CFG = ExpCfg(name="multitool_sdf")
EXP_CFG.general.name = "multitool_sdf"
EXP_CFG.paths_yaml = "paths_new.yaml"
EXP_CFG.model.name = "multitool_sdf"

EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.name = "contact_gen_multitool_new"
EXP_CFG.contact_gen.enabled = True
#EXP_CFG.contact_gen.regenerate = True

EXP_CFG.contact_gen.num_pairs = 10000
EXP_CFG.contact_gen.num_object_poses = 1
EXP_CFG.contact_gen.M = 4096
EXP_CFG.contact_gen.chunk_B = 256
EXP_CFG.contact_gen.B = 4096
EXP_CFG.contact_gen.physics.t_stabilize = 30
EXP_CFG.contact_gen.visualization.enabled = False
#EXP_CFG.contact_gen.visualization.stabilization_picture = True
#EXP_CFG.contact_gen.visualization.stabilization_picture_num = 8
#EXP_CFG.contact_gen.visualization.postcontact_video = True
#EXP_CFG.contact_gen.visualization.postcontact_video_num = 8

EXP_CFG.pretrain = clone_cfg(SDF_CFG)
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "multitools_pretrain"
EXP_CFG.pretrain.wandb_run_name = "sdf_only"
EXP_CFG.pretrain.epochs = 10
EXP_CFG.pretrain.optimizer.learning_rate = 5e-4


EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500

EXP_CFG.rl.enabled = True
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024

EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "all_tools"
EXP_CFG.rl.launch.run_name = "sdf_only"
EXP_CFG.rl.reward.object_goal_tracking_term_weight = 3
EXP_CFG.rl.reward.object_goal_tracking_fine_term_weight = 6
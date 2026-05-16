"""Default contact generation followed by SDF-only pretraining."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import SDF_CFG, clone_cfg


EXP_CFG = ExpCfg(name="fork_sdf")
EXP_CFG.general.name = "fork_sdf"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef/tools_selected.json"
EXP_CFG.model.name = "tce_sdf_only"

EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.name = "contact_gen_multitool"
EXP_CFG.contact_gen.enabled = False
#EXP_CFG.contact_gen.regenerate = True

EXP_CFG.contact_gen.num_pairs = 10000
EXP_CFG.contact_gen.num_object_poses = 1
EXP_CFG.contact_gen.M = 4096
EXP_CFG.contact_gen.chunk_B = 256
EXP_CFG.contact_gen.B = 4096
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
EXP_CFG.pretrain.epoch = 10
EXP_CFG.pretrain.optimizer.learning_rate = 3e-4
EXP_CFG.pretrain.checkpoint_policy.resume_checkpoint = "/mnt/project/world_model/tool_generalist/artifacts/encoder/fork_sdf/contact_gen_multitool/sdf_only_tce_sdf_only/09bf49adfdbb9b9c4ffb0034e7ba8e8c4cb6311d708050de0d136771b6db2d7f/best.pt"

EXP_CFG.pretrain.loss.sdf_relative_loss = True
EXP_CFG.pretrain.loss.sdf_relative_eps = 0.005

EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500

EXP_CFG.rl.enabled = True
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024

EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "all_tools"
EXP_CFG.rl.launch.run_name = "sdf_only_rellos_256_newreward"
EXP_CFG.rl.reward.object_goal_tracking_term_weight = 3
EXP_CFG.rl.reward.object_goal_tracking_fine_term_weight = 6
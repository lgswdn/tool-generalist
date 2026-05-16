"""Default contact generation followed by SDF-only pretraining."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import SDF_POST_CFG, clone_cfg


EXP_CFG = ExpCfg(name="fork_sdf")
EXP_CFG.general.name = "fork_sdf"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef/tools_fork.json"
EXP_CFG.model.name = "tce_sdf_only"

EXP_CFG.num_gpus = 1

EXP_CFG.contact_gen.name = "contact_gen_default"
EXP_CFG.contact_gen.num_pairs = 2
EXP_CFG.contact_gen.num_object_poses = 1
EXP_CFG.contact_gen.enabled = True
#EXP_CFG.contact_gen.regenerate = True

EXP_CFG.contact_gen.M = 4096
EXP_CFG.contact_gen.chunk_B = 256
EXP_CFG.contact_gen.B = 4096
EXP_CFG.contact_gen.visualization.enabled = False
#EXP_CFG.contact_gen.visualization.stabilization_picture = True
#EXP_CFG.contact_gen.visualization.stabilization_picture_num = 8
#EXP_CFG.contact_gen.visualization.postcontact_video = True
#EXP_CFG.contact_gen.visualization.postcontact_video_num = 8

EXP_CFG.pretrain = clone_cfg(SDF_POST_CFG)
EXP_CFG.pretrain.enabled = True

EXP_CFG.rl.enabled = True

EXP_CFG.rl.env.num_envs = 512


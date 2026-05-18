"""Panda-hand single-tool RL using the pretrained multitool SDF encoder."""

from copy import deepcopy

from configs.experiments.multitools_new import EXP_CFG as MULTITOOL_CFG


EXP_CFG = deepcopy(MULTITOOL_CFG)

EXP_CFG.name = "panda_hand_sdf"
EXP_CFG.general.name = "panda_hand_sdf"
EXP_CFG.paths_yaml = "paths_panda_hand.yaml"
EXP_CFG.model.name = "multitool_sdf"

EXP_CFG.pretrain_reuse = "multitools_diff.py"

EXP_CFG.rl.launch.wandb_project = "panda_hand"
EXP_CFG.rl.launch.run_name = "from_multitool_diff"

# EXP_CFG.num_gpus = 1
# EXP_CFG.rl.launch.distributed = False
# EXP_CFG.rl.env.num_envs = 1024

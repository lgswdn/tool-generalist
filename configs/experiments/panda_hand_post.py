"""Panda-hand single-tool RL using the pretrained multitool post encoder."""

from copy import deepcopy

from configs.experiments.multitools_post import EXP_CFG as MULTITOOL_POST_CFG


EXP_CFG = deepcopy(MULTITOOL_POST_CFG)

EXP_CFG.name = "panda_hand_post"
EXP_CFG.general.name = "panda_hand_post"
EXP_CFG.paths_yaml = "paths_panda_hand.yaml"
EXP_CFG.model.name = "multitool_post_only"

EXP_CFG.pretrain_reuse = "multitools_post.py"

EXP_CFG.num_gpus = 4
EXP_CFG.rl.name = "panda_hand_post"
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.wandb_project = "panda_hand"
EXP_CFG.rl.launch.run_name = "panda_hand_post"
EXP_CFG.rl.separate_actor_critic_fusion = True

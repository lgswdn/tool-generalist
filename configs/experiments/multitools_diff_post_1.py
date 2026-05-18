"""Ablation: split actor/critic fusion for multitools_diff_post RL."""

from copy import deepcopy

from configs.experiments.multitools_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_diff_post_1"
EXP_CFG.pretrain_reuse = "multitools_diff_post.py"

EXP_CFG.num_gpus = 4
EXP_CFG.rl.name = "multitools_diff_post_1"
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.run_name = "multitools_diff_post_1"
EXP_CFG.rl.separate_actor_critic_fusion = True

"""Ablation 1: split actor/critic fusion for multitools_diff RL."""

from copy import deepcopy

from configs.experiments.multitools_diff import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_diff_1"
EXP_CFG.pretrain_reuse = "multitools_diff.py"

EXP_CFG.num_gpus = 8
EXP_CFG.rl.name = "multitools_diff_1"
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.run_name = "multitools_diff_1"
EXP_CFG.rl.separate_actor_critic_fusion = True
EXP_CFG.rl.ppo.entropy_coef = 0.006
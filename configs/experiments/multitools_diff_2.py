"""Ablation 2: reduce task success reward for multitools_diff RL."""

from copy import deepcopy

from configs.experiments.multitools_diff import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_diff_2"
EXP_CFG.pretrain_reuse = "multitools_diff.py"

EXP_CFG.num_gpus = 4
EXP_CFG.rl.name = "multitools_diff_2"
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.run_name = "multitools_diff_2"
EXP_CFG.rl.reward.task_success_term_weight = 500.0

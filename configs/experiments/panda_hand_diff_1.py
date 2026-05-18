"""Ablation: split actor/critic fusion for panda_hand_diff RL."""

from copy import deepcopy

from configs.experiments.panda_hand_diff import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "panda_hand_diff_1"
EXP_CFG.general.name = "panda_hand_diff"
EXP_CFG.pretrain_reuse = "panda_hand_diff.py"

EXP_CFG.num_gpus = 4
EXP_CFG.rl.name = "panda_hand_diff_1"
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.run_name = "panda_hand_diff_1"
EXP_CFG.rl.separate_actor_critic_fusion = True

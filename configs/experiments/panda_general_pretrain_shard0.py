"""Shard 0/2 for generated-gripper contact generation."""

from copy import deepcopy

from configs.experiments.panda_general_pretrain import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "panda_general_pretrain_shard0"
EXP_CFG.contact_gen.shard_count = 2
EXP_CFG.contact_gen.shard_index = 0

EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.enabled = False

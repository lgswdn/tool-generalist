"""Ablation: split actor/critic fusion for multitools_sdf RL."""

from copy import deepcopy

from configs.experiments.multitools_sdf import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_sdf_1"
EXP_CFG.general.name = "multitools_sdf"
EXP_CFG.pretrain_reuse = "multitools_sdf.py"

EXP_CFG.num_gpus = 8
EXP_CFG.rl.name = "multitools_sdf_1"
EXP_CFG.rl.encoder_checkpoint = _BASE_EXP_CFG.pretrain.checkpoint_policy.resume_checkpoint
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.run_name = "multitools_sdf_1"
EXP_CFG.rl.separate_actor_critic_fusion = True

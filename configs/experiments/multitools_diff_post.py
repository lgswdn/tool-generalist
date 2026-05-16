"""Multitool diff + postcontact movement pretraining."""

from copy import deepcopy

from configs.experiments.multitools_diff import EXP_CFG as DIFF_CFG


EXP_CFG = deepcopy(DIFF_CFG)

EXP_CFG.name = "multitools_diff_post"
EXP_CFG.general.name = "multitools_diff_post"
EXP_CFG.model.name = "multitool_diff_post"

EXP_CFG.pretrain.name = "diff_post"
EXP_CFG.pretrain.enabled_heads = ["diff", "postcontact"]
EXP_CFG.pretrain.tasks.sdf = False
EXP_CFG.pretrain.tasks.diffusion = True
EXP_CFG.pretrain.tasks.postcontact = True
EXP_CFG.pretrain.wandb_run_name = "diff_post"
EXP_CFG.pretrain.condition_normalization = True
EXP_CFG.pretrain.condition_norm_sample_files = 64

EXP_CFG.rl.launch.run_name = "diff_post"

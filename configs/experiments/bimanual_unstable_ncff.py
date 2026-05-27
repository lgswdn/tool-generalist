"""Bimanual unstable no-curriculum RL with full tool and object sets."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_full_tool_diff_post_no_curriculum import (
    EXP_CFG as _BASE_EXP_CFG,
)


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_unstable_ncff"
EXP_CFG.general.name = "bimanual_unstable_ncff"
EXP_CFG.rl.name = "bimanual_unstable_ncff"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_ncff"

EXP_CFG.model.policy_fusion.cross_attn_heads = 2
EXP_CFG.rl.curriculum.enabled = False

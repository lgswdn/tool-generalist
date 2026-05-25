"""Bimanual full-tool diff+post RL without stable-pose curriculum."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_full_tool_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_unstable_full_tool_diff_post_no_curriculum"
EXP_CFG.general.name = "bimanual_unstable_full_tool_diff_post_no_curriculum"
EXP_CFG.rl.name = "bimanual_unstable_full_tool_diff_post_no_curriculum"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_full_tool_diff_post_no_curriculum"

EXP_CFG.rl.curriculum.enabled = False

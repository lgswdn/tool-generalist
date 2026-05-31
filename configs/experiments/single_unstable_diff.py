"""Single-arm unstable-pose RL over the full selected-tool set using diff-only encoder."""

from copy import deepcopy

from configs.experiments.single_unstable_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "single_unstable_diff"
EXP_CFG.general.name = "single_unstable_diff"
EXP_CFG.model.name = "multitool_diff_only"
EXP_CFG.pretrain_reuse = "multitools_diff.py"

EXP_CFG.rl.name = "single_unstable_diff"
EXP_CFG.rl.launch.run_name = "single_unstable_diff"

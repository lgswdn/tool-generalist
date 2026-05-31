"""Single-arm unstable-pose RL over the full selected-tool set using post-only encoder."""

from copy import deepcopy

from configs.experiments.single_unstable_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "single_unstable_post"
EXP_CFG.general.name = "single_unstable_post"
EXP_CFG.model.name = "multitool_post_only"
EXP_CFG.pretrain_reuse = "multitools_post.py"

EXP_CFG.rl.name = "single_unstable_post"
EXP_CFG.rl.launch.run_name = "single_unstable_post"

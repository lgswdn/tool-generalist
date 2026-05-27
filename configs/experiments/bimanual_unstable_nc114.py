"""Fork/foam-brick no-curriculum RL with pose-threshold reward."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_nc112 import (
    EXP_CFG as _BASE_EXP_CFG,
)


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_unstable_nc114"
EXP_CFG.general.name = "bimanual_unstable_nc114"
EXP_CFG.rl.name = "bimanual_unstable_nc114"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_nc114"

EXP_CFG.model.policy_fusion.cross_attn_heads = 4
EXP_CFG.rl.reward.object_goal_threshold_term_weight = 6.0
EXP_CFG.rl.reward.stable_success_dwell_steps = 10

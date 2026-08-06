"""DPOC RL on all 5,676 full-YES objects for 5,000 iterations."""

from configs.panda_comparison_common import (
    configure_full_yes_comparison,
)
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_full_yes_5k")
EXP_CFG.pretrain_reuse = "panda_general_pretrain.py"
configure_full_yes_comparison(EXP_CFG)

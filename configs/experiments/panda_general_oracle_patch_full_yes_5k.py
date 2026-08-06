"""Oracle patch-distance RL on all full-YES objects for 5,000 iterations."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_oracle_patch_rl_cfg


EXP_CFG = generated_gripper_oracle_patch_rl_cfg(
    "panda_general_oracle_patch_full_yes_5k"
)
configure_full_yes_comparison(EXP_CFG)

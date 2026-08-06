"""Full-YES 5k RL using the depth-1 intersecting-geometry encoder."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_intersecting_depth1_unicorn_rl_cfg,
)


EXP_CFG = generated_gripper_intersecting_depth1_unicorn_rl_cfg(
    "panda_general_unicorn_ours_intersecting_depth1_full_yes_5k"
)
configure_full_yes_comparison(EXP_CFG)

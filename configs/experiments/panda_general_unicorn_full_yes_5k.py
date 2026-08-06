"""Vanilla UniCORN RL on full-YES for the symmetric 5,000-step comparison."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_full_yes_5k",
    ours_tce=False,
)
# Uses the authors' released frozen UniCORN representation checkpoint.
configure_full_yes_comparison(EXP_CFG)

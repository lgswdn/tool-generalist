"""Concavity-biased contacts with minimum signed-SDF regression and 10k DGN."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_concavity_sdf_regression_rl_cfg,
)


EXP_CFG = parallel_concavity_sdf_regression_rl_cfg(
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_"
    "concavity_biased_sdf_dgn_10k"
)
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 10_000

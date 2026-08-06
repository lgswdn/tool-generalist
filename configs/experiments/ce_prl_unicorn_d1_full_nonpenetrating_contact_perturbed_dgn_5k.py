"""Nonpenetrating contacts with paper-scale perturbation, then 5k DGN RL."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_paper_contact_quality_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_perturbed_dgn_5k",
    contact_variant="nonpenetrating_contact_perturbed",
    point_jitter_std=0.001,
    contact_eps=0.002,
    dgn_iterations=5_000,
    perturb_nonpenetrating=True,
    nonpenetrating_penetration_eps=5e-4,
)
configure_full_yes_comparison(EXP_CFG)

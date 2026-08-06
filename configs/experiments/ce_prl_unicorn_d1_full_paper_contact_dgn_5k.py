"""Paper tangent contacts, cached 2 mm SDF labels, then DGN RL for 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_paper_contact_quality_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d1_full_paper_contact_dgn_5k",
    contact_variant="paper_contact",
    point_jitter_std=0.001,
    contact_eps=0.002,
    dgn_iterations=5_000,
)
configure_full_yes_comparison(EXP_CFG)

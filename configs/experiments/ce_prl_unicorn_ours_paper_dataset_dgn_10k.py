"""Proven nonpenetrating recipe trained on the 1M paper-contact dataset."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_proven_nonpenetrating_recipe_paper_dataset_rl_cfg,
)


EXP_CFG = parallel_proven_nonpenetrating_recipe_paper_dataset_rl_cfg(
    "ce_prl_unicorn_ours_paper_dataset_dgn_10k"
)
configure_dgn_10k_comparison(EXP_CFG)

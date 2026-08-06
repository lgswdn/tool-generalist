"""Parallel-only depth-1 UniCORN-ours on raw penetrating contacts and DGN."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_unicorn_rl_cfg,
)


EXP_CFG = parallel_depth1_full_attention_unicorn_rl_cfg(
    "ce_prl_unicorn_ours_raw_dgn_10k",
    raw_contact=True,
)
configure_dgn_10k_comparison(EXP_CFG)

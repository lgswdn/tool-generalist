"""Parallel-only depth-1 full-attention UniCORN on non-penetrating contacts."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg,
)


EXP_CFG = parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(
    "ce_prl_unicorn_ours_nonpenetrating_dgn_10k"
)
configure_dgn_10k_comparison(EXP_CFG)

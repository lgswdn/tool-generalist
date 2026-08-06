"""Parallel-only depth-1 diffusion-only pretrain followed by DGN RL."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_diff_rl_cfg,
)


EXP_CFG = parallel_depth1_full_attention_diff_rl_cfg(
    "ce_prl_diff_dgn_10k",
)
configure_dgn_10k_comparison(EXP_CFG)

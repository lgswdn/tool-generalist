"""Parallel-only depth-1 cross-only UniCORN on raw contacts and DGN."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_unicorn_rl_cfg,
)


EXP_CFG = parallel_depth1_full_attention_unicorn_rl_cfg(
    "ce_prl_unicorn_ours_raw_cross_dgn_10k",
    raw_contact=True,
)
# This value is consumed by both fresh encoder pretraining and RL. The
# explicit-attention checkpoint contract rejects missing or mismatched modes.
EXP_CFG.model.tce.vit_attention_mode = "cross_only"
configure_dgn_10k_comparison(EXP_CFG)

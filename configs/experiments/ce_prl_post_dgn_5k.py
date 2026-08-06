"""Newest-200 parallel grippers with postcontact-only pretrain and DGN RL."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_post_rl_cfg,
)


EXP_CFG = parallel_depth1_full_attention_post_rl_cfg(
    "ce_prl_post_dgn_5k"
)
configure_full_yes_comparison(EXP_CFG)

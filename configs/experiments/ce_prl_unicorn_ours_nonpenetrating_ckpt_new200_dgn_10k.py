"""RL-only transfer of the proven nonpenetrating encoder to the new 200 tools."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_new200_proven_nonpenetrating_encoder_rl_cfg,
)


EXP_CFG = parallel_new200_proven_nonpenetrating_encoder_rl_cfg(
    "ce_prl_unicorn_ours_nonpenetrating_ckpt_new200_dgn_10k"
)
configure_dgn_10k_comparison(EXP_CFG)

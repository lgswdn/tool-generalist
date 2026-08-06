"""Legacy nonpenetrating pretrain recipe and DGN RL on the newest 200 tools."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import (
    parallel_new200_proven_nonpenetrating_recipe_rl_cfg,
)


EXP_CFG = parallel_new200_proven_nonpenetrating_recipe_rl_cfg(
    "ce_prl_unicorn_ours_nonpenetrating_new200_dgn_10k"
)
configure_dgn_10k_comparison(EXP_CFG)

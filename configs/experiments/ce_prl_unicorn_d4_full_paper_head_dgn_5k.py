"""Four-layer paper-head pretraining, then 5k DGN RL."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_paper_contact_quality_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d4_full_paper_head_dgn_5k",
    contact_variant="paper_head",
    transformer_depth=4,
    dgn_iterations=5_000,
)
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000

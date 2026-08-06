"""Combined grippers: D4 ordinary encoder, global-concavity contacts, DGN 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import general_d4_full_rl_cfg


EXP_CFG = general_d4_full_rl_cfg(
    "ce_general_d4_full_concavity_global_raw_dgn_5k",
    contact_quality="concavity_global",
    architecture="raw",
)
configure_full_yes_comparison(EXP_CFG)

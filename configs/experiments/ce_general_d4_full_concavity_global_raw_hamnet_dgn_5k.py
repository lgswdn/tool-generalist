"""Combined grippers: frozen concavity/raw D4 encoder with HAMNet policy."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import general_d4_hamnet_rl_cfg


EXP_CFG = general_d4_hamnet_rl_cfg(
    "ce_general_d4_full_concavity_global_raw_hamnet_dgn_5k"
)
configure_full_yes_comparison(EXP_CFG)

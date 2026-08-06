"""Newest-200 normalized post-pretrained direct PointNet trainable in DGN5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_native_pointnet_normalized_postcontact_rl_cfg,
)


PRETRAIN_SOURCE = "ce_prl_native_pointnet_normalized_post_pretrain.py"
NAME = "ce_prl_native_pointnet_normalized_post_unfrozen_dgn_5k"

EXP_CFG = parallel_native_pointnet_normalized_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain_reuse = PRETRAIN_SOURCE
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.freeze_encoder = False
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000

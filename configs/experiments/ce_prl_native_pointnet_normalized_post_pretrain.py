"""Shared normalized direct-128 PointNet post-only pretraining stage."""

from configs.panda_experiment_common import (
    parallel_native_pointnet_normalized_postcontact_rl_cfg,
)


NAME = "ce_prl_native_pointnet_normalized_post_pretrain"

EXP_CFG = parallel_native_pointnet_normalized_postcontact_rl_cfg(NAME)
EXP_CFG.rl.enabled = False

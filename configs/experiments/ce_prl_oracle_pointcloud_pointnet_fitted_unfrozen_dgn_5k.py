"""Newest-200 parallel grippers with the original fitted PointNet trainable."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_fitted_oracle_pointcloud_pointnet_rl_cfg,
)


EXP_CFG = parallel_fitted_oracle_pointcloud_pointnet_rl_cfg(
    "ce_prl_oracle_pointcloud_pointnet_fitted_unfrozen_dgn_5k"
)
configure_full_yes_comparison(EXP_CFG)

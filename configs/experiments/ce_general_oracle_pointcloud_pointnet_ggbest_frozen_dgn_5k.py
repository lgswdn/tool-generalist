"""DGN5k with the best GG-adapted oracle PointNet frozen."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    GG_BEST_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
    general_frozen_gg_oracle_pointcloud_pointnet_rl_cfg,
)


GG_BEST_CHECKPOINT = GG_BEST_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT


EXP_CFG = general_frozen_gg_oracle_pointcloud_pointnet_rl_cfg(
    "ce_general_oracle_pointcloud_pointnet_ggbest_frozen_dgn_5k",
    checkpoint_path=GG_BEST_CHECKPOINT,
)
configure_full_yes_comparison(EXP_CFG)

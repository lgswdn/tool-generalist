"""Depth-12 TCE shape contract for combined-gripper PCA extraction."""

from configs.panda_experiment_common import general_d4_full_rl_cfg


EXP_CFG = general_d4_full_rl_cfg(
    "ce_general_oracle_rebuild_d12_pca_source",
    contact_quality="paper",
    architecture="raw",
)
EXP_CFG.model.tce.vit_depth = 12

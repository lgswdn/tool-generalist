"""Depth-12 TCE shape contract for canonical-cloud PCA extraction."""

from configs.panda_experiment_common import (
    parallel_paper_contact_quality_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_oracle_rebuild_d12_pca_source",
    contact_variant="paper_head",
    transformer_depth=12,
    dgn_iterations=5_000,
)

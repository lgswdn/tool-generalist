"""Paper contact data with the 1 mm/no-scale augmentation ablation."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_paper_contact_quality_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d1_full_paper_contact_1mm_no_scale_dgn_5k",
    contact_variant="paper_contact",
    point_jitter_std=0.001,
    contact_eps=0.002,
    dgn_iterations=5_000,
)
# Match the non-penetrating augmentation ablation while retaining the paper
# contact geometry and exact convex-union point labels.
EXP_CFG.pretrain.unicorn.augment.log_scale_range = (0.0, 0.0)
EXP_CFG.pretrain.unicorn.augment.noise_std = 0.001
configure_full_yes_comparison(EXP_CFG)

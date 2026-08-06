"""Nonpenetrating contact geometry with 1M cases and matched 1 mm augmentation."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg,
)


EXP_CFG = parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg(
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_1m_1mm_no_scale_dgn_5k",
    contact_variant="nonpenetrating_contact",
)
configure_full_yes_comparison(EXP_CFG)

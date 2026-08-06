"""GG 15k continuation from the D4 paper-contact kinematic 5k parent."""

from configs.ce_prl_d4_gg_common import d4_gg_continuation_cfg


EXP_CFG = d4_gg_continuation_cfg(
    "ce_prl_unicorn_d4_full_paper_contact_kinematic_gg_15k",
    contact_variant="paper_contact",
    architecture="kinematic",
)

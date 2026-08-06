"""GG 15k continuation of the combined paper/kinematic DGN parent."""

from configs.ce_general_d4_gg_common import general_d4_gg_continuation_cfg


EXP_CFG = general_d4_gg_continuation_cfg(
    "ce_general_d4_full_paper_kinematic_gg_15k",
    contact_quality="paper",
    architecture="kinematic",
)

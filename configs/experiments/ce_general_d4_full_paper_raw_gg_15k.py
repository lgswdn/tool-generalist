"""GG 15k continuation of the combined paper/raw DGN parent."""

from configs.ce_general_d4_gg_common import general_d4_gg_continuation_cfg


EXP_CFG = general_d4_gg_continuation_cfg(
    "ce_general_d4_full_paper_raw_gg_15k",
    contact_quality="paper",
    architecture="raw",
)

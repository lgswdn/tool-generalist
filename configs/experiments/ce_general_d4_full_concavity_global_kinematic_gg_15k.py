"""GG 15k continuation of the combined concavity/kinematic DGN parent."""

from configs.ce_general_d4_gg_common import general_d4_gg_continuation_cfg


EXP_CFG = general_d4_gg_continuation_cfg(
    "ce_general_d4_full_concavity_global_kinematic_gg_15k",
    contact_quality="concavity_global",
    architecture="kinematic",
)

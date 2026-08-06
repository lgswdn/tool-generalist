"""Oracle patch-distance GG 15k initialized from its completed full-YES run."""

from configs.panda_comparison_common import completed_parent_checkpoint, configure_gg_comparison
from configs.panda_experiment_common import generated_gripper_oracle_patch_rl_cfg


PARENT_EXPERIMENT = "panda_general_oracle_patch_full_yes_5k"

EXP_CFG = generated_gripper_oracle_patch_rl_cfg(
    "panda_general_oracle_patch_gg_from_full_yes_5k"
)
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    encoder_family="oracle_patch",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

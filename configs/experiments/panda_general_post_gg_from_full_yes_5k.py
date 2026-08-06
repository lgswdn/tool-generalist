"""Post-only DPOC GG fine-tuning initialized by its full-YES 5k run."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
    configure_post_contact_reuse,
)
from configs.panda_experiment_common import generated_gripper_post_rl_cfg


PARENT_EXPERIMENT = "panda_general_post_full_yes_5k"


EXP_CFG = generated_gripper_post_rl_cfg(
    "panda_general_post_gg_from_full_yes_5k"
)
EXP_CFG.pretrain_reuse = "panda_general_post_full_yes_5k.py"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
)
EXP_CFG.rl.resume_checkpoint = None
configure_post_contact_reuse(EXP_CFG)
configure_gg_comparison(EXP_CFG)

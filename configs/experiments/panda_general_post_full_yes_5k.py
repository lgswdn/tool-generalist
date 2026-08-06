"""Post-only DPOC RL on full-YES for 5,000 iterations."""

from configs.panda_comparison_common import (
    configure_full_yes_comparison,
    configure_post_contact_reuse,
)
from configs.panda_experiment_common import generated_gripper_post_rl_cfg


EXP_CFG = generated_gripper_post_rl_cfg(
    "panda_general_post_full_yes_5k"
)
EXP_CFG.pretrain_reuse = None
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain.wandb_project = "ungrasp"
EXP_CFG.pretrain.wandb_run_name = "panda_general_post_pretrain"
configure_post_contact_reuse(EXP_CFG)
configure_full_yes_comparison(EXP_CFG)

"""Unicorn-ours GG 15k fine-tuning initialized by its completed full-YES 5k run."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


PARENT_EXPERIMENT = "panda_general_unicorn_ours_full_yes_5k"

EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_gg_from_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = "unicorn_pretrain_ours_generated_gripper.py"
EXP_CFG.model.pretrained_encoder.checkpoint_path = None
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(PARENT_EXPERIMENT)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

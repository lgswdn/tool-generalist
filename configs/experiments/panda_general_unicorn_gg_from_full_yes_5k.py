"""Vanilla UniCORN GG 15k fine-tuning from its completed full-YES 5k run."""

from configs.panda_comparison_common import completed_parent_checkpoint, configure_gg_comparison
from configs.panda_experiment_common import (
    OFFICIAL_UNICORN_REPRESENTATION_CHECKPOINT,
    generated_gripper_unicorn_rl_cfg,
)


PARENT_EXPERIMENT = "panda_general_unicorn_full_yes_5k"

EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_gg_from_full_yes_5k",
    ours_tce=False,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    encoder_family="UniCORN",
    expected_pretrained_encoder_checkpoint=(
        OFFICIAL_UNICORN_REPRESENTATION_CHECKPOINT
    ),
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

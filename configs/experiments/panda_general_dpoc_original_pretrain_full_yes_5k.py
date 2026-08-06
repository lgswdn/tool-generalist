"""July-1 DPOC encoder RL on full-YES objects for 5,000 iterations."""

from configs.panda_comparison_common import (
    ORIGINAL_DPOC_ENCODER_CHECKPOINT,
    configure_full_yes_comparison,
)
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


EXP_CFG = generated_gripper_diff_post_rl_cfg(
    "panda_general_dpoc_original_pretrain_full_yes_5k"
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = ORIGINAL_DPOC_ENCODER_CHECKPOINT
configure_full_yes_comparison(EXP_CFG)

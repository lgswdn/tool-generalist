"""July-1 DPOC encoder GG fine-tuning initialized by its own full-YES 5k run."""

from configs.panda_comparison_common import (
    ORIGINAL_DPOC_ENCODER_CHECKPOINT,
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


PARENT_EXPERIMENT = "panda_general_dpoc_original_pretrain_full_yes_5k"


EXP_CFG = generated_gripper_diff_post_rl_cfg(
    "panda_general_dpoc_original_pretrain_gg_from_full_yes_5k"
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = ORIGINAL_DPOC_ENCODER_CHECKPOINT
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

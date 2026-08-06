"""DPOC GG 15k fine-tuning initialized by its completed full-YES 5k run."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


PARENT_EXPERIMENT = "panda_general_dpoc_full_yes_5k"
DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "002002d13240f5618b67ce62952e1e18c95cba23269c2ee2f628e8e381ff74d9/"
    "best.pt"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg(
    "panda_general_dpoc_gg_from_full_yes_5k"
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = DPOC_ENCODER_CHECKPOINT
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(PARENT_EXPERIMENT)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

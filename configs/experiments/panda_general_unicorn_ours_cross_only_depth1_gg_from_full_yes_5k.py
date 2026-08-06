"""GG 15k transfer from the completed depth-1 cross-only full-YES run."""

from configs.panda_comparison_common import completed_parent_checkpoint, configure_gg_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


PARENT_EXPERIMENT = "panda_general_unicorn_ours_cross_only_depth1_full_yes_5k"

EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_cross_only_depth1_gg_from_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = "unicorn_pretrain_ours_cross_only_depth1.py"
EXP_CFG.model.pretrained_encoder.checkpoint_path = None
EXP_CFG.model.tce.vit_depth = 1
EXP_CFG.model.tce.vit_attention_mode = "cross_only"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

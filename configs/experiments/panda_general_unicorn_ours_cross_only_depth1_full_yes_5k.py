"""Full-YES 5k RL using the depth-1 cross-only UniCORN-ours encoder."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_cross_only_depth1_full_yes_5k",
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
configure_full_yes_comparison(EXP_CFG)

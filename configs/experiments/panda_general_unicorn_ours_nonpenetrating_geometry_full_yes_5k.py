"""Full-YES 5k RL using the non-penetrating-geometry UniCORN-ours encoder."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_nonpenetrating_geometry_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = "unicorn_pretrain_ours_nonpenetrating_geometry.py"
EXP_CFG.model.pretrained_encoder.checkpoint_path = None
configure_full_yes_comparison(EXP_CFG)

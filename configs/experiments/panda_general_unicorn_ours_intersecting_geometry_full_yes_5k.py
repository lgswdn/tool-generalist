"""Full-YES 5k RL using the intersection-trained UniCORN-ours encoder."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    INTERSECTING_GEOMETRY_PRETRAIN_CHECKPOINT,
    generated_gripper_unicorn_rl_cfg,
)


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_intersecting_geometry_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = INTERSECTING_GEOMETRY_PRETRAIN_CHECKPOINT
configure_full_yes_comparison(EXP_CFG)

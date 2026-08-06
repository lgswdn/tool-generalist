"""Matched CE DGN-10k RL initialized from raw/intersecting contact pretrain."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import ce_unicorn_ours_contact_rl_cfg


RAW_CONTACT_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "ce_unicorn_ours_raw_contact_full_yes_5k/contact_gen_ce_raw_contact/"
    "unicorn_ce_raw_contact_unicorn_ce_raw_contact/"
    "2fffbb6e3d7f93946b56ed5eb40371985b193a3cd17b815065f38db098876085/"
    "best.pt"
)


EXP_CFG = ce_unicorn_ours_contact_rl_cfg(
    "ce_unicorn_ours_raw_contact_dgn_10k",
    allow_penetration=True,
)
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = RAW_CONTACT_PRETRAIN_CHECKPOINT
configure_dgn_10k_comparison(EXP_CFG)

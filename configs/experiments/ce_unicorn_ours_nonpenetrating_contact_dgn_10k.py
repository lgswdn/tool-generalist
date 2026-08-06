"""Matched CE DGN-10k RL initialized from non-penetrating contact pretrain."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import ce_unicorn_ours_contact_rl_cfg


NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "ce_unicorn_ours_nonpenetrating_contact_full_yes_5k/"
    "contact_gen_ce_nonpenetrating_contact/"
    "unicorn_ce_nonpenetrating_contact_unicorn_ce_nonpenetrating_contact/"
    "0d35aaf102186c7a913c239ca4d30c91c737acb86261304fc97993305e8a98f3/"
    "best.pt"
)


EXP_CFG = ce_unicorn_ours_contact_rl_cfg(
    "ce_unicorn_ours_nonpenetrating_contact_dgn_10k",
    allow_penetration=False,
)
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = (
    NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT
)
configure_dgn_10k_comparison(EXP_CFG)

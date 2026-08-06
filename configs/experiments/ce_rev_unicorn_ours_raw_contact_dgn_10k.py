"""Revolute-only DGN-10k RL initialized from raw-contact pretraining."""

from configs.experiments.ce_unicorn_ours_raw_contact_dgn_10k import (
    RAW_CONTACT_PRETRAIN_CHECKPOINT,
)
from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import generated_revolute_unicorn_rl_cfg


EXP_CFG = generated_revolute_unicorn_rl_cfg(
    "ce_rev_unicorn_ours_raw_contact_dgn_10k",
    ours_tce=True,
)
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = RAW_CONTACT_PRETRAIN_CHECKPOINT
configure_dgn_10k_comparison(EXP_CFG)

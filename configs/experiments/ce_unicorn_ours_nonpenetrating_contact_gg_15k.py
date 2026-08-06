"""CE nonpenetrating-contact GG 15k transfer from its completed DGN 10k run."""

from configs.experiments.ce_unicorn_ours_nonpenetrating_contact_dgn_10k import (
    NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT,
)
from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import ce_unicorn_ours_contact_rl_cfg


PARENT_EXPERIMENT = "ce_unicorn_ours_nonpenetrating_contact_dgn_10k"


EXP_CFG = ce_unicorn_ours_contact_rl_cfg(
    "ce_unicorn_ours_nonpenetrating_contact_gg_15k",
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
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    expected_paths_yaml="configs/paths/ce_contact_pretrain.yaml",
    expected_max_iterations=10000,
    expected_num_gpus=8,
    expected_pretrained_encoder_checkpoint=(
        NONPENETRATING_CONTACT_PRETRAIN_CHECKPOINT
    ),
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

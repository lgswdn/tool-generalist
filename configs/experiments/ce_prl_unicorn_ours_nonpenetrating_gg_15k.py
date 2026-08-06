"""Parallel-only non-penetrating GG 15k transfer from matching DGN 10k."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


PARENT_EXPERIMENT = "ce_prl_unicorn_ours_nonpenetrating_dgn_10k"


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "ce_prl_unicorn_ours_nonpenetrating_gg_15k",
    ours_tce=True,
)
EXP_CFG.paths_yaml = "configs/paths/generated_gripper_contact.yaml"
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = (
    "ce_prl_unicorn_ours_nonpenetrating_dgn_10k.py"
)
EXP_CFG.model.pretrained_encoder.checkpoint_path = None
EXP_CFG.model.tce.vit_depth = 1
EXP_CFG.model.tce.vit_attention_mode = "joint_self"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    expected_paths_yaml="configs/paths/generated_gripper_contact.yaml",
    expected_max_iterations=10000,
    expected_num_gpus=8,
    expected_vit_attention_contract="explicit_v1",
    expected_vit_attention_mode="joint_self",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

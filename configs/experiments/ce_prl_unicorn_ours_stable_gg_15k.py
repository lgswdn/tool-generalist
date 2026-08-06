"""Parallel-only stable-contact GG 15k transfer from matching DGN 10k."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    parallel_depth1_full_attention_unicorn_rl_cfg,
)


PARENT_EXPERIMENT = "ce_prl_unicorn_ours_stable_dgn_10k"


EXP_CFG = parallel_depth1_full_attention_unicorn_rl_cfg(
    "ce_prl_unicorn_ours_stable_gg_15k",
    raw_contact=False,
)
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = "ce_prl_unicorn_ours_stable_dgn_10k.py"
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

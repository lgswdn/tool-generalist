"""GG 15k continuation from the exact-budget raw-contact DGN parent."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    parallel_paper_contact_quality_rl_cfg,
)


PARENT_EXPERIMENT = "ce_prl_unicorn_d1_full_raw_contact_dgn_10k"

EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d1_full_raw_contact_gg_15k",
    contact_variant="raw_contact",
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = f"{PARENT_EXPERIMENT}.py"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="contact_gen_prl_raw_contact_500k",
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=10000,
    expected_num_gpus=8,
    expected_vit_attention_contract="explicit_v1",
    expected_vit_attention_mode="joint_self",
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

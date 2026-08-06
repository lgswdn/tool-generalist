"""GG 15k transfer from the matching 1 mm/no-scale DGN parent."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    parallel_paper_contact_quality_rl_cfg,
)


PARENT_EXPERIMENT = (
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_1mm_no_scale_dgn_5k"
)

EXP_CFG = parallel_paper_contact_quality_rl_cfg(
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_1mm_no_scale_gg_15k",
    contact_variant="nonpenetrating_contact",
    point_jitter_std=0.001,
    contact_eps=0.002,
    dgn_iterations=5_000,
    perturb_nonpenetrating=False,
    nonpenetrating_penetration_eps=5e-4,
)
EXP_CFG.pretrain.unicorn.augment.log_scale_range = (0.0, 0.0)
EXP_CFG.pretrain.unicorn.augment.noise_std = 0.001
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = f"{PARENT_EXPERIMENT}.py"
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="contact_gen_prl_nonpenetrating_contact_500k",
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=5000,
    expected_num_gpus=8,
    expected_vit_attention_contract="explicit_v1",
    expected_vit_attention_mode="joint_self",
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

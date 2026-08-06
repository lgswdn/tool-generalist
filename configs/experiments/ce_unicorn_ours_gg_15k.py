"""CE UniCORN-ours GG 15k transfer from its completed DGN 10k run."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import cross_embodiment_gripper_unicorn_rl_cfg


PARENT_EXPERIMENT = "ce_unicorn_ours_dgn_10k"
UNICORN_OURS_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)


EXP_CFG = cross_embodiment_gripper_unicorn_rl_cfg(
    "ce_unicorn_ours_gg_15k",
    ours_tce=True,
)
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_PRETRAIN_CHECKPOINT
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    expected_paths_yaml="configs/paths/cross_embodiment_generated_revolute.yaml",
    expected_max_iterations=10000,
    expected_num_gpus=8,
    expected_pretrained_encoder_checkpoint=UNICORN_OURS_PRETRAIN_CHECKPOINT,
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

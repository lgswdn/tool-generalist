"""Rank-16 bottleneck GG 15k transfer from its completed full-YES parent."""

from configs.panda_comparison_common import (
    ORIGINAL_GRIPPER_MANIFEST_RESTORED_AT_UTC,
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


PARENT_EXPERIMENT = "panda_general_unicorn_ours_encoder_bottleneck_rank16_full_yes_5k"
UNICORN_OURS_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)
ENCODER_TOKEN_PCA = (
    "/mnt/home/zhengyixin/tool-generalist/artifacts/projections/"
    "unicorn_ours_encoder_pre_mlp_pca.pt"
)


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_encoder_bottleneck_rank16_gg_from_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_PRETRAIN_CHECKPOINT
EXP_CFG.model.tce.rl_token_source = "encoder"
EXP_CFG.model.tce.encoder_token_pca_rank = EXP_CFG.model.tce.encoder_channel
EXP_CFG.model.tce.encoder_token_pca_path = None
EXP_CFG.model.tce.encoder_token_bottleneck_rank = 16
EXP_CFG.model.tce.encoder_token_bottleneck_pca_path = ENCODER_TOKEN_PCA
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT_EXPERIMENT,
    contact_name="no-contact",
    expected_bottleneck_rank=16,
    created_at_or_after=ORIGINAL_GRIPPER_MANIFEST_RESTORED_AT_UTC,
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

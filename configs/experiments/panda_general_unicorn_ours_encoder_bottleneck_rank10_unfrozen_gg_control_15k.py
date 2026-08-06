"""Unfreeze the rank-10 transformer in the original-setting GG control."""

from pathlib import Path

from configs.panda_comparison_common import configure_gg_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_PATHS_YAML,
    generated_gripper_unicorn_rl_cfg,
)


NAME = (
    "panda_general_unicorn_ours_encoder_bottleneck_rank10_"
    "unfrozen_gg_control_15k"
)
PRETRAIN_CHECKPOINT = Path(
    "/mnt/home/zhengyixin/tool-generalist/artifacts/"
    "oracle_pointnet_rebuild_new200_d12/depth12_joint_self_explicit_v1.pt"
)
BOTTLENECK_PCA = Path(
    "/mnt/home/zhengyixin/tool-generalist/artifacts/projections/"
    "unicorn_ours_encoder_pre_mlp_pca.pt"
)
PARENT_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_unicorn_ours_encoder_bottleneck_rank10_full_yes_5k/"
    "no-contact/TCE/"
    "panda_general_unicorn_ours_encoder_bottleneck_rank10_full_yes_5k/"
    "20260718T131611Z/model_best.pt"
)
for required in (PRETRAIN_CHECKPOINT, BOTTLENECK_PCA, PARENT_CHECKPOINT):
    if not required.is_file():
        raise FileNotFoundError(f"Required control checkpoint is missing: {required}")


EXP_CFG = generated_gripper_unicorn_rl_cfg(NAME, ours_tce=True)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_PATHS_YAML
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = str(PRETRAIN_CHECKPOINT)
EXP_CFG.model.tce.rl_token_source = "encoder"
EXP_CFG.model.tce.encoder_token_pca_rank = EXP_CFG.model.tce.encoder_channel
EXP_CFG.model.tce.encoder_token_pca_path = None
EXP_CFG.model.tce.encoder_token_bottleneck_rank = 10
EXP_CFG.model.tce.encoder_token_bottleneck_pca_path = str(BOTTLENECK_PCA)
EXP_CFG.rl.init_checkpoint = str(PARENT_CHECKPOINT)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = False
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.launch.wandb_project = "ungrasp"

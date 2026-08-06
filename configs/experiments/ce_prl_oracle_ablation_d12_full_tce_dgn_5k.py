"""Newest-200 DGN ablation using the full frozen depth-12 TCE tokens."""

from configs.oracle_pointnet_rebuild_common import SOURCE_PRETRAIN_CHECKPOINT
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    generated_gripper_unicorn_rl_cfg,
)


NAME = "ce_prl_oracle_ablation_d12_full_tce_dgn_5k"
if not SOURCE_PRETRAIN_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Explicit depth-12 encoder is missing: {SOURCE_PRETRAIN_CHECKPOINT}"
    )

EXP_CFG = generated_gripper_unicorn_rl_cfg(NAME, ours_tce=True)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = str(
    SOURCE_PRETRAIN_CHECKPOINT
)
EXP_CFG.model.tce.vit_depth = 12
EXP_CFG.model.tce.vit_attention_mode = "joint_self"
EXP_CFG.model.tce.rl_token_source = "encoder"
EXP_CFG.model.tce.encoder_token_pca_rank = 128
EXP_CFG.model.tce.encoder_token_pca_path = None
EXP_CFG.model.tce.encoder_token_bottleneck_rank = 128
EXP_CFG.model.tce.encoder_token_bottleneck_pca_path = None
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"


"""Train the new-200 frozen-TCE rank-10 teacher on DGN for 5k."""

from configs.oracle_pointnet_rebuild_common import (
    PCA_CHECKPOINT,
    SOURCE_PRETRAIN_CHECKPOINT,
)
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    generated_gripper_unicorn_rl_cfg,
)


NAME = "ce_prl_oracle_rebuild_d12_bottleneck_dgn_5k"
PRETRAIN_CHECKPOINT = SOURCE_PRETRAIN_CHECKPOINT
if not PRETRAIN_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Selected paper-head encoder is missing: {PRETRAIN_CHECKPOINT}"
    )
if not PCA_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Rank-10 PCA initialization is missing: {PCA_CHECKPOINT}")

EXP_CFG = generated_gripper_unicorn_rl_cfg(NAME, ours_tce=True)
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = str(PRETRAIN_CHECKPOINT)
EXP_CFG.model.tce.vit_depth = 12
EXP_CFG.model.tce.vit_attention_mode = "joint_self"
EXP_CFG.model.tce.rl_token_source = "encoder"
EXP_CFG.model.tce.encoder_token_pca_rank = 128
EXP_CFG.model.tce.encoder_token_pca_path = None
EXP_CFG.model.tce.encoder_token_bottleneck_rank = 10
EXP_CFG.model.tce.encoder_token_bottleneck_pca_path = str(PCA_CHECKPOINT)
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

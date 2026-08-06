"""Train the combined-gripper depth-12 rank-10 teacher on DGN for 5k."""

from configs.oracle_pointnet_general_rebuild_common import PCA_CHECKPOINT
from configs.oracle_pointnet_rebuild_common import SOURCE_PRETRAIN_CHECKPOINT
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    cross_embodiment_gripper_unicorn_rl_cfg,
)


NAME = "ce_general_oracle_rebuild_d12_bottleneck_dgn_5k"
if not SOURCE_PRETRAIN_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Explicit depth-12 encoder is missing: {SOURCE_PRETRAIN_CHECKPOINT}"
    )
if not PCA_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Rank-10 PCA initialization is missing: {PCA_CHECKPOINT}")

EXP_CFG = cross_embodiment_gripper_unicorn_rl_cfg(NAME, ours_tce=True)
EXP_CFG.paths_yaml = CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
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
EXP_CFG.model.tce.encoder_token_bottleneck_rank = 10
EXP_CFG.model.tce.encoder_token_bottleneck_pca_path = str(PCA_CHECKPOINT)
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

"""GG15k continuation of the clean parallel fitted-PointNet DGN parent."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT,
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    parallel_fitted_oracle_pointcloud_pointnet_rl_cfg,
)


PARENT = "ce_prl_oracle_pointcloud_pointnet_fitted_unfrozen_dgn_5k"

EXP_CFG = parallel_fitted_oracle_pointcloud_pointnet_rl_cfg(
    "ce_prl_oracle_pointcloud_pointnet_fitted_unfrozen_gg_15k"
)
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
    expected_pretrained_encoder_checkpoint=(
        FITTED_ORACLE_POINTCLOUD_POINTNET_CHECKPOINT
    ),
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)

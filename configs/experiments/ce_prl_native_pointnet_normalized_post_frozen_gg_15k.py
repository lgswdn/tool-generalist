"""GG15k continuation of the frozen normalized direct-PointNet DGN parent."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    parallel_native_pointnet_normalized_postcontact_rl_cfg,
)


PRETRAIN_SOURCE = "ce_prl_native_pointnet_normalized_post_pretrain.py"
PARENT = "ce_prl_native_pointnet_normalized_post_frozen_dgn_5k"
NAME = "ce_prl_native_pointnet_normalized_post_frozen_gg_15k"

EXP_CFG = parallel_native_pointnet_normalized_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain_reuse = PRETRAIN_SOURCE
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = True
configure_gg_comparison(EXP_CFG)

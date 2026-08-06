"""GG15k from safe-velocity Original400 DGN5k, PointNet frozen."""

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_PATHS_YAML,
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


PARENT = "panda_general_native_pointnet_post_original400_safe_velocity_frozen_dgn_5k"
NAME = "panda_general_native_pointnet_post_original400_safe_velocity_frozen_gg_15k"
PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "panda_general_native_pointnet_post_original400_dgn_5k/"
    "contact_gen_generated_gripper/"
    "panda_general_native_pointnet_post_original400_dgn_5k_oracle_pointcloud_pointnet/"
    "f41517a2990854a1d27510e02ff09e83eee009aa95a667315362523f2ffb3d46/"
    "best.pt"
)

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = PRETRAIN_CHECKPOINT
EXP_CFG.rl.init_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="oracle_pointcloud_pointnet",
    expected_paths_yaml=GENERATED_GRIPPER_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_last.pt",
)
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.freeze_encoder = True
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 0.05
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.launch.wandb_project = "ungrasp"

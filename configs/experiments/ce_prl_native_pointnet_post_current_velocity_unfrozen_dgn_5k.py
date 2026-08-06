"""Post-pretrained native PointNet on current-velocity CE-PRL DGN."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    generated_gripper_native_pointnet_postcontact_rl_cfg,
)


POST_PRETRAIN_SOURCE = "panda_general_native_pointnet_post_original400_dgn_5k"
NAME = "ce_prl_native_pointnet_post_current_velocity_unfrozen_dgn_5k"

EXP_CFG = generated_gripper_native_pointnet_postcontact_rl_cfg(NAME)
EXP_CFG.pretrain_reuse = f"{POST_PRETRAIN_SOURCE}.py"
EXP_CFG.paths_yaml = GENERATED_GRIPPER_NEW_PATHS_YAML
EXP_CFG.num_gpus = 8
EXP_CFG.rl.freeze_encoder = False
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 5_000

"""Rich direct-128 PointNet RL on stable goals and the small YES set."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


SMALL_YES_MANIFEST = (
    "../object_selections/panda_general_small_yes_unique.json"
)


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_stable_yes_5k",
    checkpoint_path=None,
)

# Keep the current PointNet encoder architecture and train it directly with RL:
# rich 21D point features -> 64 -> 128 -> 128 -> max pool -> 128 -> 128.
# There is no fitted rank-10 target, fitted checkpoint, or rank-10 bottleneck.
EXP_CFG.model.oracle_pointcloud_pointnet.feature_mode = "rich21"
EXP_CFG.model.oracle_pointcloud_pointnet.load_fitted_weights = False
EXP_CFG.model.oracle_pointcloud_pointnet.use_rank10_bottleneck = False

# Retain the comparison contract (5k iterations, 0.06 action scale and object
# scale randomization), but train on the original small YES object manifest.
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.general.rl_objects_manifest = SMALL_YES_MANIFEST

# Sample stable target poses throughout training, with no curriculum mixture.
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 1.0

# One step is the zero-dwell representation: terminate on the first step for
# which the object is inside both the position and rotation success windows.
EXP_CFG.rl.reward.stable_success_dwell_steps = 1

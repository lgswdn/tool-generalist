"""Unsigned point-to-mesh patchwise PointNet pretrain plus full-YES RL."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointmesh_pointnet_rl_cfg,
)


EXP_CFG = generated_gripper_oracle_pointmesh_pointnet_rl_cfg(
    "panda_general_oracle_pointmesh_pointnet_full_yes_5k"
)
configure_full_yes_comparison(EXP_CFG)

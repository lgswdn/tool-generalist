"""Shared-PointNet three-state gripper kinematics, then 10k DGN RL."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import (
    parallel_kinematic_conditioning_rl_cfg,
)


EXP_CFG = parallel_kinematic_conditioning_rl_cfg(
    "ce_prl_unicorn_d1_full_nonpenetrating_contact_concavity_biased_kinematic_dgn_10k"
)
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.rl.ppo.max_iterations = 10_000

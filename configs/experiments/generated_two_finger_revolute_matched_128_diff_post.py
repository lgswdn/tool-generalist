"""Visualization config for the corrected 200-gripper revolute population."""

from configs.panda_experiment_common import generated_revolute_diff_post_rl_cfg


EXP_CFG = generated_revolute_diff_post_rl_cfg(
    "generated_two_finger_revolute_matched_128_diff_post"
)
EXP_CFG.paths_yaml = (
    "configs/paths/generated_two_finger_revolute_matched_128.yaml"
)

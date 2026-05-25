"""Bimanual stable-target RL using the full-tool diff+post encoder."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_full_tool_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_stable_full_tool_diff_post"
EXP_CFG.general.name = "bimanual_stable_full_tool_diff_post"
EXP_CFG.rl.name = "bimanual_stable_full_tool_diff_post"
EXP_CFG.rl.launch.run_name = "bimanual_stable_full_tool_diff_post"

# There is no separate registered tool-bimanual-stable-v0 task.  Use the
# bimanual target-pose env with target sampling fixed to stable poses.
EXP_CFG.rl.isaac_task_id = "tool-bimanual-unstable-v0"
EXP_CFG.rl.curriculum.enabled = True
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 1.0

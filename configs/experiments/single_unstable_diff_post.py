"""Single-arm unstable-pose RL over the full selected-tool set, without curriculum."""

from copy import deepcopy

from configs.experiments.fork_unstable_diff_post import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "single_unstable_diff_post"
EXP_CFG.general.name = "single_unstable_diff_post"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef_new/tools_selected.json"

EXP_CFG.rl.name = "single_unstable_diff_post"
EXP_CFG.rl.launch.run_name = "single_unstable_diff_post"
EXP_CFG.rl.launch.wandb_project = "single_unstable"
EXP_CFG.rl.curriculum.enabled = False

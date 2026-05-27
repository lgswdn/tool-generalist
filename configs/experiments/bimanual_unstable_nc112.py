"""Bimanual unstable no-curriculum RL with one fork tool and one foam-brick object."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_full_tool_diff_post_no_curriculum import (
    EXP_CFG as _BASE_EXP_CFG,
)


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_unstable_nc112"
EXP_CFG.general.name = "bimanual_unstable_nc112"
EXP_CFG.general.tools_selected_json = "configs/tool_selections/fork_only.json"
EXP_CFG.general.objects_manifest = "configs/object_selections/ycb_foam_brick_only.json"
EXP_CFG.general.randomize_tool_assignment = False
EXP_CFG.general.randomize_object_assignment = False

EXP_CFG.pretrain_reuse = "multitools_full_tool_diff_post.py"

EXP_CFG.rl.name = "bimanual_unstable_nc112"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_nc112"
EXP_CFG.rl.curriculum.enabled = False

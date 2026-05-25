"""Bimanual RL using the full-tool diff pretrained encoder."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_diff_post import EXP_CFG as _BIMANUAL_BASE_CFG
from configs.experiments.multitools_full_tool_contact import FULL_YES_MANIFEST


EXP_CFG = deepcopy(_BIMANUAL_BASE_CFG)

EXP_CFG.name = "bimanual_unstable_full_tool_diff"
EXP_CFG.general.name = "bimanual_unstable_full_tool_diff"
EXP_CFG.general.objects_manifest = FULL_YES_MANIFEST
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef_new/tools_selected.json"
EXP_CFG.pretrain_reuse = "multitools_full_tool_diff.py"

EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.enabled = True
EXP_CFG.rl.name = "bimanual_unstable_full_tool_diff"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_full_tool_diff"

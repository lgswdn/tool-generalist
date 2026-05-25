"""Full-DGN object-vs-selected-tool contact generation only."""

from copy import deepcopy

from configs.config_contact_gen import (
    ROTATION_SELECTION_RANDOM_LEGAL,
    TOOL_SOURCE_SELECTED_TOOLS,
)
from configs.experiments.multitools_diff import EXP_CFG as _BASE_EXP_CFG


FULL_YES_MANIFEST = "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"
FULL_TEST_MANIFEST = "/mnt/project/world_model/tool_generalist/assets/DGN/full_test.json"


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_full_tool_contact"
EXP_CFG.general.name = "multitools_full_contact"
EXP_CFG.general.objects_manifest = FULL_YES_MANIFEST
EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.name = "contact_gen_full_tool"
EXP_CFG.contact_gen.enabled = True
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.contact_gen.rotation_selection = ROTATION_SELECTION_RANDOM_LEGAL
EXP_CFG.contact_gen.tool_source = TOOL_SOURCE_SELECTED_TOOLS
EXP_CFG.contact_gen.object_tool_manifest = None
EXP_CFG.contact_gen.allow_self_object_tool_pairs = False
EXP_CFG.contact_gen.shard_count = 1
EXP_CFG.contact_gen.shard_index = 0

EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.enabled = False
EXP_CFG.rl.launch.distributed = False

"""Shard 0/2 for full-DGN object-vs-selected-tool contact generation."""

from copy import deepcopy

from configs.experiments.multitools_full_tool_contact import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "multitools_full_tool_contact_shard0"
EXP_CFG.contact_gen.shard_count = 2
EXP_CFG.contact_gen.shard_index = 0

"""Bimanual unstable no-curriculum RL with one fork tool and the full object set."""

from copy import deepcopy

from configs.experiments.bimanual_unstable_nc112 import EXP_CFG as _BASE_EXP_CFG
from configs.experiments.multitools_full_tool_contact import FULL_YES_MANIFEST


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "bimanual_unstable_nc1f"
EXP_CFG.general.name = "bimanual_unstable_nc1f"
EXP_CFG.general.objects_manifest = FULL_YES_MANIFEST

EXP_CFG.rl.name = "bimanual_unstable_nc1f"
EXP_CFG.rl.launch.run_name = "bimanual_unstable_nc1f"

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

# Core (always needed)
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_tool_unicorn import ActorCriticToolUnicorn
from .normalizer import EmpiricalNormalization
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

# Optional modules — wrap in try/except to avoid import failures
# from missing dependencies (PointTransformerV3, concerto, etc.)

def _try_import(module_attr, from_module, names):
    """Try to import names from a module, return None for each on failure."""
    results = {}
    try:
        mod = __import__(f"rsl_rl.modules.{from_module}", fromlist=names)
        for name in names:
            results[name] = getattr(mod, name)
    except (ImportError, ModuleNotFoundError):
        for name in names:
            results[name] = None
    return results

_optional = {}
_optional.update(_try_import(globals(), "actor_critic_tg", ["ActorCriticTG"]))
_optional.update(_try_import(globals(), "actor_critic_tg_bimanual", ["ActorCriticTGBimanual"]))
_optional.update(_try_import(globals(), "actor_critic_point2vec", ["ActorCriticPoint2Vec"]))
_optional.update(_try_import(globals(), "actor_critic_icp", ["ActorCriticICP"]))
_optional.update(_try_import(globals(), "rnd", ["RandomNetworkDistillation"]))

# Inject into module namespace for existing import consumers.
globals().update({k: v for k, v in _optional.items() if v is not None})

# Convenience aliases
ActorCriticTG = _optional.get("ActorCriticTG")
ActorCriticTGBimanual = _optional.get("ActorCriticTGBimanual")
ActorCriticPoint2Vec = _optional.get("ActorCriticPoint2Vec")
ActorCriticICP = _optional.get("ActorCriticICP")
RandomNetworkDistillation = _optional.get("RandomNetworkDistillation")

POLICY_REGISTRY = {
    "ActorCriticTG": ActorCriticTG,
    "ActorCriticTGBimanual": ActorCriticTGBimanual,
    "ActorCriticPoint2Vec": ActorCriticPoint2Vec,
    "ActorCriticICP": ActorCriticICP,
}


def resolve_policy_class(class_name: str):
    """Resolve policy class names without eval.

    Tool-generalist configs select a policy by explicit registry entry.
    """

    if class_name not in POLICY_REGISTRY:
        raise ValueError(f"Unsupported policy class for config-driven RL: {class_name}")
    policy_class = POLICY_REGISTRY[class_name]
    if policy_class is None:
        raise ImportError(
            f"Policy class {class_name} is registered but its optional dependencies "
            "could not be imported."
        )
    return policy_class

__all__ = [
    "ActorCritic",
    "ActorCriticPoint2Vec",
    "ActorCriticICP",
    "ActorCriticRecurrent",
    "ActorCriticToolUnicorn",
    "ActorCriticTG",
    "ActorCriticTGBimanual",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "POLICY_REGISTRY",
    "resolve_policy_class",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]

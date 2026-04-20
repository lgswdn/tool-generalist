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
_optional.update(_try_import(globals(), "actor_critic_pointnet", ["ActorCriticPointNet"]))
_optional.update(_try_import(globals(), "actor_critic_multi_icp", ["ActorCriticMultiICP", "ActorCriticMultiICP_HandState"]))
_optional.update(_try_import(globals(), "actor_critic_unicorn", ["ActorCriticUnicorn", "ActorCriticMultiUnicorn"]))
_optional.update(_try_import(globals(), "actor_critic_momentum", ["ActorCriticMomentum"]))
_optional.update(_try_import(globals(), "actor_critic_ptv3_momentum", ["ActorCriticPTV3Momentum"]))
_optional.update(_try_import(globals(), "actor_critic_concerto", ["ActorCriticConcerto"]))
_optional.update(_try_import(globals(), "actor_critic_point2vec", ["ActorCriticPoint2Vec"]))
_optional.update(_try_import(globals(), "actor_critic_icp", ["ActorCriticICP"]))
_optional.update(_try_import(globals(), "actor_critic_sdf", ["ActorCriticSDF"]))
_optional.update(_try_import(globals(), "rnd", ["RandomNetworkDistillation"]))

# Inject into module namespace so eval(class_name) still works
globals().update({k: v for k, v in _optional.items() if v is not None})

# Convenience aliases
ActorCriticPointNet = _optional.get("ActorCriticPointNet")
ActorCriticMultiICP = _optional.get("ActorCriticMultiICP")
ActorCriticMultiICP_HandState = _optional.get("ActorCriticMultiICP_HandState")
ActorCriticUnicorn = _optional.get("ActorCriticUnicorn")
ActorCriticMultiUnicorn = _optional.get("ActorCriticMultiUnicorn")
ActorCriticMomentum = _optional.get("ActorCriticMomentum")
ActorCriticPTV3Momentum = _optional.get("ActorCriticPTV3Momentum")
ActorCriticConcerto = _optional.get("ActorCriticConcerto")
ActorCriticPoint2Vec = _optional.get("ActorCriticPoint2Vec")
ActorCriticICP = _optional.get("ActorCriticICP")
ActorCriticSDF = _optional.get("ActorCriticSDF")
RandomNetworkDistillation = _optional.get("RandomNetworkDistillation")

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticToolUnicorn",
    "ActorCriticPointNet",
    "ActorCriticUnicorn",
    "ActorCriticMultiUnicorn",
    "ActorCriticICP",
    "ActorCriticSDF",
    "ActorCriticMomentum",
    "ActorCriticPTV3Momentum",
    "ActorCriticConcerto",
    "ActorCriticPoint2Vec",
    "ActorCriticMultiICP",
    "ActorCriticMultiICP_HandState",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]

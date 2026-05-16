# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO

ALGORITHM_REGISTRY = {
    "PPO": PPO,
    "Distillation": Distillation,
}


def resolve_algorithm_class(class_name: str):
    algorithm_class = ALGORITHM_REGISTRY.get(class_name)
    if algorithm_class is None:
        raise ValueError(f"Unsupported algorithm class: {class_name}")
    return algorithm_class


__all__ = ["PPO", "Distillation", "ALGORITHM_REGISTRY", "resolve_algorithm_class"]

"""Curriculum terms for non-prehensile manipulation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_pose_stability_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "target_object_pose",
    start_step: int = 0,
    end_step: int = 100000,
    start_stable_probability: float = 1.0,
    end_stable_probability: float = 0.0,
) -> dict[str, float]:
    """Anneal target sampling from stable poses to arbitrary random poses."""

    del env_ids
    start_step = int(start_step)
    end_step = int(end_step)
    step = int(env.common_step_counter)
    if end_step <= start_step:
        progress = 1.0 if step >= end_step else 0.0
    else:
        progress = (step - start_step) / float(end_step - start_step)
        progress = max(0.0, min(1.0, progress))

    start_prob = float(start_stable_probability)
    end_prob = float(end_stable_probability)
    stable_probability = start_prob + progress * (end_prob - start_prob)
    stable_probability = max(0.0, min(1.0, stable_probability))

    command_term = env.command_manager.get_term(command_name)
    command_term.stable_pose_probability = stable_probability
    if hasattr(command_term.cfg, "stable_pose_probability"):
        command_term.cfg.stable_pose_probability = stable_probability

    return {
        "progress": progress,
        "stable_pose_probability": stable_probability,
        "random_pose_probability": 1.0 - stable_probability,
    }

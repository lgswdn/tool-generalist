"""Per-environment-step caches shared by MDP terms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch


_T = TypeVar("_T")


def get_or_compute_step_value(
    env: Any,
    key: tuple[Any, ...],
    factory: Callable[[], _T],
) -> _T:
    """Return a value computed at most once for the current environment step.

    ``common_step_counter`` is incremented immediately after physics and before
    termination/reward evaluation.  Terms called outside that lifecycle are not
    cached because there is no reliable invalidation token.
    """

    step = getattr(env, "common_step_counter", None)
    if step is None:
        return factory()

    step = int(step)
    if getattr(env, "_mdp_step_cache_step", None) != step:
        env._mdp_step_cache_step = step
        env._mdp_step_cache = {}

    cache = env._mdp_step_cache
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def object_goal_geometry(
    env: Any,
    object: Any,
    *,
    command_name: str,
) -> dict[str, Any]:
    """Compute object/goal geometry once and share it across MDP terms."""

    def compute():
        command = env.command_manager.get_command(command_name)
        position_delta = command[:, :3] - (
            object.data.root_pos_w[:, :3] - env.scene.env_origins
        )
        dot_product = torch.sum(object.data.root_quat_w * command[:, 3:7], dim=1)
        dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
        return {
            "position_delta": position_delta,
            "position_distance": torch.norm(position_delta, dim=1),
            "angular_distance": 2.0 * torch.acos(dot_product),
            "success_masks": {},
            "combined_distances": {},
        }

    return get_or_compute_step_value(
        env,
        ("object_goal_geometry", id(object), command_name),
        compute,
    )


def object_pose_success_mask(
    env: Any,
    object: Any,
    *,
    command_name: str,
    threshold: float,
    rotation_threshold: float,
    planar: bool = False,
) -> torch.Tensor:
    """Return a cached goal-pose threshold mask for reward/termination terms."""

    geometry = object_goal_geometry(env, object, command_name=command_name)
    mask_key = (float(threshold), float(rotation_threshold), bool(planar))
    masks = geometry["success_masks"]
    if mask_key not in masks:
        if planar:
            position_distance = torch.norm(geometry["position_delta"][:, :2], dim=1)
            masks[mask_key] = position_distance < threshold
        else:
            masks[mask_key] = (
                (geometry["position_distance"] < threshold)
                & (geometry["angular_distance"] < rotation_threshold)
            )
    return masks[mask_key]

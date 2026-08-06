"""Pure runtime-spec compatibility helpers for fixed-episode recording."""

from __future__ import annotations

import math
from typing import Any


def backfill_legacy_fixed_episode_fields(spec: dict[str, Any]) -> None:
    """Restore audited fields omitted by legacy runtime-spec serialization."""

    sections = (
        ("observation_params", spec.get("observation_params")),
        ("policy_params", spec.get("policy_params")),
    )
    for section_name, section in sections:
        if not isinstance(section, dict):
            raise RuntimeError(f"runtime_spec {section_name} must be a JSON object")
        if "task_embedding_dim" in section:
            value = section["task_embedding_dim"]
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise RuntimeError(
                    f"runtime_spec {section_name}.task_embedding_dim must be 0 for legacy "
                    f"fixed-episode replay, got {value!r}"
                )
        else:
            section["task_embedding_dim"] = 0

    object_pose_sampling = spec.get("object_pose_sampling_params")
    if not isinstance(object_pose_sampling, dict):
        raise RuntimeError("runtime_spec object_pose_sampling_params must be a JSON object")

    if "secondary_task" in object_pose_sampling:
        secondary_task = object_pose_sampling["secondary_task"]
        if not isinstance(secondary_task, str) or secondary_task not in {
            "random_pose",
            "grasp_lift",
        }:
            raise RuntimeError(
                "runtime_spec object_pose_sampling_params.secondary_task must be "
                f"'random_pose' or 'grasp_lift', got {secondary_task!r}"
            )
    else:
        object_pose_sampling["secondary_task"] = "random_pose"

    if "grasp_lift_height" in object_pose_sampling:
        grasp_lift_height = object_pose_sampling["grasp_lift_height"]
        if (
            isinstance(grasp_lift_height, bool)
            or not isinstance(grasp_lift_height, (int, float))
            or not math.isfinite(float(grasp_lift_height))
            or float(grasp_lift_height) <= 0.0
        ):
            raise RuntimeError(
                "runtime_spec object_pose_sampling_params.grasp_lift_height must be a "
                f"positive finite number, got {grasp_lift_height!r}"
            )
    else:
        object_pose_sampling["grasp_lift_height"] = 0.05

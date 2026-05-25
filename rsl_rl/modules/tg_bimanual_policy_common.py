"""Shared helpers for bimanual Tool-Generalist policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class SliceSpec:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BimanualObservationLayout:
    """Named slices for env_tool_bimanual_unstable.PolicyCfg order."""

    object_cloud: SliceSpec
    tool1_cloud: SliceSpec
    tool2_cloud: SliceSpec
    object_bbox_center: SliceSpec
    tool1_bbox_center: SliceSpec
    tool2_bbox_center: SliceSpec
    hand1_state: SliceSpec
    hand2_state: SliceSpec
    robot1_state: SliceSpec
    robot2_state: SliceSpec
    previous_action: SliceSpec
    relative_goal_pose: SliceSpec
    object_velocity: SliceSpec
    physics: SliceSpec
    total_dim: int

    @classmethod
    def build(
        cls,
        *,
        num_points: int,
        point_dim: int,
        hand_state_dim: int,
        robot_state_dim: int,
        previous_action_dim: int,
        relative_goal_dim: int,
        object_velocity_dim: int,
        physics_dim: int,
    ) -> "BimanualObservationLayout":
        offset = 0

        def take(name: str, dim: int, shape: tuple[int, ...]) -> SliceSpec:
            nonlocal offset
            spec = SliceSpec(name=name, start=offset, stop=offset + dim, shape=shape)
            offset += dim
            return spec

        hand_each = int(hand_state_dim) // 2
        robot_each = int(robot_state_dim) // 2
        object_cloud = take("object_cloud", num_points * point_dim, (num_points, point_dim))
        tool1_cloud = take("tool1_cloud", num_points * point_dim, (num_points, point_dim))
        tool2_cloud = take("tool2_cloud", num_points * point_dim, (num_points, point_dim))
        object_bbox_center = take("object_bbox_center", 3, (3,))
        tool1_bbox_center = take("tool1_bbox_center", 3, (3,))
        tool2_bbox_center = take("tool2_bbox_center", 3, (3,))
        hand1_state = take("hand1_state", hand_each, (hand_each,))
        hand2_state = take("hand2_state", hand_each, (hand_each,))
        robot1_state = take("robot1_state", robot_each, (robot_each,))
        robot2_state = take("robot2_state", robot_each, (robot_each,))
        previous_action = take("previous_action", previous_action_dim, (previous_action_dim,))
        relative_goal_pose = take("relative_goal_pose", relative_goal_dim, (relative_goal_dim,))
        object_velocity = take("object_velocity", object_velocity_dim, (object_velocity_dim,))
        physics = take("physics", physics_dim, (physics_dim,))
        return cls(
            object_cloud=object_cloud,
            tool1_cloud=tool1_cloud,
            tool2_cloud=tool2_cloud,
            object_bbox_center=object_bbox_center,
            tool1_bbox_center=tool1_bbox_center,
            tool2_bbox_center=tool2_bbox_center,
            hand1_state=hand1_state,
            hand2_state=hand2_state,
            robot1_state=robot1_state,
            robot2_state=robot2_state,
            previous_action=previous_action,
            relative_goal_pose=relative_goal_pose,
            object_velocity=object_velocity,
            physics=physics,
            total_dim=offset,
        )


def split_bimanual_observations(obs: torch.Tensor, layout: BimanualObservationLayout) -> dict[str, torch.Tensor]:
    def take(spec: SliceSpec) -> torch.Tensor:
        value = obs[:, spec.start:spec.stop]
        if spec.stop == spec.start:
            return value.new_zeros((obs.shape[0], 0))
        if len(spec.shape) == 2:
            return value.view(-1, spec.shape[0], spec.shape[1])
        return value.view(-1, spec.shape[0])

    return {
        "object_cloud": take(layout.object_cloud),
        "tool1_cloud": take(layout.tool1_cloud),
        "tool2_cloud": take(layout.tool2_cloud),
        "object_bbox_center": take(layout.object_bbox_center),
        "tool1_bbox_center": take(layout.tool1_bbox_center),
        "tool2_bbox_center": take(layout.tool2_bbox_center),
        "hand1_state": take(layout.hand1_state),
        "hand2_state": take(layout.hand2_state),
        "robot1_state": take(layout.robot1_state),
        "robot2_state": take(layout.robot2_state),
        "previous_action": take(layout.previous_action),
        "relative_goal_pose": take(layout.relative_goal_pose),
        "object_velocity": take(layout.object_velocity),
        "physics": take(layout.physics),
    }


def center_bimanual_clouds_by_bbox(
    object_cloud: torch.Tensor,
    tool1_cloud: torch.Tensor,
    tool2_cloud: torch.Tensor,
    object_bbox_center: torch.Tensor,
    tool1_bbox_center: torch.Tensor,
    tool2_bbox_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        object_cloud - object_bbox_center.unsqueeze(1),
        tool1_cloud - tool1_bbox_center.unsqueeze(1),
        tool2_cloud - tool2_bbox_center.unsqueeze(1),
    )


def bimanual_context_dim(
    *,
    hand_state_dim: int,
    robot_state_dim: int,
    previous_action_dim: int,
    relative_goal_dim: int,
    object_velocity_dim: int,
    physics_dim: int,
) -> int:
    return (
        6
        + 3
        + hand_state_dim
        + robot_state_dim
        + previous_action_dim
        + relative_goal_dim
        + object_velocity_dim
        + physics_dim
    )


def build_bimanual_context_vector(parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    object_bbox_center = parts["object_bbox_center"]
    tool1_bbox_center = parts["tool1_bbox_center"]
    tool2_bbox_center = parts["tool2_bbox_center"]
    return torch.cat(
        [
            tool1_bbox_center - object_bbox_center,
            tool2_bbox_center - object_bbox_center,
            object_bbox_center,
            parts["hand1_state"],
            parts["hand2_state"],
            parts["robot1_state"],
            parts["robot2_state"],
            parts["previous_action"],
            parts["relative_goal_pose"],
            parts["object_velocity"],
            parts["physics"],
        ],
        dim=-1,
    )

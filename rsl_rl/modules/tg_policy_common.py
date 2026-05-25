"""Shared Tool-Generalist policy utilities.

ActorCriticTG and ActorCriticPoint2Vec may use different point-cloud encoders,
but the observation split, bbox-center relative clouds, context vector, and
learnable policy heads are owned here so their RL parameters stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn

from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet


@dataclass(frozen=True)
class SliceSpec:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ObservationLayout:
    """Named observation slices matching env_tool.PolicyCfg concatenate order."""

    object_cloud: SliceSpec
    tool_cloud: SliceSpec
    object_bbox_center: SliceSpec
    tool_bbox_center: SliceSpec
    hand_state: SliceSpec
    robot_state: SliceSpec
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
    ) -> "ObservationLayout":
        offset = 0

        def take(name: str, dim: int, shape: tuple[int, ...]) -> SliceSpec:
            nonlocal offset
            spec = SliceSpec(name=name, start=offset, stop=offset + dim, shape=shape)
            offset += dim
            return spec

        object_cloud = take("object_cloud", num_points * point_dim, (num_points, point_dim))
        tool_cloud = take("tool_cloud", num_points * point_dim, (num_points, point_dim))
        object_bbox_center = take("object_bbox_center", 3, (3,))
        tool_bbox_center = take("tool_bbox_center", 3, (3,))
        hand_state = take("hand_state", hand_state_dim, (hand_state_dim,))
        robot_state = take("robot_state", robot_state_dim, (robot_state_dim,))
        previous_action = take("previous_action", previous_action_dim, (previous_action_dim,))
        relative_goal_pose = take("relative_goal_pose", relative_goal_dim, (relative_goal_dim,))
        object_velocity = take("object_velocity", object_velocity_dim, (object_velocity_dim,))
        physics = take("physics", physics_dim, (physics_dim,))
        return cls(
            object_cloud=object_cloud,
            tool_cloud=tool_cloud,
            object_bbox_center=object_bbox_center,
            tool_bbox_center=tool_bbox_center,
            hand_state=hand_state,
            robot_state=robot_state,
            previous_action=previous_action,
            relative_goal_pose=relative_goal_pose,
            object_velocity=object_velocity,
            physics=physics,
            total_dim=offset,
        )


def split_observations(obs: torch.Tensor, layout: ObservationLayout) -> dict[str, torch.Tensor]:
    def take(spec: SliceSpec) -> torch.Tensor:
        value = obs[:, spec.start:spec.stop]
        if spec.stop == spec.start:
            return value.new_zeros((obs.shape[0], 0))
        if len(spec.shape) == 2:
            return value.view(-1, spec.shape[0], spec.shape[1])
        return value.view(-1, spec.shape[0])

    return {
        "object_cloud": take(layout.object_cloud),
        "tool_cloud": take(layout.tool_cloud),
        "object_bbox_center": take(layout.object_bbox_center),
        "tool_bbox_center": take(layout.tool_bbox_center),
        "hand_state": take(layout.hand_state),
        "robot_state": take(layout.robot_state),
        "previous_action": take(layout.previous_action),
        "relative_goal_pose": take(layout.relative_goal_pose),
        "object_velocity": take(layout.object_velocity),
        "physics": take(layout.physics),
    }


def center_clouds_by_bbox(
    object_cloud: torch.Tensor,
    tool_cloud: torch.Tensor,
    object_bbox_center: torch.Tensor,
    tool_bbox_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clouds relative to mesh AABB centers, preserving env-frame obs."""

    object_cloud_rel = object_cloud - object_bbox_center.unsqueeze(1)
    tool_cloud_rel = tool_cloud - tool_bbox_center.unsqueeze(1)
    return object_cloud_rel, tool_cloud_rel


def context_dim(
    *,
    hand_state_dim: int,
    robot_state_dim: int,
    previous_action_dim: int,
    relative_goal_dim: int,
    object_velocity_dim: int,
    physics_dim: int,
) -> int:
    return (
        3
        + 3
        + hand_state_dim
        + robot_state_dim
        + previous_action_dim
        + relative_goal_dim
        + object_velocity_dim
        + physics_dim
    )


def build_context_vector(parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Strict context order shared by all TG-compatible actor classes."""

    object_bbox_center = parts["object_bbox_center"]
    tool_bbox_center = parts["tool_bbox_center"]
    return torch.cat(
        [
            tool_bbox_center - object_bbox_center,
            object_bbox_center,
            parts["hand_state"],
            parts["robot_state"],
            parts["previous_action"],
            parts["relative_goal_pose"],
            parts["object_velocity"],
            parts["physics"],
        ],
        dim=-1,
    )


def build_mlp(
    input_dim: int,
    hidden_dims,
    activation: nn.Module,
    output_dim: Optional[int] = None,
) -> nn.Module:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden)))
        layers.append(activation)
        prev_dim = int(hidden)
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers) if layers else nn.Identity()


def build_fusion_mlp(input_dim: int, hidden_dims, activation: nn.Module) -> nn.Module:
    return build_mlp(input_dim, hidden_dims, activation, output_dim=None)


def build_state_cross_attention(
    *,
    total_num_tokens: int,
    token_dim: int,
    ctx_dim: int,
    sd_num_query: int,
    sd_emb_dim: int,
    sd_cat_query: bool,
    sd_query_keys: tuple[str, ...],
) -> tuple[StateDependentCrossFeatNet, int]:
    """Build the shared state-dependent cross-attention fusion block."""

    module = StateDependentCrossFeatNet(
        StateDependentCrossFeatNet.Config(
            dim_in=(int(total_num_tokens), int(token_dim)),
            dim_out=int(sd_emb_dim),
            query_keys=tuple(sd_query_keys),
            num_query=int(sd_num_query),
            ctx_dim=int(ctx_dim),
            emb_dim=int(sd_emb_dim),
            cat_query=bool(sd_cat_query),
            cat_ctx=False,
        )
    )
    output_dim = int(sd_num_query) * int(sd_emb_dim)
    if sd_cat_query:
        output_dim += int(sd_num_query) * int(sd_emb_dim)
    return module, output_dim

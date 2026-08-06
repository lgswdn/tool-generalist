"""Shared Tool-Generalist policy utilities.

ActorCriticTG and ActorCriticPoint2Vec may use different point-cloud encoders,
but the observation split, bbox-center relative clouds, context vector, and
learnable policy heads are owned here so their RL parameters stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

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
    kinematic_gripper_clouds: SliceSpec
    oracle_mesh_signed_sdf: SliceSpec
    oracle_mesh_unsigned_distance: SliceSpec
    object_bbox_center: SliceSpec
    tool_bbox_center: SliceSpec
    hand_state: SliceSpec
    robot_state: SliceSpec
    previous_action: SliceSpec
    relative_goal_pose: SliceSpec
    object_velocity: SliceSpec
    task_embedding: SliceSpec
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
        task_embedding_dim: int = 0,
        oracle_mesh_sdf_dim: int = 0,
        oracle_mesh_unsigned_distance_dim: int = 0,
        include_kinematic_gripper_clouds: bool = False,
    ) -> "ObservationLayout":
        offset = 0

        def take(name: str, dim: int, shape: tuple[int, ...]) -> SliceSpec:
            nonlocal offset
            spec = SliceSpec(name=name, start=offset, stop=offset + dim, shape=shape)
            offset += dim
            return spec

        object_cloud = take("object_cloud", num_points * point_dim, (num_points, point_dim))
        tool_cloud = take("tool_cloud", num_points * point_dim, (num_points, point_dim))
        kinematic_gripper_clouds = take(
            "kinematic_gripper_clouds",
            3 * num_points * point_dim if include_kinematic_gripper_clouds else 0,
            (3, num_points, point_dim)
            if include_kinematic_gripper_clouds
            else (0,),
        )
        oracle_mesh_signed_sdf = take(
            "oracle_mesh_signed_sdf",
            oracle_mesh_sdf_dim,
            (oracle_mesh_sdf_dim,),
        )
        oracle_mesh_unsigned_distance = take(
            "oracle_mesh_unsigned_distance",
            oracle_mesh_unsigned_distance_dim,
            (oracle_mesh_unsigned_distance_dim,),
        )
        object_bbox_center = take("object_bbox_center", 3, (3,))
        tool_bbox_center = take("tool_bbox_center", 3, (3,))
        hand_state = take("hand_state", hand_state_dim, (hand_state_dim,))
        robot_state = take("robot_state", robot_state_dim, (robot_state_dim,))
        previous_action = take("previous_action", previous_action_dim, (previous_action_dim,))
        relative_goal_pose = take("relative_goal_pose", relative_goal_dim, (relative_goal_dim,))
        object_velocity = take("object_velocity", object_velocity_dim, (object_velocity_dim,))
        task_embedding = take("task_embedding", task_embedding_dim, (task_embedding_dim,))
        physics = take("physics", physics_dim, (physics_dim,))
        return cls(
            object_cloud=object_cloud,
            tool_cloud=tool_cloud,
            kinematic_gripper_clouds=kinematic_gripper_clouds,
            oracle_mesh_signed_sdf=oracle_mesh_signed_sdf,
            oracle_mesh_unsigned_distance=oracle_mesh_unsigned_distance,
            object_bbox_center=object_bbox_center,
            tool_bbox_center=tool_bbox_center,
            hand_state=hand_state,
            robot_state=robot_state,
            previous_action=previous_action,
            relative_goal_pose=relative_goal_pose,
            object_velocity=object_velocity,
            task_embedding=task_embedding,
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
        if len(spec.shape) == 3:
            return value.view(
                -1, spec.shape[0], spec.shape[1], spec.shape[2]
            )
        return value.view(-1, spec.shape[0])

    return {
        "object_cloud": take(layout.object_cloud),
        "tool_cloud": take(layout.tool_cloud),
        "kinematic_gripper_clouds": take(layout.kinematic_gripper_clouds),
        "oracle_mesh_signed_sdf": take(layout.oracle_mesh_signed_sdf),
        "oracle_mesh_unsigned_distance": take(layout.oracle_mesh_unsigned_distance),
        "object_bbox_center": take(layout.object_bbox_center),
        "tool_bbox_center": take(layout.tool_bbox_center),
        "hand_state": take(layout.hand_state),
        "robot_state": take(layout.robot_state),
        "previous_action": take(layout.previous_action),
        "relative_goal_pose": take(layout.relative_goal_pose),
        "object_velocity": take(layout.object_velocity),
        "task_embedding": take(layout.task_embedding),
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
    task_embedding_dim: int = 0,
) -> int:
    return (
        3
        + 3
        + hand_state_dim
        + robot_state_dim
        + previous_action_dim
        + relative_goal_dim
        + object_velocity_dim
        + task_embedding_dim
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
            parts["task_embedding"],
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


def validate_observation_layout(
    *,
    policy_name: str,
    num_actor_obs: int,
    num_critic_obs: int,
    layout: Any,
) -> None:
    if num_actor_obs != layout.total_dim:
        raise ValueError(
            f"{policy_name} observation layout mismatch: "
            f"num_actor_obs={num_actor_obs}, expected={layout.total_dim}"
        )
    if num_critic_obs != num_actor_obs:
        raise ValueError(
            f"{policy_name} expects critic observations to use the same named layout "
            f"as actor observations, got num_critic_obs={num_critic_obs}, "
            f"num_actor_obs={num_actor_obs}"
        )


def initialize_action_noise(
    module: nn.Module,
    *,
    num_actions: int,
    init_noise_std: float,
    noise_std_type: str,
) -> None:
    if noise_std_type == "scalar":
        module.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif noise_std_type == "log":
        module.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
        raise ValueError("noise_std_type must be 'scalar' or 'log'")
    module.distribution = None
    Normal.set_default_validate_args(False)


class TGActorCriticHeadMixin:
    """Shared TG-style actor/critic action distribution helpers."""

    def _get_features(self, observations: torch.Tensor, *, branch: str = "actor") -> torch.Tensor:
        all_tokens, ctx_vec = self._tokenize(observations)
        return self._features_from_tokens_context(all_tokens, ctx_vec, branch=branch)

    def _action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(self._get_features(observations))
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        return self.actor(self._get_features(observations))

    def reset(self, dones=None):
        pass

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(self._get_features(critic_observations, branch="critic"))

    def get_cached_encoder_features(self, observations: torch.Tensor):
        return self._tokenize(observations)

    def act_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        mean = self.actor(self._features_from_tokens_context(all_tokens, ctx_vec))
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))
        return self.distribution.sample()

    def evaluate_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        return self.critic(self._features_from_tokens_context(all_tokens, ctx_vec, branch="critic"))

    def get_actions_log_prob_from_cached_features(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        return self.actor(self._features_from_tokens_context(all_tokens, ctx_vec))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

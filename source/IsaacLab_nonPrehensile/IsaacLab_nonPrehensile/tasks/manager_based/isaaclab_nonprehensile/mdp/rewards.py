# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import get_head_area_pos_w
from .step_cache import (
    get_or_compute_step_value,
    object_goal_geometry,
    object_pose_success_mask,
)


def _reward_head_area_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute the head-area position once per environment step for reward terms."""

    return get_or_compute_step_value(
        env,
        ("reward_head_area_pos_w",),
        lambda: get_head_area_pos_w(env),
    )


def _object_ee_distance(
    env: ManagerBasedRLEnv,
    object: RigidObject,
) -> torch.Tensor:
    """Compute the object/head-area distance once per step."""

    return get_or_compute_step_value(
        env,
        ("object_ee_distance", id(object)),
        lambda: torch.norm(object.data.root_pos_w - _reward_head_area_pos_w(env), dim=1),
    )


def _object_ee_within_threshold(
    env: ManagerBasedRLEnv,
    object: RigidObject,
    threshold: float,
) -> torch.Tensor:
    """Share the contact gate used by coarse and fine goal rewards."""

    threshold = float(threshold)
    return get_or_compute_step_value(
        env,
        ("object_ee_within_threshold", id(object), threshold),
        lambda: _object_ee_distance(env, object) < threshold,
    )


def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    object_ee_distance = _object_ee_distance(env, object)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    obj_ee_distance_threshold: float = 0.05,
    rotation_distance_divisor: float = 5.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    obj_ee_dist_cond = _object_ee_within_threshold(
        env, object, obj_ee_distance_threshold
    )
    geometry = object_goal_geometry(env, object, command_name=command_name)
    combined_key = float(rotation_distance_divisor)
    combined_distances = geometry["combined_distances"]
    if combined_key not in combined_distances:
        combined_distances[combined_key] = geometry["position_distance"] + (
            torch.clamp(geometry["angular_distance"], max=torch.pi)
            / rotation_distance_divisor
        )

    return obj_ee_dist_cond * (1 - torch.tanh(combined_distances[combined_key] / std))

def joint_power_penalty(
    env: ManagerBasedRLEnv,
    k_e: float = 0.0001,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Energy penalty
    
    Penalty form: c_energy = k_e * Σ(τ_i * q̇_i)
    where τ_i is joint torque and q̇_i is joint velocity.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get joint torques and velocities
    joint_torques = robot.data.applied_torque  # (num_envs, num_joints)
    joint_velocities = robot.data.joint_vel    # (num_envs, num_joints)
    
    # Calculate power for each joint: τ * q̇
    joint_power = joint_torques * joint_velocities  # (num_envs, num_joints)
    
    # Sum over all joints and apply scaling
    total_power = torch.sum(torch.abs(joint_power), dim=1)  # (num_envs,)
    
    penalty = k_e * total_power
    
    return penalty


def task_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar: bool = False,  # If True, only consider x and y position
    base_reward: float = 1.0,  # Base reward for success
) -> torch.Tensor:
    """Task success reward: gives base_reward when object reaches target pose within thresholds."""
    object: RigidObject = env.scene[object_cfg.name]
    success_mask = object_pose_success_mask(
        env,
        object,
        command_name=command_name,
        threshold=threshold,
        rotation_threshold=rotation_threshold,
        planar=planar,
    )

    if planar:
        # Preserve the legacy planar behavior, which ignored base_reward.
        return success_mask.float()
    
    # Final reward: base_reward for success, 0.0 for failure
    reward = torch.where(
        success_mask,
        base_reward,
        torch.zeros_like(success_mask, dtype=torch.float)
    )

    return reward


def task_success_from_termination(
    env: ManagerBasedRLEnv,
    term_name: str = "reached",
    base_reward: float = 1.0,
) -> torch.Tensor:
    """Reward only when the named termination term reports episode success."""

    return env.termination_manager.get_term(term_name).float() * float(base_reward)


def object_within_goal_threshold(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.05,
    rotation_threshold: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    base_reward: float = 1.0,
) -> torch.Tensor:
    """Reward the object being inside the target pose success window."""

    object: RigidObject = env.scene[object_cfg.name]
    pose_success = _object_pose_success_mask(
        env,
        object,
        command_name=command_name,
        threshold=threshold,
        rotation_threshold=rotation_threshold,
    )
    return torch.where(
        pose_success,
        torch.full_like(object.data.root_pos_w[:, 0], float(base_reward)),
        torch.zeros_like(object.data.root_pos_w[:, 0]),
    )


def _object_pose_success_mask(
    env: ManagerBasedRLEnv,
    object: RigidObject,
    *,
    command_name: str,
    threshold: float,
    rotation_threshold: float,
) -> torch.Tensor:
    return object_pose_success_mask(
        env,
        object,
        command_name=command_name,
        threshold=threshold,
        rotation_threshold=rotation_threshold,
    )

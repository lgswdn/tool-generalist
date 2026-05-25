# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations_bimanual import get_head_area_pos_w_for_slot

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _bimanual_head_positions_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    return (
        get_head_area_pos_w_for_slot(env, ee_frame_name="ee_frame_1", offsets_attr="_head_area_offsets_1"),
        get_head_area_pos_w_for_slot(env, ee_frame_name="ee_frame_2", offsets_attr="_head_area_offsets_2"),
    )


def _farthest_ee_distance_to_object(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee1_pos, ee2_pos = _bimanual_head_positions_w(env)
    dist1 = torch.norm(obj.data.root_pos_w - ee1_pos, dim=1)
    dist2 = torch.norm(obj.data.root_pos_w - ee2_pos, dim=1)
    return torch.maximum(dist1, dist2)


def bimanual_object_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward both bimanual end-effectors being close to the object."""

    distance = _farthest_ee_distance_to_object(env, object_cfg=object_cfg)
    return 1 - torch.tanh(distance / float(std))


def bimanual_object_goal_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    obj_ee_distance_threshold: float = 0.05,
    rotation_distance_divisor: float = 5.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object-goal reward gated by both bimanual end-effectors."""

    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_ee_distance = _farthest_ee_distance_to_object(env, object_cfg=object_cfg)
    obj_ee_dist_cond = object_ee_distance < float(obj_ee_distance_threshold)

    des_pos_env = command[:, :3]
    object_pos_env = obj.data.root_pos_w[:, :3] - env.scene.env_origins
    pos_distance = torch.norm(des_pos_env - object_pos_env, dim=1)

    des_rot_env = command[:, 3:7]
    object_quat_w = obj.data.root_quat_w
    dot_product = torch.sum(object_quat_w * des_rot_env, dim=1)
    dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
    ang_distance = 2 * torch.acos(dot_product)
    ang_distance = torch.clamp(ang_distance, max=torch.pi)
    pose_distance = pos_distance + ang_distance / float(rotation_distance_divisor)

    return obj_ee_dist_cond * (1 - torch.tanh(pose_distance / float(std)))


def bimanual_joint_power_penalty(
    env: "ManagerBasedRLEnv",
    k_e: float = 0.0001,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Energy penalty summed across both robot articulations."""

    total = None
    for cfg in (robot1_cfg, robot2_cfg):
        robot = env.scene[cfg.name]
        power = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)
        total = power if total is None else total + power
    return float(k_e) * total

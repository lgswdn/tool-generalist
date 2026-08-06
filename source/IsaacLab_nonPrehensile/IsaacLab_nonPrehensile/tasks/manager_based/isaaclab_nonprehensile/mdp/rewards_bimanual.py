# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from .observations_bimanual import get_head_area_pos_w_for_slot
from .step_cache import get_or_compute_step_value, object_goal_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _bimanual_head_positions_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    return get_or_compute_step_value(
        env,
        ("bimanual_head_positions_w",),
        lambda: (
            get_head_area_pos_w_for_slot(env, ee_frame_name="ee_frame_1", offsets_attr="_head_area_offsets_1"),
            get_head_area_pos_w_for_slot(env, ee_frame_name="ee_frame_2", offsets_attr="_head_area_offsets_2"),
        ),
    )


def _farthest_ee_distance_to_object(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]

    def compute():
        ee1_pos, ee2_pos = _bimanual_head_positions_w(env)
        dist1 = torch.norm(obj.data.root_pos_w - ee1_pos, dim=1)
        dist2 = torch.norm(obj.data.root_pos_w - ee2_pos, dim=1)
        return torch.maximum(dist1, dist2)

    return get_or_compute_step_value(
        env,
        ("bimanual_farthest_ee_distance", id(obj)),
        compute,
    )


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
    threshold = float(obj_ee_distance_threshold)
    obj_ee_dist_cond = get_or_compute_step_value(
        env,
        ("bimanual_ee_within_threshold", id(obj), threshold),
        lambda: _farthest_ee_distance_to_object(env, object_cfg=object_cfg) < threshold,
    )
    geometry = object_goal_geometry(env, obj, command_name=command_name)
    combined_key = float(rotation_distance_divisor)
    combined_distances = geometry["combined_distances"]
    if combined_key not in combined_distances:
        combined_distances[combined_key] = geometry["position_distance"] + (
            torch.clamp(geometry["angular_distance"], max=torch.pi)
            / float(rotation_distance_divisor)
        )

    return obj_ee_dist_cond * (1 - torch.tanh(combined_distances[combined_key] / float(std)))


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


def bimanual_link_min_distance(
    env: "ManagerBasedRLEnv",
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Minimum pairwise distance between selected bodies on the two bimanual arms."""

    robot1 = env.scene[robot1_cfg.name]
    robot2 = env.scene[robot2_cfg.name]

    def compute():
        pos1 = robot1.data.body_pos_w[:, robot1_cfg.body_ids, :]
        pos2 = robot2.data.body_pos_w[:, robot2_cfg.body_ids, :]
        distances = torch.cdist(pos1, pos2)
        return torch.amin(distances, dim=(1, 2))

    return get_or_compute_step_value(
        env,
        (
            "bimanual_link_min_distance",
            id(robot1),
            repr(robot1_cfg.body_ids),
            id(robot2),
            repr(robot2_cfg.body_ids),
        ),
        compute,
    )


def bimanual_link_proximity_penalty(
    env: "ManagerBasedRLEnv",
    warning_distance: float = 0.20,
    failure_distance: float = 0.15,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Penalty that starts below warning distance and steepens at the failure distance."""

    warning_distance = float(warning_distance)
    failure_distance = float(failure_distance)
    if warning_distance <= failure_distance:
        raise ValueError("warning_distance must be greater than failure_distance")

    min_distance = bimanual_link_min_distance(env, robot1_cfg=robot1_cfg, robot2_cfg=robot2_cfg)
    normalized_violation = torch.clamp(
        (warning_distance - min_distance) / (warning_distance - failure_distance),
        min=0.0,
    )
    return torch.square(normalized_violation)


def _bimanual_runtime():
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import env_tool_bimanual_unstable as env_mod

    return env_mod


def _tool_slot_assignment(tool_slot: int):
    env_mod = _bimanual_runtime()
    if int(tool_slot) == 1:
        return env_mod.get_tool1_index_for_env, env_mod.get_tool1_data_for_env
    if int(tool_slot) == 2:
        return env_mod.get_tool2_index_for_env, env_mod.get_tool2_data_for_env
    raise ValueError(f"tool_slot must be 1 or 2, got {tool_slot!r}")


def _tool_local_points(td: dict, env_mod, device: torch.device) -> torch.Tensor:
    cloud = env_mod.get_cached_cloud(td["obj_path"])
    if hasattr(cloud, "_get_points_torch"):
        base_pts = cloud._get_points_torch(device).float()
    else:
        base_pts = torch.tensor(cloud.points, dtype=torch.float32, device=device)
    pts = base_pts * float(env_mod.TOOL_SCALE)

    base_center_norm = td.get("base_center")
    if base_center_norm is not None:
        bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
        bbox_min_raw = base_pts.min(dim=0).values
        bbox_max_raw = base_pts.max(dim=0).values
        body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * float(env_mod.TOOL_SCALE)
        return pts - body_origin

    pts = pts.clone()
    pts[:, 2] = pts[:, 2] - pts[:, 2].min()
    return pts


def _deterministic_point_subset(points: torch.Tensor, num_points: int) -> torch.Tensor:
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    if points.shape[0] == 0:
        raise ValueError("tool point cloud must contain at least one point")
    if points.shape[0] >= num_points:
        indices = torch.linspace(0, points.shape[0] - 1, num_points, device=points.device).round().long()
    else:
        indices = torch.arange(num_points, device=points.device) % points.shape[0]
    return points[indices].contiguous()


def _tool_local_distance_points(env: "ManagerBasedRLEnv", *, tool_slot: int, num_points: int) -> torch.Tensor:
    cache_attr = f"_tool{tool_slot}_local_distance_points_{int(num_points)}"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    env_mod = _bimanual_runtime()
    get_tool_index_for_env, get_tool_data_for_env = _tool_slot_assignment(tool_slot)
    local_points = torch.empty((env.num_envs, int(num_points), 3), device=env.device, dtype=torch.float32)
    per_tool_cache: dict[int, torch.Tensor] = {}

    for env_id in range(env.num_envs):
        tool_idx = get_tool_index_for_env(env_id)
        points = per_tool_cache.get(tool_idx)
        if points is None:
            td = get_tool_data_for_env(env_id)
            points = _deterministic_point_subset(
                _tool_local_points(td, env_mod, env.device),
                int(num_points),
            )
            per_tool_cache[tool_idx] = points

        local_points[env_id] = points

    setattr(env, cache_attr, local_points)
    return local_points


def _transform_tool_points_w(body_state: torch.Tensor, local_points: torch.Tensor) -> torch.Tensor:
    rot = matrix_from_quat(body_state[:, 3:7])
    points_rotated = torch.bmm(
        rot,
        local_points.transpose(1, 2),
    ).transpose(1, 2)
    return points_rotated + body_state[:, :3].unsqueeze(1)


def _batched_pointcloud_min_distance(points1: torch.Tensor, points2: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    min_distance = torch.full((points1.shape[0],), float("inf"), device=points1.device, dtype=points1.dtype)
    for start in range(0, points1.shape[1], int(chunk_size)):
        distances = torch.cdist(points1[:, start : start + int(chunk_size), :], points2)
        min_distance = torch.minimum(min_distance, torch.amin(distances, dim=(1, 2)))
    return min_distance


def bimanual_tool_pointcloud_min_distance(
    env: "ManagerBasedRLEnv",
    num_points: int = 128,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Approximate tool-tool clearance by the nearest pair in sampled tool point clouds."""

    robot1 = env.scene[robot1_cfg.name]
    robot2 = env.scene[robot2_cfg.name]

    def compute():
        body1 = robot1.data.body_state_w[:, robot1_cfg.body_ids[0], :]
        body2 = robot2.data.body_state_w[:, robot2_cfg.body_ids[0], :]
        points1_w = _transform_tool_points_w(
            body1,
            _tool_local_distance_points(env, tool_slot=1, num_points=int(num_points)),
        )
        points2_w = _transform_tool_points_w(
            body2,
            _tool_local_distance_points(env, tool_slot=2, num_points=int(num_points)),
        )
        return _batched_pointcloud_min_distance(points1_w, points2_w)

    return get_or_compute_step_value(
        env,
        (
            "bimanual_tool_pointcloud_min_distance",
            int(num_points),
            id(robot1),
            robot1_cfg.body_ids[0],
            id(robot2),
            robot2_cfg.body_ids[0],
        ),
        compute,
    )


def bimanual_tool_proximity_penalty(
    env: "ManagerBasedRLEnv",
    warning_clearance: float = 0.02,
    contact_clearance: float = 0.005,
    num_points: int = 128,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Penalty that rises when sampled tool surfaces get too close."""

    warning_clearance = float(warning_clearance)
    contact_clearance = float(contact_clearance)
    if warning_clearance <= contact_clearance:
        raise ValueError("warning_clearance must be greater than contact_clearance")

    clearance = bimanual_tool_pointcloud_min_distance(
        env,
        num_points=int(num_points),
        robot1_cfg=robot1_cfg,
        robot2_cfg=robot2_cfg,
    )
    normalized_violation = torch.clamp(
        (warning_clearance - clearance) / (warning_clearance - contact_clearance),
        min=0.0,
    )
    return torch.square(normalized_violation)


def bimanual_wrist_surface_proximity_penalty(
    env: "ManagerBasedRLEnv",
    surface_z: float = 0.0,
    warning_height: float = 0.12,
    contact_height: float = 0.06,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Penalty for bringing bimanual wrist links close to the table/ground surface."""

    warning_height = float(warning_height)
    contact_height = float(contact_height)
    if warning_height <= contact_height:
        raise ValueError("warning_height must be greater than contact_height")

    min_clearance = None
    for cfg in (robot1_cfg, robot2_cfg):
        robot = env.scene[cfg.name]
        wrist_z = robot.data.body_pos_w[:, cfg.body_ids, 2]
        clearance = torch.amin(wrist_z - float(surface_z), dim=1)
        min_clearance = clearance if min_clearance is None else torch.minimum(min_clearance, clearance)

    normalized_violation = torch.clamp(
        (warning_height - min_clearance) / (warning_height - contact_height),
        min=0.0,
    )
    return torch.square(normalized_violation)

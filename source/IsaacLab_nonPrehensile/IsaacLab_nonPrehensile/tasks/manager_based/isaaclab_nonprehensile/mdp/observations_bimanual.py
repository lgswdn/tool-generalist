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

from .observations import (
    _HAND_GOAL_MEAN,
    _HAND_GOAL_STD,
    _bbox_center_env,
    _dbg,
    _dbg_cloud,
    get_object_pointcloud_in_env_frame,
    object_root_velocity,
    phys_params,
    profile_obs,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TOOL_BODY_NAME = "link_coacd_convex_piece_0"


def _bimanual_runtime():
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import env_tool_bimanual_unstable as env_mod

    return env_mod


def _tool_assignment(tool_slot: int):
    env_mod = _bimanual_runtime()
    if int(tool_slot) == 1:
        return env_mod.get_tool1_index_for_env, env_mod.get_tool1_data_for_env
    if int(tool_slot) == 2:
        return env_mod.get_tool2_index_for_env, env_mod.get_tool2_data_for_env
    raise ValueError(f"tool_slot must be 1 or 2, got {tool_slot!r}")


def _tool_body_pose_w(
    env: "ManagerBasedRLEnv",
    *,
    robot_name: str,
    cache_attr: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene[robot_name]
    if not hasattr(env, cache_attr):
        cfg = SceneEntityCfg(robot_name, body_names=[TOOL_BODY_NAME])
        cfg.resolve(env.scene)
        setattr(env, cache_attr, cfg)
    body_cfg = getattr(env, cache_attr)
    body_idx = body_cfg.body_ids[0]
    body_state = robot.data.body_state_w[:, body_idx, :]
    return body_state[:, :3], body_state[:, 3:7]


def _tool_local_points(td: dict, device: torch.device) -> torch.Tensor:
    env_mod = _bimanual_runtime()
    tool_cloud = env_mod.get_cached_cloud(td["obj_path"])
    base_pts = tool_cloud._get_points_torch(device).float()
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


def _tool_env_groups(
    env: "ManagerBasedRLEnv",
    *,
    tool_slot: int,
    get_tool_index_for_env,
) -> dict[int, torch.Tensor]:
    cache_attr = f"_tool{tool_slot}_env_groups"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    groups: dict[int, list[int]] = {}
    for env_id in range(env.num_envs):
        tidx = get_tool_index_for_env(env_id)
        groups.setdefault(tidx, []).append(env_id)

    device_groups = {
        tidx: torch.tensor(env_ids, device=env.device, dtype=torch.long)
        for tidx, env_ids in groups.items()
    }
    setattr(env, cache_attr, device_groups)
    return device_groups


def _tool_local_points_cached(
    env: "ManagerBasedRLEnv",
    *,
    tool_slot: int,
    tool_idx: int,
    tool_data: dict,
) -> torch.Tensor:
    cache_attr = f"_tool{tool_slot}_local_points_cache"
    cache = getattr(env, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(env, cache_attr, cache)
    cached = cache.get(tool_idx)
    if cached is None:
        cached = _tool_local_points(tool_data, env.device)
        cache[tool_idx] = cached
    return cached


def get_tool_pointcloud_for_slot(
    env: "ManagerBasedRLEnv",
    *,
    tool_slot: int,
    robot_name: str,
    bbox_cache_attr: str,
    body_cache_attr: str,
) -> torch.Tensor:
    """Return one bimanual tool point cloud in env-frame, flattened."""

    get_tool_index_for_env, get_tool_data_for_env = _tool_assignment(tool_slot)
    num_envs = env.num_envs
    device = env.device
    tool_pos_w, tool_quat_w = _tool_body_pose_w(env, robot_name=robot_name, cache_attr=body_cache_attr)

    out_tensor = None
    tool_to_envs = _tool_env_groups(
        env,
        tool_slot=tool_slot,
        get_tool_index_for_env=get_tool_index_for_env,
    )

    for tidx, env_indices in tool_to_envs.items():
        td = get_tool_data_for_env(int(env_indices[0].item()))
        pts = _tool_local_points_cached(env, tool_slot=tool_slot, tool_idx=tidx, tool_data=td)

        batch_quat = tool_quat_w[env_indices].contiguous()
        batch_pos = tool_pos_w[env_indices].contiguous()
        batch_rot = matrix_from_quat(batch_quat)
        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(env_indices.numel(), -1, -1),
        ).transpose(1, 2)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        if out_tensor is None:
            out_tensor = torch.empty((num_envs, pts_world.shape[1], 3), device=device, dtype=pts_world.dtype)
        out_tensor[env_indices] = pts_world

    if out_tensor is None:
        raise RuntimeError(f"No point cloud output was produced for tool slot {tool_slot}")

    pointcloud_env = out_tensor - env.scene.env_origins.unsqueeze(1)
    setattr(env, bbox_cache_attr, _bbox_center_env(pointcloud_env).detach())

    if getattr(env.cfg, f"visualize_tool{tool_slot}_pointcloud", False):
        _dbg_cloud(env, f"tool{tool_slot}_cloud", pointcloud_env)

    return pointcloud_env.reshape(num_envs, -1)


@profile_obs
def get_tool1_pointcloud_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return get_tool_pointcloud_for_slot(
        env,
        tool_slot=1,
        robot_name="robot_1",
        bbox_cache_attr="_obs_tool1_bbox_center",
        body_cache_attr="_tool1_body_cfg_resolved",
    )


@profile_obs
def get_tool2_pointcloud_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return get_tool_pointcloud_for_slot(
        env,
        tool_slot=2,
        robot_name="robot_2",
        bbox_cache_attr="_obs_tool2_bbox_center",
        body_cache_attr="_tool2_body_cfg_resolved",
    )


def get_tool1_bbox_center(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_obs_tool1_bbox_center"):
        return env._obs_tool1_bbox_center
    get_tool1_pointcloud_in_env_frame(env)
    return env._obs_tool1_bbox_center


def get_tool2_bbox_center(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_obs_tool2_bbox_center"):
        return env._obs_tool2_bbox_center
    get_tool2_pointcloud_in_env_frame(env)
    return env._obs_tool2_bbox_center


def compute_head_area_offsets_for_slot(env: "ManagerBasedRLEnv", *, tool_slot: int) -> torch.Tensor:
    """Compute per-env local head-area offsets for one bimanual tool slot."""

    env_mod = _bimanual_runtime()
    get_tool_index_for_env, get_tool_data_for_env = _tool_assignment(tool_slot)
    head_area_offsets = torch.zeros(env.num_envs, 3, device=env.device)
    per_tool_offset_cache: dict[int, torch.Tensor] = {}

    for env_id in range(env.num_envs):
        tool_idx = get_tool_index_for_env(env_id)
        if tool_idx in per_tool_offset_cache:
            head_area_offsets[env_id] = per_tool_offset_cache[tool_idx]
            continue

        td = get_tool_data_for_env(env_id)
        head_area_norm = td.get("head_area")
        if head_area_norm is None:
            print(f"[WARNING bimanual_head_area] No head_area for tool '{td['name']}', offset=[0,0,0]")
            per_tool_offset_cache[tool_idx] = torch.zeros(3, device=env.device)
            continue

        mid_norm = [(head_area_norm[0][i] + head_area_norm[1][i]) / 2.0 for i in range(3)]
        cloud = env_mod.get_cached_cloud(td["obj_path"])
        pts = torch.tensor(cloud.points, dtype=torch.float32, device=env.device)
        bbox_min = pts.min(dim=0).values
        bbox_max = pts.max(dim=0).values
        mid_norm_t = torch.tensor(mid_norm, dtype=torch.float32, device=env.device)
        head_area_unscaled = bbox_min + mid_norm_t * (bbox_max - bbox_min)
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, dtype=torch.float32, device=env.device)
            body_origin = bbox_min + bc * (bbox_max - bbox_min)
            head_area_local = (head_area_unscaled - body_origin) * float(env_mod.TOOL_SCALE)
        else:
            head_area_from_attachment = head_area_unscaled.clone()
            head_area_from_attachment[2] = head_area_unscaled[2] - bbox_min[2]
            head_area_local = head_area_from_attachment * float(env_mod.TOOL_SCALE)

        per_tool_offset_cache[tool_idx] = head_area_local
        head_area_offsets[env_id] = head_area_local

    return head_area_offsets


def get_head_area_pos_w_for_slot(
    env: "ManagerBasedRLEnv",
    *,
    ee_frame_name: str,
    offsets_attr: str,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_name]
    tool_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    offsets = getattr(env, offsets_attr, None)
    if offsets is None:
        return tool_pos_w
    tool_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    rot = matrix_from_quat(tool_quat_w)
    return tool_pos_w + torch.bmm(rot, offsets.unsqueeze(-1)).squeeze(-1)


def hand_state_for_slot(
    env: "ManagerBasedRLEnv",
    *,
    ee_frame_name: str,
    offsets_attr: str,
    debug_name: str,
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_name]
    ee_pos_w = get_head_area_pos_w_for_slot(env, ee_frame_name=ee_frame_name, offsets_attr=offsets_attr)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_env = ee_pos_w - env.scene.env_origins
    rot_matrix = matrix_from_quat(ee_quat_w)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    hand_state_9d = torch.cat([ee_pos_env, rot_6d], dim=1)
    if getattr(env.cfg, "normalize_observations", True):
        mean = _HAND_GOAL_MEAN.to(hand_state_9d.device).view(1, 9)
        std = _HAND_GOAL_STD.to(hand_state_9d.device).view(1, 9)
        hand_state_9d = (hand_state_9d - mean) / torch.clamp(std, min=1e-6)
    return _dbg(env, debug_name, hand_state_9d)


@profile_obs
def hand1_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hand_state_for_slot(env, ee_frame_name="ee_frame_1", offsets_attr="_head_area_offsets_1", debug_name="hand1_state")


@profile_obs
def hand2_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hand_state_for_slot(env, ee_frame_name="ee_frame_2", offsets_attr="_head_area_offsets_2", debug_name="hand2_state")


def robot_state_for_asset(
    env: "ManagerBasedRLEnv",
    *,
    asset_cfg: SceneEntityCfg,
    debug_name: str,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, :7]
    joint_vel = asset.data.joint_vel[:, :7]
    if getattr(env.cfg, "normalize_observations", True):
        default_pos = asset.data.default_joint_pos[:, :7]
        soft_limits = asset.data.soft_joint_pos_limits[:, :7, :]
        mins = soft_limits[..., 0]
        maxs = soft_limits[..., 1]
        half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
        pos_norm = torch.clamp((joint_pos - default_pos) / half_ranges, -1.0, 1.0)
        vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, :7], min=1e-6)
        vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
        vel_norm = (vel_norm + 1.0) * 0.5
        return _dbg(env, debug_name, torch.cat([pos_norm, vel_norm], dim=1))
    return _dbg(env, debug_name, torch.cat([joint_pos, joint_vel], dim=1))


@profile_obs
def robot1_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return robot_state_for_asset(env, asset_cfg=SceneEntityCfg("robot_1"), debug_name="robot1_state")


@profile_obs
def robot2_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return robot_state_for_asset(env, asset_cfg=SceneEntityCfg("robot_2"), debug_name="robot2_state")


def bimanual_phys_params(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    field_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Physics observation for the bimanual task.

    The existing schema has one set of `tool_*` fields. For the first bimanual
    version we bind those fields to robot_1 to preserve the runtime-spec shape.
    Tool-specific per-arm physics can be added later with new explicit field
    names.
    """

    return phys_params(
        env,
        object_cfg=object_cfg,
        hand_cfg=SceneEntityCfg("robot_1"),
        field_names=field_names,
    )

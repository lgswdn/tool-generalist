# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from scipy.spatial.transform import Rotation as R
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp

# Lightweight profiling utilities for observation functions
import time
from functools import wraps
import torch

def _ensure_obs_timers(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_obs_timers"):
        env._obs_timers = {}
    return env._obs_timers

def profile_obs(fn):
    @wraps(fn)
    def wrapper(env, *args, **kwargs):
        timers = _ensure_obs_timers(env)
        name = fn.__name__
        t0 = time.perf_counter()
        result = fn(env, *args, **kwargs)
        dt = time.perf_counter() - t0
        entry = timers.get(name)
        if entry is None:
            timers[name] = {"time": dt, "count": 1}
        else:
            entry["time"] += dt
            entry["count"] += 1
        return result
    return wrapper

def print_obs_timers(env: "ManagerBasedRLEnv") -> None:
    timers = getattr(env, "_obs_timers", {})
    if not timers:
        print("[obs timers] no data collected yet")
        return
    print("[obs timers] summary:")
    for name, entry in timers.items():
        total = entry["time"]
        count = entry["count"]
        avg = total / count if count > 0 else 0.0
        print(f"  {name}: total={total:.6f}s count={count} avg={avg:.6f}s")

# Debug: print observation stats every N calls
DEBUG_OBS_EVERY = 10000000

_HAND_GOAL_MEAN = torch.tensor([0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # z mean = 0.15
_HAND_GOAL_STD = torch.tensor([0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

def _dbg(env: "ManagerBasedRLEnv", name: str, tensor: torch.Tensor) -> torch.Tensor:
    cnt = getattr(env, "_dbg_obs_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        t = tensor
        if t.dim() >= 2:
            mins = t.min(dim=0).values
            maxs = t.max(dim=0).values
            print(f"[obs] {name}: min={mins.tolist()} max={maxs.tolist()} shape={tuple(t.shape)}")
        else:
            print(f"[obs] {name}: min={t.min().item():.3f}, max={t.max().item():.3f}, shape={tuple(t.shape)}")
    env._dbg_obs_cnt = cnt + 1
    return tensor


def _dbg_cloud(env: "ManagerBasedRLEnv", name: str, cloud_env: torch.Tensor) -> None:
    cnt = getattr(env, "_dbg_cloud_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        # cloud_env shape: (num_envs, num_points, 3)
        x = cloud_env[..., 0]
        y = cloud_env[..., 1]
        z = cloud_env[..., 2]
        print(
            f"[obs] {name}: x[min={x.min().item():.3f}, max={x.max().item():.3f}] "
            f"y[min={y.min().item():.3f}, max={y.max().item():.3f}] "
            f"z[min={z.min().item():.3f}, max={z.max().item():.3f}], shape={tuple(cloud_env.shape)}"
        )
    env._dbg_cloud_cnt = cnt + 1


def _bbox_center_env(pointcloud_env: torch.Tensor) -> torch.Tensor:
    bbox_min = pointcloud_env.min(dim=1).values
    bbox_max = pointcloud_env.max(dim=1).values
    return (bbox_min + bbox_max) * 0.5


def get_head_area_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the head area center in world space for every environment.

    Uses the tool body (link_coacd_convex_piece_0) pose from the ee_frame sensor
    plus the per-env local offset stored in env._head_area_offsets (computed once
    in post_reset via OBJ bounding-box queries).

    Returns:
        torch.Tensor: Shape (num_envs, 3) – world-space positions.
    """
    ee_frame = env.scene["ee_frame"]
    tool_pos_w = ee_frame.data.target_pos_w[..., 0, :]   # (N, 3)

    if not (hasattr(env, "_head_area_offsets") and env._head_area_offsets is not None):
        return tool_pos_w

    tool_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (N, 4)
    R = matrix_from_quat(tool_quat_w)                      # (N, 3, 3)
    offset = env._head_area_offsets                         # (N, 3)
    head_pos_w = tool_pos_w + torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)

    # --- Debug visualization: red sphere at head area center ---
    _visualize_head_area_center(env, head_pos_w)

    return head_pos_w


def _visualize_head_area_center(env: "ManagerBasedRLEnv", head_pos_w: torch.Tensor):
    """Draw a small red sphere at the head area center for each environment."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    if not hasattr(env, "_head_area_visualizer"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/HeadAreaCenter",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        env._head_area_visualizer = VisualizationMarkers(marker_cfg)

    num_envs = head_pos_w.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=head_pos_w.device).expand(num_envs, -1)
    env._head_area_visualizer.visualize(head_pos_w, orientations)


@profile_obs
def hand_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hand state observation (9D: position[3] + rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9) containing [x,y,z, r11,r12,r13, r21,r22,r23]
    """
    ee_frame = env.scene[ee_frame_cfg.name]

    # Head area world position (tool body + rotated per-env local offset)
    ee_pos_w = get_head_area_pos_w(env)          # (num_envs, 3)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    # Convert to environment coordinates
    ee_pos_env = ee_pos_w - env.scene.env_origins
    
    # Rotation as 6D
    rot_matrix = matrix_from_quat(ee_quat_w)  # (num_envs, 3, 3)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    hand_state_9d = torch.cat([ee_pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters (similar to corn config)
        device = hand_state_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        hand_state_9d = (hand_state_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "hand_state", hand_state_9d)


@profile_obs
def robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot state observation (14D: joint_positions[7] + joint_velocities[7]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 14)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get joint positions and velocities
    joint_pos = asset.data.joint_pos[:, :7]
    joint_vel = asset.data.joint_vel[:, :7]
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Normalize joint positions using soft limits around default pos -> [-1,1]
        default_pos = asset.data.default_joint_pos[:, :7]
        soft_limits = asset.data.soft_joint_pos_limits[:, :7, :]
        mins = soft_limits[..., 0]
        maxs = soft_limits[..., 1]
        centers = default_pos
        half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
        pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

        # Normalize joint velocities to [0,1] using soft velocity limits
        vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, :7], min=1e-6)
        vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
        vel_norm = (vel_norm + 1.0) * 0.5
        
        return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))
    else:
        # Return raw joint states without normalization
        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))


@profile_obs
def abs_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
) -> torch.Tensor:
    """Absolute pose goal observation (9D: target position[3] + target rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9)
    """
    from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
    
    target_goal = env.command_manager.get_command(command_name)
    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    # Command now directly contains quaternions, convert to rotation matrix
    rot_matrix = matrix_from_quat(target_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    goal_9d = torch.cat([target_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use goal-specific normalization parameters (similar to corn config)
        device = goal_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        goal_9d = (goal_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "abs_goal", goal_9d)


def rel_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Relative pose goal observation (9D: goal relative to current object pose)."""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate, matrix_from_quat

    target_goal = env.command_manager.get_command(command_name)  # (num_envs, 7)
    object_pose_7d = object_pose_in_env_frame(env, object_cfg, normalize=False)
    obj_pos_env = object_pose_7d[:, :3]
    obj_quat_w = object_pose_7d[:, 3:7]

    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    rel_pos = target_pos - obj_pos_env
    current_quat_inv = quat_conjugate(obj_quat_w)
    rel_quat = quat_mul(target_quat, current_quat_inv)
    rot_matrix = matrix_from_quat(rel_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine relative position and rotation
    rel_pose_9d = torch.cat([rel_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for relative pose goal
        device = rel_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        rel_pose_9d = (rel_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return rel_pose_9d


PHYS_PARAM_FIELD_NAMES = (
    "object_mass",
    "object_static_friction",
    "object_dynamic_friction",
    "object_restitution",
    "tool_mass",
    "tool_static_friction",
    "tool_dynamic_friction",
    "tool_restitution",
    "ground_static_friction",
    "ground_dynamic_friction",
    "ground_restitution",
    "table_static_friction",
    "table_dynamic_friction",
    "table_restitution",
)


@profile_obs
def phys_params(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    field_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Physical parameter observation assembled from explicit sampled fields."""
    device = env.scene[object_cfg.name].data.root_pos_w.device
    object: RigidObject = env.scene[object_cfg.name]
    hand: RigidObject = env.scene[hand_cfg.name]
    if field_names is None:
        field_names = getattr(env.cfg, "physics_observation_fields", ())
    field_names = tuple(field_names)
    env._phys_param_field_names = field_names

    # 1. Get object mass from IsaacLab's built-in interface
    object_mass = object.root_physx_view.get_masses().squeeze(-1)  # Shape: (num_envs,)

    # 2. Get object material properties from PhysX view
    # Material properties format: [static_friction, dynamic_friction, restitution]
    object_material_props = object.root_physx_view.get_material_properties()  # Shape: (num_envs, num_bodies, 3)
    object_static_friction = object_material_props[:, :, 0].mean(dim=1)
    object_dynamic_friction = object_material_props[:, :, 1].mean(dim=1)
    object_restitution = object_material_props[:, :, 2].mean(dim=1)

    # 3. Get tool mass and friction from robot articulation
    # Resolve tool body index (link_coacd_convex_piece_0)
    if not hasattr(env, "_tool_body_idx"):
        tool_body_cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        tool_body_cfg.resolve(env.scene)
        env._tool_body_idx = tool_body_cfg.body_ids[0]

    tool_idx = env._tool_body_idx
    robot_masses = hand.root_physx_view.get_masses()  # (num_envs, num_bodies)
    tool_mass = robot_masses[:, tool_idx]  # (num_envs,)

    # Tool friction from robot's material properties
    robot_material_props = hand.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3)
    num_shapes = robot_material_props.shape[1]
    num_bodies = robot_masses.shape[1]
    shapes_per_body = num_shapes // num_bodies
    tool_shape_start = tool_idx * shapes_per_body
    tool_shape_end = min((tool_idx + 1) * shapes_per_body, num_shapes)
    tool_material = robot_material_props[:, tool_shape_start:tool_shape_end, :]
    tool_static_friction = tool_material[:, :, 0].mean(dim=1)
    tool_dynamic_friction = tool_material[:, :, 1].mean(dim=1)
    tool_restitution = tool_material[:, :, 2].mean(dim=1)

    # 4. Get ground/terrain material only when runtime spec requests ground fields.
    ground_static_value = None
    ground_dynamic_value = None
    ground_restitution_value = None
    ground_fields_requested = any(name.startswith("ground_") for name in field_names)
    if ground_fields_requested:
        if bool(getattr(env.cfg, "table_enabled", False)):
            raise ValueError(
                "phys_params requested ground_* fields, but table is enabled and "
                "the ground terrain is not part of the scene."
            )
        try:
            terrain = env.scene["terrain"]
        except Exception as exc:
            raise ValueError(
                "phys_params requested ground_* fields, but scene terrain is unavailable."
            ) from exc
        terrain_prim_path = terrain.cfg.prim_path + "/terrain"
        physics_material_path = f"{terrain_prim_path}/physicsMaterial"

        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdPhysics

        physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)
        if (
            physics_material_prim
            and physics_material_prim.IsValid()
            and physics_material_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            physics_material = UsdPhysics.MaterialAPI(physics_material_prim)
            ground_static_value = physics_material.GetStaticFrictionAttr().Get()
            ground_dynamic_value = physics_material.GetDynamicFrictionAttr().Get()
            ground_restitution_value = physics_material.GetRestitutionAttr().Get()
    ground_static_friction = torch.full_like(object_mass, 1.0 if ground_static_value is None else float(ground_static_value))
    ground_dynamic_friction = torch.full_like(object_mass, 1.0 if ground_dynamic_value is None else float(ground_dynamic_value))
    ground_restitution = torch.full_like(object_mass, 0.0 if ground_restitution_value is None else float(ground_restitution_value))

    # 5. Table material fields are configured explicitly by RLCfg.  If a future
    # table material randomizer stores sampled values on env, those values take
    # precedence over the config defaults here.
    table_material = getattr(env.cfg, "table_material", None)
    sampled_table = getattr(env, "_sampled_table_material", None)
    table_static_default = getattr(table_material, "static_friction", 0.8)
    table_dynamic_default = getattr(table_material, "dynamic_friction", 0.8)
    table_restitution_default = getattr(table_material, "restitution", 0.0)
    table_static_friction = torch.full_like(
        object_mass,
        float(getattr(sampled_table, "static_friction", table_static_default)),
    )
    table_dynamic_friction = torch.full_like(
        object_mass,
        float(getattr(sampled_table, "dynamic_friction", table_dynamic_default)),
    )
    table_restitution = torch.full_like(
        object_mass,
        float(getattr(sampled_table, "restitution", table_restitution_default)),
    )

    values = {
        "object_mass": object_mass.to(device=device),
        "object_static_friction": object_static_friction.to(device=device),
        "object_dynamic_friction": object_dynamic_friction.to(device=device),
        "object_restitution": object_restitution.to(device=device),
        "tool_mass": tool_mass.to(device=device),
        "tool_static_friction": tool_static_friction.to(device=device),
        "tool_dynamic_friction": tool_dynamic_friction.to(device=device),
        "tool_restitution": tool_restitution.to(device=device),
        "ground_static_friction": ground_static_friction.to(device=device),
        "ground_dynamic_friction": ground_dynamic_friction.to(device=device),
        "ground_restitution": ground_restitution.to(device=device),
        "table_static_friction": table_static_friction.to(device=device),
        "table_dynamic_friction": table_dynamic_friction.to(device=device),
        "table_restitution": table_restitution.to(device=device),
    }
    missing = [name for name in field_names if name not in values]
    if missing:
        raise ValueError(f"Unknown phys_params fields: {missing}")

    if not field_names:
        phys_params_tensor = torch.empty((env.num_envs, 0), device=device)
    else:
        phys_params_tensor = torch.stack([values[name] for name in field_names], dim=1)

    return _dbg(env, "phys_params", phys_params_tensor)


def object_pose_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    normalize: bool | None = None,
) -> torch.Tensor:
    """The pose of the object in the environment coordinate frame.

    Returns:
        torch.Tensor: Shape (num_envs, 7) containing [x, y, z, qw, qx, qy, qz] in environment coordinates
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_quat_w = object.data.root_quat_w       # (num_envs, 4)
    object_pos_env = object_pos_w - env.scene.env_origins  # (num_envs, 3)
    pose_7d = torch.cat([object_pos_env, object_quat_w], dim=1)  # (num_envs, 7)

    if env.cfg.visualize_current_object_pose:
        visualize_object_pose_in_env(env, pose_7d)

    # Check normalization setting from environment config unless an internal
    # caller needs the raw pose for pose arithmetic.
    if normalize is None:
        normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for object pose
        device = pose_7d.device
        # For 7D pose: position [x,y,z] + quaternion [qw,qx,qy,qz]
        # Use position normalization from hand_goal params
        pos_mean = _HAND_GOAL_MEAN[:3].to(device).view(1, 3)  # [x, y, z] mean
        pos_std = _HAND_GOAL_STD[:3].to(device).view(1, 3)    # [x, y, z] std
        # For quaternion, use simple normalization (quaternions are already normalized)
        quat_mean = torch.zeros(4, device=device).view(1, 4)  # [qw, qx, qy, qz] mean
        quat_std = torch.ones(4, device=device).view(1, 4)    # [qw, qx, qy, qz] std
        
        # Normalize position and quaternion separately
        pos_norm = (pose_7d[:, :3] - pos_mean) / torch.clamp(pos_std, min=1e-6)
        quat_norm = (pose_7d[:, 3:7] - quat_mean) / torch.clamp(quat_std, min=1e-6)
        
        pose_7d = torch.cat([pos_norm, quat_norm], dim=1)
        
    return pose_7d


def object_pose_9d_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The object's current pose in the environment frame as 9D [x,y,z, r11,r12,r13, r21,r22,r23]."""
    pose_7d = object_pose_in_env_frame(env, object_cfg)
    pos_env = pose_7d[:, :3]
    quat_wxyz = pose_7d[:, 3:7]

    # Convert to rotation matrix
    rot_matrix = matrix_from_quat(quat_wxyz)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    object_pose_9d = torch.cat([pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand/goal-specific normalization parameters for object pose too
        device = object_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        object_pose_9d = (object_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "cur_pose", object_pose_9d)


def visualize_object_pose_in_env(
    env: ManagerBasedRLEnv,
    object_pose_7d: torch.Tensor,
    marker_scale: tuple = (0.08, 0.08, 0.08),
) -> None:
    """Visualize the object's current pose in environment coordinates.
    
    This function creates visualization markers to show the object's current pose
    in the environment coordinate frame, using the same approach as the target pose visualization.
    """
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import quat_from_euler_xyz
    
    # Create visualization markers if they don't exist (similar to target pose visualization)
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"  # Different path from target pose
        marker_cfg.markers["frame"].scale = marker_scale  # Make frames visible but distinct
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)

    # Extract position and quaternion (same as target pose visualization)
    local_positions = object_pose_7d[:, :3]  # (num_envs, 3)
    quaternions = object_pose_7d[:, 3:7]  # (num_envs, 4)

    # Convert local positions to world positions by adding environment origins (same as target)
    world_positions = local_positions + env.scene.env_origins
    
    # Visualize current pose frames using world positions (same method as target)
    env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def create_object_pose_visualizer(env: ManagerBasedRLEnv, marker_scale: tuple = (0.08, 0.08, 0.08)) -> None:
    """Initialize the object pose visualizer. Call this once during environment setup."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = marker_scale
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)


def update_object_pose_visualization(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> None:
    """Update the object pose visualization. Call this during environment step/reset."""
    from isaaclab.utils.math import quat_from_euler_xyz
    
    if hasattr(env, '_current_object_pose_visualizer'):
        # Get object pose in environment frame
        object_pose_7d = object_pose_in_env_frame(env, object_cfg)
        
        # Extract position and euler angles
        local_positions = object_pose_7d[:, :3]
        world_positions = local_positions + env.scene.env_origins
        
        quaternions = object_pose_7d[:, 3:7]
        
        # Update visualization (same method as target pose)
        env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def object_pose_with_visualization(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object pose in environment frame and update visualization as a side effect.
    
    This function serves dual purpose:
    1. Returns the object pose for observations
    2. Triggers visualization update each time it's called
    
    Returns:
        torch.Tensor: Shape (num_envs, 6) containing [x, y, z, roll, pitch, yaw] in environment coordinates
    """
    # Initialize visualizer if not already done
    if not hasattr(env, '_current_object_pose_visualizer'):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)
    
    # Get object pose
    object_pose_7d = object_pose_in_env_frame(env, object_cfg)
    
    # Update visualization
    update_object_pose_visualization(env, object_cfg)
    
    return object_pose_7d


def visualize_object_pointcloud(
    env: ManagerBasedRLEnv,
    pointcloud_tensor: torch.Tensor,
    point_size: float = 0.005,
    color: tuple = (0.0, 1.0, 0.0),  # Green color for point cloud
) -> None:
    """Visualize the object's point cloud for debugging purposes.
    
    The point cloud is displayed in world coordinates, showing the actual 
    transformed points at the object's current position and orientation.
    
    Args:
        env: The RL environment
        pointcloud_tensor: Pre-computed point cloud tensor, shape (num_envs, num_points*3)
        point_size: Size of the visualization spheres
        color: RGB color tuple for the point cloud visualization
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils
    
    # Create visualization markers if they don't exist
    if not hasattr(env, '_pointcloud_visualizer'):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        
        env._pointcloud_visualizer = VisualizationMarkers(marker_cfg)
    
    # Reshape flattened point cloud back to (num_envs, num_points, 3) for visualization
    num_envs = pointcloud_tensor.shape[0]
    points_per_env = pointcloud_tensor.shape[1] // 3
    pointcloud_reshaped = pointcloud_tensor.view(num_envs, points_per_env, 3)
    
    # Flatten all envs into a single set of points for visualization
    all_points = pointcloud_reshaped.reshape(-1, 3)  # Shape: (num_envs * num_points, 3)
    
    # Create identity quaternions for all points (spheres don't need rotation)
    total_points = all_points.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=all_points.device).expand(total_points, -1)
    
    # Visualize the points in world coordinates
    env._pointcloud_visualizer.visualize(
        translations=all_points,
        orientations=orientations
    )


@profile_obs
def get_object_pointcloud(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud in world coordinates.

    Uses per-env Cloud instances with baked-in scale and pose caching.

    Returns:
        torch.Tensor: shape (num_envs, num_points*3) in world coordinates.
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Initialize per-env Cloud instances on first call
    if not hasattr(env, "_object_clouds"):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_object_asset_cfg_for_env,
        )

        num_envs = object.data.root_pos_w.shape[0]
        device = object.data.root_pos_w.device
        scales = mdp.get_rigid_body_scale(
            env, SceneEntityCfg("object"), list(range(num_envs))
        )

        env._object_clouds = []
        for env_idx in range(num_envs):
            obj_path = get_object_asset_cfg_for_env(env_idx).obj_path
            initial_scale = scales[env_idx].detach().cpu().numpy()
            cloud = Cloud(
                obj_path,
                target_num_points=512,
                device=device,
                dtype=torch.float16,
                initial_scale=initial_scale,
                trans_cache_threshold=0.01,
                rot_cache_threshold=0.01,
            )
            env._object_clouds.append(cloud)

        print(
            f"[INFO] Initialized {num_envs} Cloud instances for objects on device: {device}"
        )

    device = object.data.root_pos_w.device
    num_envs = object.data.root_pos_w.shape[0]

    # Process each environment using its own Cloud instance
    pointclouds = []
    for env_idx in range(num_envs):
        cloud = env._object_clouds[env_idx]

        # Get pose for this environment
        pos_w = object.data.root_pos_w[env_idx : env_idx + 1, :3].contiguous()
        quat_w = object.data.root_quat_w[env_idx : env_idx + 1].contiguous()

        # Get pointcloud (will use cache if pose unchanged)
        pc = cloud.get_pointcloud(translation=pos_w, rotation=quat_w, use_cache=True)
        pointclouds.append(pc)

    # Stack all pointclouds and flatten
    all_pointclouds = torch.cat(pointclouds, dim=0).view(num_envs, -1)

    # Optional visualization for debugging
    if env.cfg.visualize_object_pointcloud:
        visualize_object_pointcloud(env, all_pointclouds.float())

    return all_pointclouds

@profile_obs
def get_object_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud in the environment frame.

    The point cloud remains in env-frame coordinates for the observation.  A
    separate bbox-center term is cached for the policy context.

    Returns:
        Tensor (num_envs, num_points*3) in env-frame coordinates.
    """
    pointcloud_w = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_w.shape
    num_points = flat_dim // 3
    pointcloud_w_reshaped = pointcloud_w.view(num_envs, num_points, 3)
    # Convert to env frame (subtract world env origin)
    pointcloud_env = pointcloud_w_reshaped - env.scene.env_origins.unsqueeze(1)  # (N, P, 3)

    obj_bbox_center = _bbox_center_env(pointcloud_env)
    env._obs_obj_bbox_center = obj_bbox_center.detach()

    return pointcloud_env.reshape(num_envs, num_points * 3)


def get_obj_bbox_center(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object cloud AABB center in the environment frame (3D per env).

    Must be listed AFTER ``object_cloud`` in the observation config so the
    bbox-center cache (``env._obs_obj_bbox_center``) is populated first.

    Returns:
        Tensor (num_envs, 3)
    """
    if hasattr(env, "_obs_obj_bbox_center"):
        return env._obs_obj_bbox_center
    # Fallback: recompute (should not happen in normal operation)
    pointcloud_w = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_w.shape
    num_points = flat_dim // 3
    pc_env = pointcloud_w.view(num_envs, num_points, 3) - env.scene.env_origins.unsqueeze(1)
    bbox_center = _bbox_center_env(pc_env)
    env._obs_obj_bbox_center = bbox_center.detach()
    return bbox_center


def visualize_tool_pointcloud(
    env: ManagerBasedRLEnv,
    pointcloud_tensor: torch.Tensor,
    point_size: float = 0.005,
    color: tuple = (0.0, 0.0, 1.0),  # Blue color for tool point cloud
) -> None:
    """Visualize the tool's point cloud for debugging purposes.

    The point cloud is displayed in world coordinates, showing the actual
    transformed points at the tool's current position and orientation.
    Only the first environment is visualized to avoid clutter.

    Args:
        env: The RL environment
        pointcloud_tensor: Pre-computed point cloud tensor, shape (num_envs, num_points*3)
        point_size: Size of the visualization spheres
        color: RGB color tuple for the point cloud visualization
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    # Create visualization markers if they don't exist
    if not hasattr(env, '_tool_pointcloud_visualizer'):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ToolPointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        env._tool_pointcloud_visualizer = VisualizationMarkers(marker_cfg)

    # Reshape flattened point cloud back to (num_envs, num_points, 3) for visualization
    num_envs = pointcloud_tensor.shape[0]
    points_per_env = pointcloud_tensor.shape[1] // 3
    pointcloud_reshaped = pointcloud_tensor.view(num_envs, points_per_env, 3)

    # Flatten all envs into a single set of points for visualization
    all_points = pointcloud_reshaped.reshape(-1, 3)  # Shape: (num_envs * num_points, 3)

    # Create identity quaternions for all points (spheres don't need rotation)
    total_points = all_points.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=all_points.device).expand(total_points, -1)

    # Visualize the points in world coordinates
    env._tool_pointcloud_visualizer.visualize(
        translations=all_points,
        orientations=orientations
    )


@profile_obs
def get_tool_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get tool point cloud in the environment frame.

    Each env may have a different tool (via MultiUsdFileCfg).  The canonical
    mesh is transformed using Cloud.get_pointcloud() with the tool body's
    world-frame pose and a uniform scale of TOOL_SCALE, then env_origins
    are subtracted to get env-frame coordinates.

    This mirrors the pattern used by get_object_pointcloud_in_env_frame.

    Returns:
        torch.Tensor: shape (num_envs, num_points*3), float32
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        from isaaclab.managers import SceneEntityCfg
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)

    out_tensor = None

    # Group envs by tool type for efficient batch processing
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for _tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])
        tool_cloud = get_cached_cloud(td["obj_path"])
        base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

        # Scale canonical points
        pts = base_pts * TOOL_SCALE  # (M, 3)

        # Compute the body-frame offset from base_center.
        # The USD body origin was placed at the base_center position (by adjust_meshes.py).
        # We must shift the point cloud by the same amount so it aligns with the body frame.
        pts = pts.clone()
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
            bbox_min_raw = base_pts.min(dim=0).values
            bbox_max_raw = base_pts.max(dim=0).values
            body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
            pts = pts - body_origin
        else:
            # Fallback: legacy Z-shift if base_center is not available
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Get per-env rotation matrices and positions
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        # Rotate points by body orientation and translate to body position
        # pts: (M, 3) -> (1, 3, M);  batch_rot: (B, 3, 3)
        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, M, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        # --- Debug: print key coordinates on first call ---
        if not hasattr(env, "_tool_pc_debug_done"):
            env._tool_pc_debug_done = True
            print(f"[DEBUG tool_pc] OBJ: {td['name']}")
            print(f"[DEBUG tool_pc] Canonical pts (scaled) min: {pts.min(dim=0).values.cpu().numpy()}")
            print(f"[DEBUG tool_pc] Canonical pts (scaled) max: {pts.max(dim=0).values.cpu().numpy()}")
            print(f"[DEBUG tool_pc] Canonical pts (scaled) mean: {pts.mean(dim=0).cpu().numpy()}")
            print(f"[DEBUG tool_pc] Body pos (world): {batch_pos[0].cpu().numpy()}")
            print(f"[DEBUG tool_pc] Body quat (world): {batch_quat[0].cpu().numpy()}")
            print(f"[DEBUG tool_pc] Rot matrix [2,:]: {batch_rot[0, 2, :].cpu().numpy()}")
            print(f"[DEBUG tool_pc] PC world min: {pts_world[0].min(dim=0).values.cpu().numpy()}")
            print(f"[DEBUG tool_pc] PC world max: {pts_world[0].max(dim=0).values.cpu().numpy()}")
            print(f"[DEBUG tool_pc] PC world mean: {pts_world[0].mean(dim=0).cpu().numpy()}")
            # Also print link7 position for reference
            link7_names = [n for n in robot.data.body_names if "link7" in n]
            if link7_names:
                l7_idx = list(robot.data.body_names).index(link7_names[0])
                l7_pos = robot.data.body_state_w[0, l7_idx, :3]
                print(f"[DEBUG tool_pc] panda_link7 pos (world): {l7_pos.cpu().numpy()}")
                print(f"[DEBUG tool_pc] body-link7 delta: {(batch_pos[0] - l7_pos).cpu().numpy()}")

        # Allocate output on first iteration
        if out_tensor is None:
            num_points = pts_world.shape[1]
            out_tensor = torch.empty(
                (num_envs, num_points, 3),
                device=device,
                dtype=pts_world.dtype,
            )

        out_tensor[env_indices] = pts_world

    # Convert to env frame: subtract env_origins
    pointcloud_w = out_tensor  # (num_envs, M, 3) in world frame
    pointcloud_env = pointcloud_w - env.scene.env_origins.unsqueeze(1)  # (N, M, 3)

    tool_bbox_center = _bbox_center_env(pointcloud_env)
    env._obs_tool_bbox_center = tool_bbox_center.detach()

    # Flatten to (num_envs, M*3)
    pointcloud_env_flat = pointcloud_env.reshape(num_envs, -1)

    # Optional visualization (world frame for marker rendering)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_tensor.reshape(num_envs, -1).float())

    return pointcloud_env_flat


def get_tool_bbox_center(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return the tool cloud AABB center in the environment frame.

    Must be listed AFTER ``tool_cloud`` in the observation config so the
    bbox-center cache (``env._obs_tool_bbox_center``) is populated first.

    Returns:
        Tensor (num_envs, 3)
    """
    if hasattr(env, "_obs_tool_bbox_center"):
        return env._obs_tool_bbox_center
    # Fallback: recompute by calling the full function
    get_tool_pointcloud_in_env_frame(env)
    return env._obs_tool_bbox_center


# ---------------------------------------------------------------------------
# 7D point cloud observations: position + mass + velocity
# ---------------------------------------------------------------------------

def _check_object_robot_contact(
    env: ManagerBasedRLEnv,
    pointcloud_w: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if point cloud is in contact with robot (any body/link).

    Returns:
        torch.Tensor: (num_envs,) bool mask, True if any point is within threshold of any robot body.
    """
    robot = env.scene[robot_cfg.name]
    body_positions = robot.data.body_state_w[:, :, :3]

    pointcloud_expanded = pointcloud_w.unsqueeze(2)
    body_positions_expanded = body_positions.unsqueeze(1)

    distances = torch.norm(pointcloud_expanded - body_positions_expanded, dim=3)
    min_distances_per_point = distances.min(dim=2)[0]
    in_contact = (min_distances_per_point < contact_threshold).any(dim=1)

    return in_contact


def visualize_pointcloud_velocity_mass(
    env: ManagerBasedRLEnv,
    features_flat: torch.Tensor,
    pointcloud_w_for_viz: torch.Tensor,
    env_idx: int = 0,
    subsample: int = 16,
    velocity_scale: float = 0.1,
    max_velocity: float = 1.0,
    point_size: float = 0.003,
    prefix: str = "object",
) -> None:
    """Visualize 7D point cloud features (position + mass + velocity).

    Creates colored sphere markers for mass and arrow markers for velocity vectors.

    Args:
        env: The RL environment
        features_flat: (num_envs, num_points * 7) flat features tensor
        pointcloud_w_for_viz: (num_envs, num_points, 3) world-frame positions for visualization
        env_idx: Which environment to visualize
        subsample: Show every Nth point to reduce clutter
        velocity_scale: Scale factor for velocity arrow length
        max_velocity: Maximum velocity for color normalization
        point_size: Radius of the point spheres
        prefix: Marker namespace prefix (e.g., "object", "tool")
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    num_envs = features_flat.shape[0]
    num_points = features_flat.shape[1] // 7
    feat = features_flat.view(num_envs, num_points, 7).float()

    # Extract all envs, subsampled
    sub_points = pointcloud_w_for_viz[:, ::subsample, :].reshape(-1, 3).float()  # (num_envs*K, 3)
    num_viz = sub_points.shape[0]

    # 1. Point cloud visualization (small green spheres)
    marker_name = f"_{prefix}_pc_visualizer"
    if not hasattr(env, marker_name):
        marker_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/{prefix.title()}PC7D",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
            },
        )
        setattr(env, marker_name, VisualizationMarkers(marker_cfg))

    orientations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=sub_points.device
    ).expand(num_viz, -1)
    getattr(env, marker_name).visualize(sub_points, orientations)


@profile_obs
def get_object_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud with mass and velocity (7D per point).

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.
    Velocity is gated by a contact check: only non-zero when the object is
    in contact with the robot.

    Returns:
        torch.Tensor: shape (num_envs, 512*7) = (num_envs, 3584), dtype float16
    """
    object: RigidObject = env.scene[object_cfg.name]

    pointcloud_flat = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_flat.shape
    num_points = flat_dim // 3

    pointcloud_w = pointcloud_flat.view(num_envs, num_points, 3)
    pointcloud_env = pointcloud_w - env.scene.env_origins.unsqueeze(1)

    # Contact-gated velocity: only compute when object touches robot
    in_contact = _check_object_robot_contact(
        env, pointcloud_w, robot_cfg=SceneEntityCfg("robot")
    )
    point_velocities = torch.zeros_like(pointcloud_env)

    if in_contact.any():
        contact_mask = in_contact.unsqueeze(1).unsqueeze(2)

        root_pos = object.data.root_pos_w[:, :3].unsqueeze(1)
        root_lin_vel = object.data.root_lin_vel_w[:, :3].unsqueeze(1)
        root_ang_vel = object.data.root_ang_vel_w[:, :3].unsqueeze(1)

        rel_positions = pointcloud_w - root_pos
        point_velocities_computed = root_lin_vel + torch.cross(
            root_ang_vel.expand_as(rel_positions), rel_positions, dim=-1
        )

        point_velocities = point_velocities_computed * contact_mask

    # Mass feature: distribute total object mass evenly across points
    masses = object.root_physx_view.get_masses().view(num_envs, -1)
    object_mass = masses.sum(dim=1)
    mass_per_point = (
        (object_mass / float(num_points))
        .view(num_envs, 1, 1)
        .to(pointcloud_env.device, pointcloud_env.dtype)
    )
    mass_feature = mass_per_point.expand(-1, num_points, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_velocities], dim=-1
    )  # (num_envs, num_points, 7)

    # Optional visualization for debugging
    if (
        hasattr(env.cfg, "visualize_object_velocity_mass")
        and env.cfg.visualize_object_velocity_mass
    ):
        pointcloud_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, num_points * 7),
            pointcloud_w_for_viz,
            env_idx=0,
            subsample=16,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.003,
            prefix="object",
        )

    return features.reshape(num_envs, num_points * 7).to(torch.float16)


@profile_obs
def get_tool_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get tool point cloud with mass and velocity (7D per point).

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.
    Velocity is derived from the tool body's rigid body kinematics.
    Uses the same adjusted decomposed mesh contract as get_tool_pointcloud_in_env_frame.

    Returns:
        torch.Tensor: shape (num_envs, 512*7) = (num_envs, 3584), dtype float16
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)
    tool_lin_vel_w = robot.data.body_lin_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)
    tool_ang_vel_w = robot.data.body_ang_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)

    out_world = None  # (num_envs, M, 3) in world frame

    # Group envs by tool type for efficient batch processing
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for _tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])
        tool_cloud = get_cached_cloud(td["obj_path"])
        base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

        # Scale canonical points
        pts = base_pts * TOOL_SCALE  # (M, 3)

        # Preserve existing body-frame alignment behavior.
        pts = pts.clone()
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
            bbox_min_raw = base_pts.min(dim=0).values
            bbox_max_raw = base_pts.max(dim=0).values
            body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
            pts = pts - body_origin
        else:
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Get per-env rotation matrices and positions
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        # Rotate points by body orientation and translate to body position
        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, M, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        # Allocate output on first iteration
        if out_world is None:
            num_points = pts_world.shape[1]
            out_world = torch.empty(
                (num_envs, num_points, 3),
                device=device,
                dtype=pts_world.dtype,
            )

        out_world[env_indices] = pts_world

    # Convert to env frame
    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)  # (B, M, 3)

    # Velocity: v_point = v_lin + w x (p_world - p_link)
    rel_pos = out_world - tool_pos_w.unsqueeze(1)  # (B, M, 3)
    ang_vel_expanded = tool_ang_vel_w.unsqueeze(1).expand_as(rel_pos)
    point_vels = tool_lin_vel_w.unsqueeze(1) + torch.cross(
        ang_vel_expanded, rel_pos, dim=-1
    )  # (B, M, 3)

    # Mass feature: get tool mass from robot articulation PhysX view
    if not hasattr(env, "_tool_body_idx"):
        _tb_cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _tb_cfg.resolve(env.scene)
        env._tool_body_idx = _tb_cfg.body_ids[0]
    robot_masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies) on CPU
    tool_mass = robot_masses[:, env._tool_body_idx].to(device=device, dtype=pointcloud_env.dtype)  # (num_envs,)
    total_pts = pointcloud_env.shape[1]
    mass_per_point = (tool_mass / float(total_pts)).view(num_envs, 1, 1)
    mass_feature = mass_per_point.expand(-1, total_pts, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_vels], dim=-1
    )  # (B, M, 7)

    # Optional visualization: full tool cloud (blue spheres, position only)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(num_envs, -1).float())

    # Optional visualization: 7D features (velocity + mass)
    if getattr(env.cfg, "visualize_tool_velocity_mass", False):
        pts_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, -1),
            pts_w_for_viz,
            env_idx=0,
            subsample=4,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.003,
            prefix="tool",
        )

    # Flatten and return
    return features.reshape(num_envs, -1).to(torch.float16)


def get_tool_head_area_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
    num_points: int = 512,
) -> torch.Tensor:
    """Get tool point cloud filtered to head area region, with mass and velocity (7D).

    Only includes points within the tool's head_area bounding box (the functional
    contact region of the tool). Points outside the head area are excluded.
    The remaining points are subsampled or padded to ``num_points``.

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.

    Args:
        env: The environment instance.
        num_points: Target number of points in the output cloud (default: 512).

    Returns:
        torch.Tensor: shape (num_envs, num_points*7), dtype float16
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)
    tool_lin_vel_w = robot.data.body_lin_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)
    tool_ang_vel_w = robot.data.body_ang_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)

    # Precompute per-tool head area masks on canonical points (cached)
    if not hasattr(env, "_tool_head_area_cache"):
        env._tool_head_area_cache = {}

    out_world = torch.empty((num_envs, num_points, 3), device=device, dtype=torch.float32)

    # Group envs by tool type
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])

        # Get or compute the head-area filtered canonical points for this tool
        if tidx not in env._tool_head_area_cache:
            tool_cloud = get_cached_cloud(td["obj_path"])
            base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

            # Scale canonical points
            pts_canonical = base_pts * TOOL_SCALE  # (M, 3)
            pts_canonical = pts_canonical.clone()
            base_center_norm = td.get("base_center")
            if base_center_norm is not None:
                bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
                bbox_min_raw = base_pts.min(dim=0).values
                bbox_max_raw = base_pts.max(dim=0).values
                body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
                pts_canonical = pts_canonical - body_origin
            else:
                pts_canonical[:, 2] = pts_canonical[:, 2] - pts_canonical[:, 2].min()

            head_area_norm = td.get("head_area")
            if head_area_norm is not None:
                # head_area_norm is [[x_min, y_min, z_min], [x_max, y_max, z_max]] in [0,1] normalized coords
                bbox_min = pts_canonical.min(dim=0).values
                bbox_max = pts_canonical.max(dim=0).values
                bbox_range = bbox_max - bbox_min

                # Convert normalized head area to absolute coordinates
                ha_min = bbox_min + torch.tensor(head_area_norm[0], device=device, dtype=torch.float32) * bbox_range
                ha_max = bbox_min + torch.tensor(head_area_norm[1], device=device, dtype=torch.float32) * bbox_range

                # Filter: keep points inside head area bounding box
                mask = (
                    (pts_canonical[:, 0] >= ha_min[0]) & (pts_canonical[:, 0] <= ha_max[0]) &
                    (pts_canonical[:, 1] >= ha_min[1]) & (pts_canonical[:, 1] <= ha_max[1]) &
                    (pts_canonical[:, 2] >= ha_min[2]) & (pts_canonical[:, 2] <= ha_max[2])
                )
                head_pts = pts_canonical[mask]

                if head_pts.shape[0] == 0:
                    # Fallback: if no points in head area, use all points
                    print(f"[WARNING] No points in head area for tool '{td['name']}', using full cloud")
                    head_pts = pts_canonical
            else:
                # No head_area defined, use all points
                print(f"[WARNING] No head_area for tool '{td['name']}', using full cloud as obstacle")
                head_pts = pts_canonical

            # Subsample or pad to target num_points
            n = head_pts.shape[0]
            if n >= num_points:
                # Uniform subsample
                idx = torch.linspace(0, n - 1, num_points, device=device).round().long()
                head_pts = head_pts[idx]
            else:
                # Repeat-pad
                repeats = (num_points + n - 1) // n
                head_pts = head_pts.repeat(repeats, 1)[:num_points]

            env._tool_head_area_cache[tidx] = head_pts  # (num_points, 3) canonical

        pts = env._tool_head_area_cache[tidx]  # (num_points, 3) canonical, already filtered

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Transform to world frame
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, num_points, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        out_world[env_indices] = pts_world

    # Convert to env frame
    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)  # (B, num_points, 3)

    # Velocity: v_point = v_lin + w x (p_world - p_link)
    rel_pos = out_world - tool_pos_w.unsqueeze(1)  # (B, num_points, 3)
    ang_vel_expanded = tool_ang_vel_w.unsqueeze(1).expand_as(rel_pos)
    point_vels = tool_lin_vel_w.unsqueeze(1) + torch.cross(
        ang_vel_expanded, rel_pos, dim=-1
    )  # (B, num_points, 3)

    # Mass feature
    if not hasattr(env, "_tool_body_idx"):
        _tb_cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _tb_cfg.resolve(env.scene)
        env._tool_body_idx = _tb_cfg.body_ids[0]
    robot_masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies) on CPU
    tool_mass = robot_masses[:, env._tool_body_idx].to(device=device, dtype=pointcloud_env.dtype)  # (num_envs,)
    mass_per_point = (tool_mass / float(num_points)).view(num_envs, 1, 1)
    mass_feature = mass_per_point.expand(-1, num_points, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_vels], dim=-1
    )  # (B, num_points, 7)

    # Optional visualization: head area cloud (blue spheres, position only)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(num_envs, -1).float())

    # Optional visualization: head area 7D features (orange spheres)
    if getattr(env.cfg, "visualize_tool_head_area", False):
        pts_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, -1),
            pts_w_for_viz,
            env_idx=0,
            subsample=4,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.004,
            prefix="tool_head_area",
        )

    return features.reshape(num_envs, -1).to(torch.float16)

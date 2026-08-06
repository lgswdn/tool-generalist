# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for non-prehensile manipulation environments."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Usd, UsdPhysics, Gf, UsdGeom

from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.table_placement import (
    surface_z_for_points,
    table_contract_from_env,
    table_top_z_from_contract,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def reset_initial_object_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object position using curriculum learning ranges.
    
    This function resets the object position within ranges that can be
    dynamically updated by the curriculum learning system through the command manager.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        asset_cfg: Asset configuration.
    """
    # Get the asset
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get curriculum ranges from command manager (preferred) or environment fallback
    stable_pose_term = env.command_manager.get_term("target_object_pose")
    xy_range = stable_pose_term.initial_position_range
    placement_cfg = table_contract_from_env(env)
    surface_z = (
        table_top_z_from_contract(placement_cfg, env.device)
        if bool(placement_cfg.enabled)
        else torch.as_tensor(0.0, dtype=torch.float32, device=env.device)
    )
    
    # Sample random positions within curriculum ranges
    num_resets = len(env_ids)
    # Create poses: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    poses = torch.zeros((num_resets, 13), device=env.device)
    
    # Per-env sampling from stable pose with random yaw offset
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_object_cloud,
        get_object_asset_cfg_for_env,
    )
    scales = get_rigid_body_scale(env, SceneEntityCfg("object"), env_ids)

    for i, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())

        obj_path = get_object_asset_cfg_for_env(env_id_int).obj_path
        scale_tensor = scales[i]
        scale = tuple(scale_tensor.cpu().numpy())

        object_cloud = get_cached_object_cloud(obj_path)
        obj_pts_local = object_cloud._get_vertices_torch(env.device).float() * scale_tensor.float()
        sample_pose = object_cloud.sample_stable_pose_trimesh(scale=scale)
        _, quat = sample_pose  # (position, quaternion)
        
        # Add a random yaw offset to quaternion (rotate around Z-axis)
        yaw_offset = (torch.rand(1, device=env.device) * (2 * torch.pi) - torch.pi).squeeze(0)
        
        # Create rotation quaternion for yaw offset around Z-axis
        yaw_quat = torch.zeros(4, device=env.device)
        yaw_quat[0] = torch.cos(yaw_offset * 0.5)  # w
        yaw_quat[3] = torch.sin(yaw_offset * 0.5)  # z (around Z-axis)
        
        # Convert numpy quaternion to torch tensor
        quat_tensor = torch.as_tensor(quat, device=env.device, dtype=torch.float32)
        
        # Applies yaw rotation to the original orientation
        w1, x1, y1, z1 = yaw_quat[0], yaw_quat[1], yaw_quat[2], yaw_quat[3]
        w2, x2, y2, z2 = quat_tensor[0], quat_tensor[1], quat_tensor[2], quat_tensor[3]
        
        qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
        qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
        qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
        qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
        
        # Normalize quaternion after multiplication to ensure it's a unit quaternion
        quat_norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if quat_norm > 1e-12:
            qw = qw / quat_norm
            qx = qx / quat_norm
            qy = qy / quat_norm
            qz = qz / quat_norm
        quat_tensor = torch.stack((qw, qx, qy, qz), dim=0)

        dx = (torch.rand((), device=env.device) * 2.0 - 1.0) * xy_range
        dy = (torch.rand((), device=env.device) * 4.0 - 2.0) * xy_range
        x_env = torch.as_tensor(0.5, device=env.device) + dx
        y_env = torch.as_tensor(0.0, device=env.device) + dy
        pos_x = x_env + env.scene.env_origins[env_id_int, 0]
        pos_y = y_env + env.scene.env_origins[env_id_int, 1]
        pos_z = (
            surface_z_for_points(obj_pts_local, quat_tensor, surface_z)
            + env.scene.env_origins[env_id_int, 2]
        )
        
        # Fill pose row
        poses[i, 0] = pos_x
        poses[i, 1] = pos_y
        poses[i, 2] = pos_z
        poses[i, 3] = qw
        poses[i, 4] = qx
        poses[i, 5] = qy
        poses[i, 6] = qz
        # velocities already zeros
    
    # Apply the new poses
    asset.write_root_state_to_sim(poses, env_ids)
    

def get_rigid_body_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    env_ids: torch.Tensor | list[int] | None = None,
):
    """Get rigid body scales with simple lazy per-env cache (env._scale_cache)."""
    # ensure cache dict
    if not hasattr(env, "_scale_cache") or env._scale_cache is None:
        env._scale_cache = {}

    # normalize requested ids (list[int])
    if isinstance(env_ids, torch.Tensor):
        requested = env_ids.tolist()
    else:
        requested = env_ids

    # find missing ids
    missing = [eid for eid in requested if eid not in env._scale_cache]
    if missing:
        # resolve prim paths once
        asset: RigidObject = env.scene[asset_cfg.name]
        prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
        if len(prim_paths) == 0:
            raise ValueError(f"Could not find prims with path: {asset.cfg.prim_path}")
        import isaacsim.core.utils.prims as prim_utils
        for eid in missing:
            prim = prim_utils.get_prim_at_path(prim_paths[eid])
            scale = prim.GetAttribute("xformOp:scale").Get()
            env._scale_cache[eid] = torch.tensor(scale, device=env.device, dtype=torch.float32)

    # assemble output in request order
    return torch.stack([env._scale_cache[eid] for eid in requested], dim=0)


def preload_object_pointclouds(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | list[int] | None,
    object_cloud_source: str,
    preprocessed_dir: str,
    num_points: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Load every locally assigned object cloud and move the full batch to the GPU.

    This is a prestartup event and must run after object scale randomization. The
    cached tensor is already expanded per environment and has each environment's
    actual USD scale baked into its local-frame points.
    """
    if env_ids is not None:
        raise ValueError("preload_object_pointclouds must run globally with env_ids=None")
    if object_cloud_source not in {"preprocessed", "mesh_sampled"}:
        raise ValueError(
            "object_cloud_source must be 'preprocessed' or 'mesh_sampled', got "
            f"{object_cloud_source!r}"
        )
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_object_asset_cfg_for_env,
    )

    num_envs = int(env.scene.num_envs)
    assigned_assets = [get_object_asset_cfg_for_env(env_id) for env_id in range(num_envs)]
    object_ids = [Path(asset.obj_path).stem for asset in assigned_assets]
    unique_assets = {}
    for object_id, asset in zip(object_ids, assigned_assets):
        unique_assets.setdefault(object_id, asset)

    points_by_object: dict[str, np.ndarray] = {}
    if object_cloud_source == "preprocessed":
        pointcloud_dir = Path(preprocessed_dir).expanduser()
        if not pointcloud_dir.is_dir():
            raise FileNotFoundError(
                "Preprocessed object point-cloud directory does not exist or is not a directory: "
                f"{pointcloud_dir}"
            )
        paths_by_object = {
            object_id: pointcloud_dir / f"{object_id}_first_hit_fps_{num_points}.npy"
            for object_id in unique_assets
        }
        missing = [
            (object_id, path)
            for object_id, path in paths_by_object.items()
            if not path.is_file()
        ]
        if missing:
            preview = "\n".join(
                f"  {object_id}: {path}" for object_id, path in missing[:20]
            )
            suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
            raise FileNotFoundError(
                f"Missing preprocessed point clouds for {len(missing)} of "
                f"{len(unique_assets)} assigned objects:\n{preview}{suffix}"
            )

        for object_id, path in paths_by_object.items():
            try:
                points = np.load(path, allow_pickle=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load preprocessed point cloud for {object_id!r}: {path}"
                ) from exc
            points = np.asarray(points)
            if points.shape != (num_points, 3):
                raise RuntimeError(
                    f"Preprocessed point cloud has invalid shape for {object_id!r}: "
                    f"{path} has {points.shape}, expected {(num_points, 3)}"
                )
            if not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
                raise RuntimeError(
                    f"Preprocessed point cloud contains non-numeric or non-finite data: {path}"
                )
            points_by_object[object_id] = points.astype(np.float32, copy=False)
        source_description = str(pointcloud_dir)
    else:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud

        for object_id, asset in unique_assets.items():
            cloud = Cloud(
                asset.obj_path,
                target_num_points=num_points,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            points_by_object[object_id] = cloud.points.numpy()
        source_description = "mesh sampling / per-mesh pc_npy_cache"

    points_np = np.stack([points_by_object[object_id] for object_id in object_ids], axis=0)
    points_l = torch.from_numpy(points_np).to(device=env.device, dtype=torch.float32)
    scales = get_rigid_body_scale(
        env,
        asset_cfg,
        list(range(num_envs)),
    ).to(device=env.device, dtype=torch.float32)
    env._object_pointcloud_points_l = (points_l * scales.unsqueeze(1)).contiguous()
    env._object_pointcloud_scales = scales.contiguous()
    env._object_pointcloud_source = object_cloud_source

    memory_mib = env._object_pointcloud_points_l.numel() * 4 / (1024.0 * 1024.0)
    print(
        "[ObjectPointCloudPreload] "
        f"source={object_cloud_source} envs={num_envs} unique_objects={len(unique_assets)} "
        f"shape={tuple(env._object_pointcloud_points_l.shape)} device={env.device} "
        f"memory={memory_mib:.1f}MiB path={source_description}",
        flush=True,
    )


def randomize_terrain_material(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    """Randomize terrain material properties.
    
    This function randomizes the physics material properties of the terrain mesh.
    Since terrain is not a standard RigidObject, we need to directly access the
    terrain mesh prim and update its material properties.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize (ignored for terrain as it's global).
        static_friction_range: Range for static friction coefficient.
        dynamic_friction_range: Range for dynamic friction coefficient.
        restitution_range: Range for restitution coefficient.
        num_buckets: Number of material buckets for randomization.
    """
    # Get terrain from scene
    terrain = env.scene["terrain"]
    
    # Generate material buckets
    static_friction_buckets = torch.linspace(
        static_friction_range[0], static_friction_range[1], num_buckets, device=env.device
    )
    dynamic_friction_buckets = torch.linspace(
        dynamic_friction_range[0], dynamic_friction_range[1], num_buckets, device=env.device
    )
    restitution_buckets = torch.linspace(
        restitution_range[0], restitution_range[1], num_buckets, device=env.device
    )
    
    # Ensure dynamic friction <= static friction
    dynamic_friction_buckets = torch.min(dynamic_friction_buckets, static_friction_buckets)
    
    # Randomly select material properties
    bucket_id = torch.randint(0, num_buckets, (1,), device=env.device)
    
    static_friction = static_friction_buckets[bucket_id]
    dynamic_friction = dynamic_friction_buckets[bucket_id]
    restitution = restitution_buckets[bucket_id]
    
    # Update terrain physics material
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdPhysics, UsdShade, Sdf

    # Get the terrain prim path
    terrain_prim_path = terrain.cfg.prim_path + "/terrain"

    # Create physics material prim if it doesn't exist
    physics_material_path = f"{terrain_prim_path}/physicsMaterial"
    physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)
    if not physics_material_prim or not physics_material_prim.IsValid():
        # Create the prim and define it as a Material + PhysicsMaterial
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        physics_material_prim = stage.DefinePrim(physics_material_path, "Material")

    # Apply PhysicsMaterialAPI and set properties
    physics_material = UsdPhysics.MaterialAPI.Apply(physics_material_prim)
    physics_material.CreateStaticFrictionAttr().Set(static_friction.item())
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction.item())
    physics_material.CreateRestitutionAttr().Set(restitution.item())

    # Bind physics material to the terrain collision prim
    # Try to find a Plane child; fall back to the terrain prim itself
    collision_prim = prim_utils.get_first_matching_child_prim(
        terrain_prim_path,
        predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
    )
    if collision_prim is not None:
        target_path = collision_prim.GetPrimPath()
    else:
        # Fall back: bind to the terrain prim directly
        target_path = terrain_prim_path

    import isaaclab.sim as sim_utils
    sim_utils.bind_physics_material(target_path, physics_material_path)

def compute_head_area_offsets_from_usd(env) -> "torch.Tensor":
    """Compute per-env head area offsets in the tool's LOCAL frame using OBJ mesh bounds.

    Each env may have a different tool (via MultiUsdFileCfg).  Tool identity
    is resolved through env_tool's runtime assignment mapping.

    Uses the cached OBJ point cloud to compute bounding box bounds in
    canonical (unscaled) mesh space, applies head_area_norm interpolation,
    then scales by the constant TOOL_SCALE.
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    head_area_offsets = torch.zeros(env.num_envs, 3, device=env.device)

    # Pre-compute per-tool local offsets (cache to avoid redundant work)
    _per_tool_offset_cache: dict[int, torch.Tensor] = {}

    for env_id in range(env.num_envs):
        tool_idx = get_tool_index_for_env(env_id)
        if tool_idx in _per_tool_offset_cache:
            head_area_offsets[env_id] = _per_tool_offset_cache[tool_idx]
            continue

        td = get_tool_data_for_env(env_id)
        head_area_norm = td.get("head_area")
        if head_area_norm is None:
            print(f"[WARNING compute_head_area] No head_area_norm for tool '{td['name']}', offset=[0,0,0]")
            _per_tool_offset_cache[tool_idx] = torch.zeros(3, device=env.device)
            continue

        mid_norm = [(head_area_norm[0][i] + head_area_norm[1][i]) / 2.0 for i in range(3)]

        # Compute OBJ-space bbox from the cached point cloud
        cloud = get_cached_cloud(td["obj_path"])
        pts = torch.tensor(cloud.points, dtype=torch.float32, device=env.device)
        bbox_min = pts.min(dim=0).values
        bbox_max = pts.max(dim=0).values

        mid_norm_t = torch.tensor(mid_norm, dtype=torch.float32, device=env.device)
        head_area_unscaled = bbox_min + mid_norm_t * (bbox_max - bbox_min)

        scale_t = torch.tensor([TOOL_SCALE] * 3, dtype=torch.float32, device=env.device)

        # Compute the body origin in OBJ space from base_center, then express
        # the head area center relative to that origin.
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, dtype=torch.float32, device=env.device)
            body_origin = bbox_min + bc * (bbox_max - bbox_min)
            head_area_local = (head_area_unscaled - body_origin) * scale_t
        else:
            # Fallback: legacy Z-shift only
            head_area_from_attachment = head_area_unscaled.clone()
            head_area_from_attachment[2] = head_area_unscaled[2] - bbox_min[2]
            head_area_local = head_area_from_attachment * scale_t

        _per_tool_offset_cache[tool_idx] = head_area_local
        head_area_offsets[env_id] = head_area_local

    return head_area_offsets


# ---------------------------------------------------------------------------
# Tool (link_coacd_convex_piece_0) mass and friction randomization
# ---------------------------------------------------------------------------

def randomize_tool_mass(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    mass_range: tuple[float, float] = (0.1, 0.5),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the mass of the tool body (link_coacd_convex_piece_0).

    The tool is a separate rigid body in the robot articulation, connected
    to panda_link7 via a fixed weld joint. We target it by body name.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        mass_range: (min, max) mass in kg.
        robot_cfg: Robot scene entity config.
    """
    robot = env.scene[robot_cfg.name]
    # Resolve tool body index
    tool_body_cfg = SceneEntityCfg(robot_cfg.name, body_names=["link_coacd_convex_piece_0"])
    tool_body_cfg.resolve(env.scene)
    tool_idx = tool_body_cfg.body_ids[0]

    # Sample random masses
    num_envs = len(env_ids)
    new_masses = torch.rand(num_envs, device=env.device) * (mass_range[1] - mass_range[0]) + mass_range[0]

    # Update only the tool body's mass.
    # PhysX view tensors are always on CPU regardless of env.device.
    current_masses = robot.root_physx_view.get_masses().clone()  # CPU tensor
    env_ids_cpu = env_ids.cpu()  # index on CPU
    current_masses[env_ids_cpu, tool_idx] = new_masses.cpu()  # values also on CPU
    robot.root_physx_view.set_masses(current_masses, env_ids_cpu)


def randomize_tool_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    static_friction_range: tuple[float, float] = (0.8, 1.2),
    dynamic_friction_range: tuple[float, float] = (0.8, 1.2),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize friction of the tool body's collision shapes.

    The tool (link_coacd_convex_piece_0) has multiple collision shapes.
    We set the same friction value across all shapes for simplicity.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        static_friction_range: (min, max) static friction coefficient.
        dynamic_friction_range: (min, max) dynamic friction coefficient.
        restitution_range: (min, max) restitution coefficient.
        robot_cfg: Robot scene entity config.
    """
    robot = env.scene[robot_cfg.name]
    # Resolve tool body index
    tool_body_cfg = SceneEntityCfg(robot_cfg.name, body_names=["link_coacd_convex_piece_0"])
    tool_body_cfg.resolve(env.scene)
    tool_idx = tool_body_cfg.body_ids[0]

    num_envs = len(env_ids)
    # Sample random friction/restitution values
    static_fric = torch.rand(num_envs, device=env.device) * (static_friction_range[1] - static_friction_range[0]) + static_friction_range[0]
    dynamic_fric = torch.rand(num_envs, device=env.device) * (dynamic_friction_range[1] - dynamic_friction_range[0]) + dynamic_friction_range[0]
    # Ensure dynamic <= static
    dynamic_fric = torch.min(dynamic_fric, static_fric)
    restitution = torch.rand(num_envs, device=env.device) * (restitution_range[1] - restitution_range[0]) + restitution_range[0]

    # PhysX view tensors are always on CPU regardless of env.device.
    # Index and values must all be on CPU.
    current_props = robot.root_physx_view.get_material_properties().clone()  # CPU tensor (num_envs, num_shapes, 3)
    num_shapes = current_props.shape[1]
    num_bodies = robot.root_physx_view.get_masses().shape[1]

    # Find the shape range for the tool body (shapes are ordered by body)
    shapes_per_body = num_shapes // num_bodies
    shape_start = tool_idx * shapes_per_body
    shape_end = min((tool_idx + 1) * shapes_per_body, num_shapes)

    env_ids_cpu = env_ids.cpu()
    static_fric_cpu = static_fric.cpu()
    dynamic_fric_cpu = dynamic_fric.cpu()
    restitution_cpu = restitution.cpu()

    for shape_i in range(shape_start, shape_end):
        current_props[env_ids_cpu, shape_i, 0] = static_fric_cpu
        current_props[env_ids_cpu, shape_i, 1] = dynamic_fric_cpu
        current_props[env_ids_cpu, shape_i, 2] = restitution_cpu

    robot.root_physx_view.set_material_properties(current_props, env_ids_cpu)

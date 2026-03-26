# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for non-prehensile manipulation environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Usd, UsdPhysics, Gf, UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def setup_tool_robot_collision_filter(env: "ManagerBasedRLEnv", env_ids=None):
    """Create PhysicsCollisionGroups to filter out tool<->robot collisions.

    Called once at prestartup. Puts all tool prims in one group and all robot
    prims in another, then adds a filtered pair so PhysX ignores contacts
    between them.
    """
    stage = omni.usd.get_context().get_stage()

    # Create collision groups at world scope
    tool_group_path = "/World/CollisionGroups/ToolGroup"
    robot_group_path = "/World/CollisionGroups/RobotGroup"

    for path in (tool_group_path, robot_group_path):
        if not stage.GetPrimAtPath(path).IsValid():
            stage.DefinePrim(path, "PhysicsCollisionGroup")

    tool_group = UsdPhysics.CollisionGroup.Get(stage, tool_group_path)
    robot_group = UsdPhysics.CollisionGroup.Get(stage, robot_group_path)

    # Add filtered pair: tool group ignores robot group
    tool_group.GetFilteredGroupsRel().AddTarget(robot_group_path)
    robot_group.GetFilteredGroupsRel().AddTarget(tool_group_path)

    # Include all tool prims
    tool_includes = tool_group.GetCollidersCollectionAPI().GetIncludesRel()
    for env_id in range(env.num_envs):
        tool_includes.AddTarget(f"/World/envs/env_{env_id}/eef")

    # Include all robot prims
    robot_includes = robot_group.GetCollidersCollectionAPI().GetIncludesRel()
    for env_id in range(env.num_envs):
        robot_includes.AddTarget(f"/World/envs/env_{env_id}/Robot")

    print(f"[INFO] Tool<->Robot collision filter set up for {env.num_envs} envs.")


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
    
    # Define base position (center of table in environment coordinates)
    base_x = 0.5  # center of table x-coordinate
    base_y = 0.0  # center of table y-coordinate
    base_z = 0.0  # fixed height on table
    
    # Sample random positions within curriculum ranges
    num_resets = len(env_ids)
    # Create poses: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    poses = torch.zeros((num_resets, 13), device=env.device)
    
    # Per-env sampling from stable pose with random yaw offset
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import get_cached_cloud
    assets_cfg = asset.cfg.spawn.assets_cfg
    scales = get_rigid_body_scale(env, SceneEntityCfg("object"), env_ids)

    # Pre-compute tool AABB in world frame for each env (from eef asset current pose + mesh)
    eef_asset = env.scene["eef"]
    eef_assets_cfg = eef_asset.cfg.spawn.assets_cfg
    eef_pos_w = eef_asset.data.root_pos_w   # (num_envs, 3)
    eef_quat_w = eef_asset.data.root_quat_w  # (num_envs, 4)
    from isaaclab.utils.math import matrix_from_quat
    eef_rot = matrix_from_quat(eef_quat_w)  # (num_envs, 3, 3)

    def _get_aabb(center_w, rot, pts_local):
        """pts_local: (M,3) canonical points. Returns (min_w, max_w) each (3,)."""
        pts_w = (rot @ pts_local.T).T + center_w  # (M,3)
        return pts_w.min(dim=0).values, pts_w.max(dim=0).values

    # Cache tool aabbs per env
    tool_aabbs = []
    for env_id_int in range(env.num_envs):
        tool_idx = env_id_int % len(eef_assets_cfg)
        tool_cloud_obj = get_cached_cloud(eef_assets_cfg[tool_idx].obj_path)
        pts = tool_cloud_obj._get_points_torch(env.device).float()
        if hasattr(env, "_tool_scales"):
            pts = pts * env._tool_scales[env_id_int].float()
        t_min, t_max = _get_aabb(eef_pos_w[env_id_int], eef_rot[env_id_int], pts)
        tool_aabbs.append((t_min, t_max))

    for i, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())

        asset_idx = env_id_int % len(assets_cfg)
        obj_path = assets_cfg[asset_idx].obj_path
        scale_tensor = scales[i]
        scale = tuple(scale_tensor.cpu().numpy())

        # Object canonical points scaled (pose unknown yet, use for half-extents only)
        obj_cloud_obj = get_cached_cloud(obj_path)
        obj_pts_local = torch.tensor(obj_cloud_obj.points, dtype=torch.float32, device=env.device) * scale_tensor.float()
        obj_half = (obj_pts_local.max(dim=0).values - obj_pts_local.min(dim=0).values) / 2.0

        t_min, t_max = tool_aabbs[env_id_int]

        max_attempts = 20
        for _attempt in range(max_attempts):
            dx = sample_uniform(-xy_range, xy_range, (1,), device=env.device).squeeze(0)
            dy = sample_uniform(-2*xy_range, 2*xy_range, (1,), device=env.device).squeeze(0)
            x_env = torch.as_tensor(base_x, device=env.device) + dx
            y_env = torch.as_tensor(base_y, device=env.device) + dy

            # Enforce reachable workspace (project into XY annulus) if configured
            if hasattr(stable_pose_term, "enforce_workspace") and bool(stable_pose_term.enforce_workspace):
                cx = torch.as_tensor(getattr(stable_pose_term, "workspace_center_x", 0.0), device=env.device)
                cy = torch.as_tensor(getattr(stable_pose_term, "workspace_center_y", 0.0), device=env.device)
                rmin = torch.as_tensor(getattr(stable_pose_term, "workspace_radius_min", 0.0), device=env.device)
                rmax = torch.as_tensor(getattr(stable_pose_term, "workspace_radius_max", float("inf")), device=env.device)
                ddx = x_env - cx
                ddy = y_env - cy
                rr = torch.sqrt(ddx * ddx + ddy * ddy).clamp_min(1e-9)
                rr_clamped = torch.clamp(rr, min=rmin, max=rmax)
                scale_ws = rr_clamped / rr
                x_env = cx + ddx * scale_ws
                y_env = cy + ddy * scale_ws

            pos_x = x_env + env.scene.env_origins[env_id_int, 0]
            pos_y = y_env + env.scene.env_origins[env_id_int, 1]
            pos_z = torch.as_tensor(base_z, device=env.device)

            # Object AABB at this candidate position (axis-aligned, identity rotation approx)
            obj_center = torch.stack([pos_x, pos_y, pos_z])
            obj_min = obj_center - obj_half
            obj_max = obj_center + obj_half

            # AABB intersection test
            overlap = (obj_min <= t_max).all() and (obj_max >= t_min).all()
            if not overlap or _attempt == max_attempts - 1:
                break

        object_cloud = get_cached_cloud(obj_path)
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
    # For terrain, we need to access the physics material prim directly
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdPhysics
    
    # Get the terrain prim path
    # For plane terrain, the actual prim path is {cfg.prim_path}/terrain
    terrain_prim_path = terrain.cfg.prim_path + "/terrain"
    
    # Find the physics material prim
    physics_material_path = f"{terrain_prim_path}/physicsMaterial"
    physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)

    # Create or get the physics material
    physics_material = UsdPhysics.MaterialAPI.Apply(physics_material_prim)
    
    # Set material properties
    physics_material.CreateStaticFrictionAttr().Set(static_friction.item())
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction.item())
    physics_material.CreateRestitutionAttr().Set(restitution.item())
    
    # Apply the material to the terrain mesh
    # For plane terrain, we need to find the actual collision prim (Plane type)
    mesh_prim_path = f"{terrain_prim_path}/mesh"
    mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
    
    # Find the collision prim (Plane type) for ground plane
    collision_prim = prim_utils.get_first_matching_child_prim(
        terrain_prim_path, 
        predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
    )
    
    # Use IsaacLab's bind_physics_material function
    import isaaclab.sim as sim_utils
    sim_utils.bind_physics_material(collision_prim.GetPrimPath(), physics_material_path)

def compute_head_area_offsets_from_usd(env) -> "torch.Tensor":
    """Compute per-env head area offsets in the tool's LOCAL frame using OBJ mesh bounds.

    Uses the tool's OBJ file (via cached point cloud) to compute bounding box bounds in
    canonical (unscaled) mesh space, applies head_area_norm interpolation, then scales by
    the per-tool-type scale from UsdFileCfg.

    This avoids the instance-proxy contamination that occurs when querying USD BBoxCache
    on env_1+ prims in Isaac Lab multi-env setups: those prims are instance proxies whose
    ComputeLocalBound results include the environment world-origin offset, causing markers
    to drift by one env_spacing per tool type index.
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import get_cached_cloud

    eef_asset = env.scene["eef"]
    assets_cfg = eef_asset.cfg.spawn.assets_cfg

    head_area_offsets = torch.zeros(env.num_envs, 3, device=env.device)
    num_tool_types = len(assets_cfg)

    type_offsets = {}  # asset_idx -> torch.Tensor (3,)

    for asset_idx in range(num_tool_types):
        tool_cfg = assets_cfg[asset_idx]

        head_area_norm = getattr(tool_cfg, "head_area_norm", None)
        if head_area_norm is None:
            type_offsets[asset_idx] = torch.zeros(3, device=env.device)
            # print(f"[DEBUG head_area tool_{asset_idx}] No head_area_norm, offset=[0,0,0]")
            continue

        obj_path = getattr(tool_cfg, "obj_path", None)
        if obj_path is None:
            type_offsets[asset_idx] = torch.zeros(3, device=env.device)
            # print(f"[WARNING head_area tool_{asset_idx}] No obj_path, offset=[0,0,0]")
            continue

        mid_norm = [(head_area_norm[0][i] + head_area_norm[1][i]) / 2.0 for i in range(3)]

        # Compute OBJ-space bbox from the cached point cloud (canonical, no env offset).
        cloud = get_cached_cloud(obj_path)
        pts = torch.tensor(cloud.points, dtype=torch.float32, device=env.device)
        bbox_min = pts.min(dim=0).values
        bbox_max = pts.max(dim=0).values

        # Head area center in unscaled OBJ space
        mid_norm_t = torch.tensor(mid_norm, dtype=torch.float32, device=env.device)
        head_area_unscaled = bbox_min + mid_norm_t * (bbox_max - bbox_min)

        # Scale: read directly from UsdFileCfg.scale (set at load time, no USD query needed)
        cfg_scale = getattr(tool_cfg, "scale", None)
        if cfg_scale is not None:
            scale_t = torch.tensor(list(cfg_scale), dtype=torch.float32, device=env.device)
        elif hasattr(env, "_tool_scales"):
            scale_t = env._tool_scales[asset_idx].float()
        else:
            scale_t = torch.ones(3, device=env.device)

        # The normalized OBJ is centered at z=0, but in simulation panda_link8 sits at the
        # BASE of the tool (z≈bbox_min in normalized space, because the normalization process
        # centered the original mesh whose attachment end was at z≈0).
        # For X and Y panda_link8 is at the OBJ origin (≈0), so those components are unchanged.
        # For Z, subtract bbox_min[2] so the offset is measured from the attachment point.
        head_area_from_attachment = head_area_unscaled.clone()
        head_area_from_attachment[2] = head_area_unscaled[2] - bbox_min[2]
        head_area_local = head_area_from_attachment * scale_t
        type_offsets[asset_idx] = head_area_local

        # print(f"[DEBUG head_area tool_{asset_idx}] mid_norm={[round(v,3) for v in mid_norm]}")
        # print(f"[DEBUG head_area tool_{asset_idx}] bbox_min={[round(v,4) for v in bbox_min.tolist()]} max={[round(v,4) for v in bbox_max.tolist()]}")
        # print(f"[DEBUG head_area tool_{asset_idx}] scale={[round(v,4) for v in scale_t.tolist()]}")
        # print(f"[DEBUG head_area tool_{asset_idx}] head_area_local={[round(v,4) for v in head_area_local.tolist()]}")

    # Broadcast: assign each env the offset for its tool type.
    for env_id in range(env.num_envs):
        asset_idx = env_id % num_tool_types
        head_area_offsets[env_id] = type_offsets.get(asset_idx, torch.zeros(3, device=env.device))

    #print(f"[DEBUG head_area] Done. offsets[0]={head_area_offsets[0].tolist()}")
    #if env.num_envs > 1:
    #    print(f"[DEBUG head_area] offsets[1]={head_area_offsets[1].tolist()} (should equal offsets[0] if same tool type)")
    return head_area_offsets


def adjust_eef_origin_to_base_center(env, env_ids):
    """Shift tool mesh so base_center becomes the geometric center."""
    if env_ids is None:
        env_ids = range(env.num_envs)

    eef_asset = env.scene["eef"]
    assets_cfg = eef_asset.cfg.spawn.assets_cfg
    print(f"[DEBUG adjust_eef_origin] Processing {len(list(env_ids))} environments")

    for i in env_ids:
        ee_root_path = f"/World/envs/env_{i}/eef"
        print(f"\n[DEBUG adjust_eef_origin] === Env {i} ===")
        print(f"[DEBUG adjust_eef_origin] Tool path: {ee_root_path}")

        ee_prim = prim_utils.get_prim_at_path(ee_root_path)
        if not ee_prim:
            print(f"[DEBUG adjust_eef_origin] ERROR: Tool prim not found at {ee_root_path}")
            continue

        # Get tool config and base_center
        asset_idx = i % len(assets_cfg)
        tool_cfg = assets_cfg[asset_idx]
        base_center_norm = torch.tensor(tool_cfg.base_center, device=env.device)
        print(f"[DEBUG adjust_eef_origin] Tool config index: {asset_idx}")
        print(f"[DEBUG adjust_eef_origin] base_center_norm: {base_center_norm.tolist()}")

        # Compute bbox
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeLocalBound(ee_prim)
        bbox_range = bbox.GetRange()
        bbox_min = torch.tensor([bbox_range.GetMin()[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2]], device=env.device)
        bbox_max = torch.tensor([bbox_range.GetMax()[0], bbox_range.GetMax()[1], bbox_range.GetMax()[2]], device=env.device)
        bbox_size = bbox_max - bbox_min
        print(f"[DEBUG adjust_eef_origin] BBox min: {bbox_min.tolist()}")
        print(f"[DEBUG adjust_eef_origin] BBox max: {bbox_max.tolist()}")
        print(f"[DEBUG adjust_eef_origin] BBox size: {bbox_size.tolist()}")

        # Compute offset to shift mesh: (0.5 - base_center_norm) * bbox_size
        offset = (0.5 - base_center_norm) * bbox_size
        print(f"[DEBUG adjust_eef_origin] Computed offset: {offset.tolist()}")

        # Apply transform to mesh child prim
        child_found = False
        for child in ee_prim.GetChildren():
            child_type = child.GetTypeName()
            print(f"[DEBUG adjust_eef_origin] Checking child: {child.GetPath()} (type: {child_type})")
            if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Xform):
                xform = UsdGeom.Xformable(child)
                existing_ops = xform.GetOrderedXformOps()
                print(f"[DEBUG adjust_eef_origin] Found transformable child, existing ops: {len(existing_ops)}")

                # Get existing translate op or create if doesn't exist
                translate_op = existing_ops[0] if existing_ops else xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(offset[0].item(), offset[1].item(), offset[2].item()))
                print(f"[DEBUG adjust_eef_origin] Applied transform to {child.GetPath()}")
                child_found = True
                break

        if not child_found:
            print(f"[DEBUG adjust_eef_origin] WARNING: No Mesh or Xform child found for env {i}")

def update_eef_pose(env, env_ids=None):
    """Teleport eef tool to follow robot end effector each step."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    eef = env.scene["eef"]

    # Get panda_link7 body pose (flange/wrist)
    robot_cfg = SceneEntityCfg("robot", body_names=["panda_link8"])
    robot_cfg.resolve(env.scene)
    link7_idx = robot_cfg.body_ids[0]

    # Copy link7 pose to tool
    eef.data.root_state_w[env_ids, :3] = robot.data.body_state_w[env_ids, link7_idx, :3]
    eef.data.root_state_w[env_ids, 3:7] = robot.data.body_state_w[env_ids, link7_idx, 3:7]
    eef.write_root_state_to_sim(eef.data.root_state_w, env_ids=env_ids)

    # Visualize head area position using freshly-set eef state (avoids ee_frame sensor lag).
    if getattr(env.cfg, 'visualize_eef_position', False):
        from isaaclab.utils.math import matrix_from_quat as _mfq
        eef_pos_w = eef.data.root_state_w[:, :3]    # (N, 3) – current panda_link8 pos
        eef_quat_w = eef.data.root_state_w[:, 3:7]  # (N, 4)
        R = _mfq(eef_quat_w)                         # (N, 3, 3)
        offset = env._head_area_offsets              # (N, 3)
        head_area_pos_w = eef_pos_w + torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)
        head_area_pose_7d = torch.cat([head_area_pos_w, eef_quat_w], dim=-1)
        visualize_eef_position(env, head_area_pose_7d)

def visualize_eef_position(env, eef_pose_7d: torch.Tensor):
    """Visualize the eef tool position."""
    try:
        # Create the visualizer only once and cache it on the env object.
        # Recreating it every step causes stale USD prims and wrong visual positions.
        if not hasattr(env, '_eef_position_visualizer') or env._eef_position_visualizer is None:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import FRAME_MARKER_CFG

            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/EefPosition"
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            env._eef_position_visualizer = VisualizationMarkers(marker_cfg)

        world_positions = eef_pose_7d[:, :3]
        quaternions = eef_pose_7d[:, 3:7]
        env._eef_position_visualizer.visualize(translations=world_positions, orientations=quaternions)
    except Exception as e:
        print(f"[ERROR visualize_eef_position] {type(e).__name__}: {e}")
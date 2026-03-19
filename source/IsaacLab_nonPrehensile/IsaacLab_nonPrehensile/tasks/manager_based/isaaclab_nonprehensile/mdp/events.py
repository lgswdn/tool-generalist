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
    
    for i, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())
        # XY around base with curriculum range (environment coordinates + origin)
        dx = sample_uniform(-xy_range, xy_range, (1,), device=env.device).squeeze(0)
        dy = sample_uniform(-2*xy_range, 2*xy_range, (1,), device=env.device).squeeze(0)
        # local (environment) coordinates before origin offset
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
            scale = rr_clamped / rr
            x_env = cx + ddx * scale
            y_env = cy + ddy * scale
        
        pos_x = x_env + env.scene.env_origins[env_id_int, 0]
        pos_y = y_env + env.scene.env_origins[env_id_int, 1]
        pos_z = torch.as_tensor(base_z, device=env.device)
        
        # Determine this env's asset and sample a stable pose (roll/pitch/yaw)
        asset_idx = env_id_int % len(assets_cfg)
        obj_path = assets_cfg[asset_idx].obj_path

        # Get the actual current scale from USD (dynamic scale system)
        scale_tensor = scales[i]
        scale = tuple(scale_tensor.cpu().numpy())

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
    """Compute per-env head area offsets in the tool's LOCAL frame using USD bounding boxes.

    Uses ComputeLocalBound (mesh-local, unaffected by world rotation) + prim scale to
    correctly map head_area_norm (normalized to the OBJ mesh bbox) to a scaled local offset.
    """
    from pxr import UsdGeom, Usd

    eef_asset = env.scene["eef"]
    assets_cfg = eef_asset.cfg.spawn.assets_cfg

    head_area_offsets = torch.zeros(env.num_envs, 3, device=env.device)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
    )

    for env_id in range(env.num_envs):
        asset_idx = env_id % len(assets_cfg)
        tool_cfg = assets_cfg[asset_idx]

        head_area_norm = getattr(tool_cfg, "head_area_norm", None)
        if head_area_norm is None:
            if env_id < 3:
                print(f"[DEBUG head_area env_{env_id}] No head_area_norm on cfg, offset=[0,0,0]")
            continue

        mid_norm = [(head_area_norm[0][i] + head_area_norm[1][i]) / 2.0 for i in range(3)]

        prim_path = f"/World/envs/env_{env_id}/eef"
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            print(f"[WARNING head_area] Prim not found: {prim_path}")
            continue

        # Local bbox: in the prim's own un-scaled coordinate space (= OBJ mesh space).
        # This is rotation-independent, so head_area_norm interpolation is always correct.
        bbox_local = bbox_cache.ComputeLocalBound(prim)
        bbox_range = bbox_local.GetRange()
        bbox_min_l = [bbox_range.GetMin()[i] for i in range(3)]
        bbox_max_l = [bbox_range.GetMax()[i] for i in range(3)]

        # Head area position in un-scaled local (OBJ) space
        head_area_unscaled = [
            bbox_min_l[i] + mid_norm[i] * (bbox_max_l[i] - bbox_min_l[i])
            for i in range(3)
        ]

        # Extract scale from the prim's own xformOps (set by UsdFileCfg.scale at spawn)
        scale = [1.0, 1.0, 1.0]
        xform_ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                s = op.Get()
                scale = [s[0], s[1], s[2]]
                break

        # Scale the local offset: this is the fixed vector in link8's local frame
        head_area_local = [head_area_unscaled[i] * scale[i] for i in range(3)]
        head_area_offsets[env_id] = torch.tensor(
            head_area_local, dtype=torch.float32, device=env.device
        )

        if env_id < 3:
            print(f"[DEBUG head_area env_{env_id}] mid_norm={[round(v,3) for v in mid_norm]}")
            print(f"[DEBUG head_area env_{env_id}] bbox_local_min={[round(v,4) for v in bbox_min_l]} max={[round(v,4) for v in bbox_max_l]}")
            print(f"[DEBUG head_area env_{env_id}] scale={[round(v,4) for v in scale]}")
            print(f"[DEBUG head_area env_{env_id}] head_area_local={[round(v,4) for v in head_area_local]}")

    print(f"[DEBUG head_area] Done. offsets[0]={head_area_offsets[0].tolist()}")
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

    # Debug: print eef base vs head area once every 200 calls
    if not hasattr(env, '_update_eef_debug_counter'):
        env._update_eef_debug_counter = 0
    env._update_eef_debug_counter += 1
    if env._update_eef_debug_counter % 2 == 1:
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import get_head_area_pos_w
        eef_base_w = eef.data.root_state_w[0, :3].tolist()
        head_area_w = get_head_area_pos_w(env)[0].tolist()
        offsets = getattr(env, '_head_area_offsets', None)
        offset0 = offsets[0].tolist() if offsets is not None else None
        print(f"[DEBUG update_eef step={env._update_eef_debug_counter}]")
        print(f"  eef_base_w  (env0): {[round(v,4) for v in eef_base_w]}")
        print(f"  head_area_w (env0): {[round(v,4) for v in head_area_w]}")
        print(f"  _head_area_offsets[0]: {[round(v,4) for v in offset0] if offset0 else None}")

    # Visualize eef tool position
    if getattr(env.cfg, 'visualize_eef_position', False):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import get_head_area_pos_w
        head_area_pos_w = get_head_area_pos_w(env)  # (N, 3) – actual head area world pos
        # Build 7D pose for visualizer: [x, y, z, qw, qx, qy, qz]
        head_area_pose_7d = torch.cat([head_area_pos_w, eef.data.root_state_w[:, 3:7]], dim=-1)
        visualize_eef_position(env, head_area_pose_7d)

def visualize_eef_position(env, eef_pose_7d: torch.Tensor):
    """Visualize the eef tool position."""
    try:
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/EefPosition"
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        env._eef_position_visualizer = VisualizationMarkers(marker_cfg)
        print(f"[DEBUG visualize_eef] Visualizer created at /Visuals/EefPosition")

        world_positions = eef_pose_7d[:, :3]
        quaternions = eef_pose_7d[:, 3:7]
        env._eef_position_visualizer.visualize(translations=world_positions, orientations=quaternions)
    except Exception as e:
        print(f"[ERROR visualize_eef_position] {type(e).__name__}: {e}")
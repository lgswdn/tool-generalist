# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.table_placement import (
    surface_z_for_points,
    table_contract_from_env,
    table_top_z_from_contract,
)

# Lightweight profiling utilities for command functions
import time
from functools import wraps

def _ensure_cmd_timers(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_cmd_timers"):
        env._cmd_timers = {}
    return env._cmd_timers

def profile_cmd(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        timers = _ensure_cmd_timers(self._env)
        name = fn.__name__
        t0 = time.perf_counter()
        result = fn(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        entry = timers.get(name)
        if entry is None:
            timers[name] = {"time": dt, "count": 1}
        else:
            entry["time"] += dt
            entry["count"] += 1
        return result
    return wrapper

def print_cmd_timers(env: "ManagerBasedRLEnv") -> None:
    timers = getattr(env, "_cmd_timers", {})
    if not timers:
        print("[cmd timers] no data collected yet")
        return
    print("[cmd timers] summary:")
    for name, entry in timers.items():
        total = entry["time"]
        count = entry["count"]
        avg = total / count if count > 0 else 0.0
        print(f"  {name}: total={total:.6f}s count={count} avg={avg:.6f}s")


class StablePoseCommand(CommandTerm):
    """Command generator for stable object poses using trimesh."""

    cfg: "StablePoseCommandCfg"

    def __init__(self, cfg: "StablePoseCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # create command buffer (7D: position + quaternion)
        self._command = torch.zeros(self.num_envs, 7, device=self.device)

        # Curriculum parameters - can be modified dynamically
        self.xy_offset_range = self.cfg.xy_offset_range
        self.initial_position_range = self.cfg.initial_position_range

        # metrics
        self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rot_to_goal"] = torch.zeros(self.num_envs, device=self.device)
        
        # create visualization markers for target poses if debug visualization is enabled
        if self.cfg.debug_vis:
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/goal_pose"
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Make frames smaller
            self._target_visualizer = VisualizationMarkers(marker_cfg)
        else:
            self._target_visualizer = None

    @profile_cmd
    def _resample_command(self, env_ids: torch.Tensor):
        """Resample commands for given environment indices."""
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_cached_cloud,
            get_object_asset_cfg_for_env,
            get_object_index_for_env,
        )

        # 1) Fetch per-env scales in batch
        scales = mdp.get_rigid_body_scale(self._env, SceneEntityCfg("object"), env_ids)

        # 2) Group envs by assigned base object index (same strategy as observations)
        group = {}
        for i, env_id in enumerate(env_ids):
            env_id_int = int(env_id.item())
            asset_idx = get_object_index_for_env(env_id_int)
            group.setdefault(asset_idx, {"env_ids": [], "indices": []})
            group[asset_idx]["env_ids"].append(env_id_int)
            group[asset_idx]["indices"].append(i)

        # 3) Preallocate output tensor on device
        out = torch.zeros((env_ids.shape[0], 7), dtype=torch.float32, device=self.device)
        placement_cfg = table_contract_from_env(self._env)
        table_surface_z = (
            table_top_z_from_contract(placement_cfg, self.device)
            if bool(placement_cfg.enabled)
            else torch.as_tensor(0.0, dtype=torch.float32, device=self.device)
        )

        # 4) Process each group in batch
        for asset_idx, g in group.items():
            obj_path = get_object_asset_cfg_for_env(g["env_ids"][0]).obj_path
            sel = torch.tensor(g["indices"], device=self.device, dtype=torch.long)
            scales_sub = scales.index_select(0, sel)  # (G,3)

            object_cloud = get_cached_cloud(obj_path)

            # Batch sample stable poses (returns numpy), then convert to torch
            pos_np, quat_np = object_cloud.sample_stable_pose_trimesh_batch(scale=scales_sub.cpu().numpy())
            pos = torch.from_numpy(pos_np).to(self.device, dtype=torch.float32)      # (G,3)
            quat = torch.from_numpy(quat_np).to(self.device, dtype=torch.float32)   # (G,4)

            # Add random yaw offsets in batch to quaternion (rotate around Z-axis)
            yaw_offsets = (torch.rand(pos.shape[0], device=self.device) * (2 * torch.pi) - torch.pi)
            
            # Create rotation quaternions for yaw offsets around Z-axis
            yaw_quats = torch.zeros(pos.shape[0], 4, device=self.device)
            yaw_quats[:, 0] = torch.cos(yaw_offsets * 0.5)  # w
            yaw_quats[:, 3] = torch.sin(yaw_offsets * 0.5)  # z (around Z-axis)
        
            w1, x1, y1, z1 = yaw_quats[:, 0], yaw_quats[:, 1], yaw_quats[:, 2], yaw_quats[:, 3]
            w2, x2, y2, z2 = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            
            quat[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
            quat[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
            quat[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
            quat[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
            
            # Normalize quaternions after multiplication to ensure they are unit quaternions
            quat_norms = torch.norm(quat, dim=1, keepdim=True)
            quat = torch.where(quat_norms > 1e-12, quat / quat_norms, quat)

            object_vertices = object_cloud._get_vertices_torch(self.device).float()
            r = self.xy_offset_range
            sign_x = torch.where(torch.rand(pos.shape[0], device=self.device) < 0.5, -1.0, 1.0)
            sign_y = torch.where(torch.rand(pos.shape[0], device=self.device) < 0.5, -1.0, 1.0)
            mag_x = (0.5 + torch.rand(pos.shape[0], device=self.device) * 0.5) * r
            mag_y = (0.5 + torch.rand(pos.shape[0], device=self.device) * 0.5) * r
            pos[:, 0] = 0.5 + sign_x * mag_x
            pos[:, 1] = 2.0 * (sign_y * mag_y)
            points_local = object_vertices.unsqueeze(0) * scales_sub[:, None, :].float()
            pos[:, 2] = surface_z_for_points(points_local, quat, table_surface_z)

            # Build pose_7d: [x, y, z, qw, qx, qy, qz]
            pose_7d = torch.cat([pos, quat], dim=1)  # (G,7)

            # Scatter back into output at selected indices
            out.index_copy_(0, sel, pose_7d)

        # 5) Write results to command buffer in env_ids order
        for i, env_id in enumerate(env_ids):
            env_id_int = int(env_id.item())
            self.command[env_id_int] = out[i]

        # Update visualization after resampling commands
        if self.cfg.debug_vis and self._target_visualizer is not None:
            self._update_visualization()
                        
    def _update_command(self):
        """Update the command. No-op for this implementation."""
        pass

    def _update_visualization(self):
        """Update the visualization of target poses."""
        if self.cfg.debug_vis and self._target_visualizer is not None:
            local_positions = self.command[:, :3].clone()
            quaternions = self.command[:, 3:7].clone()
            
            # Convert local positions to world positions by adding environment origins
            world_positions = local_positions + self._env.scene.env_origins
            
            # visualize target pose frames using world positions
            self._target_visualizer.visualize(translations=world_positions, orientations=quaternions)

    def _update_metrics(self):
        """Update the metrics."""
        # compute distance to goal (simplified)
        object_asset: RigidObject = self._env.scene["object"]
        # Get object position in local coordinates (relative to environment origin)
        object_pos_world = object_asset.data.root_pos_w[:, :3]
        object_pos_local = object_pos_world - self._env.scene.env_origins
        
        # Target position is already in local coordinates
        target_pos_local = self.command[:, :3]
        
        self.metrics["distance_to_goal"] = torch.norm(object_pos_local - target_pos_local, dim=-1)

        # --- Rotation matrix distance metric ---
        # Get current and target orientation
        object_quat_w = object_asset.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
        # Target quaternion (in command)
        target_quat = self.command[:, 3:7]  # [w, x, y, z]

        # Compute quaternion angular distance (in radians)
        dot_product = torch.sum(object_quat_w * target_quat, dim=1)
        dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
        quat_angle = 2 * torch.acos(dot_product)
        self.metrics["rot_to_goal"] = quat_angle
        self._print_pose_debug(object_pos_local, object_quat_w, target_pos_local, target_quat)

    def _print_pose_debug(
        self,
        object_pos_local: torch.Tensor,
        object_quat_w: torch.Tensor,
        target_pos_local: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> None:
        count = int(getattr(self._env, "_rl_pose_debug_print_count", 0))
        setattr(self._env, "_rl_pose_debug_print_count", count + 1)
        if count >= 10 and count % 1000 != 0:
            return
        if self.num_envs <= 0:
            return

        ee_frame = self._env.scene["ee_frame"]
        hand_pos_local = mdp.get_head_area_pos_w(self._env) - self._env.scene.env_origins
        hand_quat_w = ee_frame.data.target_quat_w[..., 0, :]
        env_i = 0

        def _vals(tensor: torch.Tensor) -> list[float]:
            return [round(float(v), 6) for v in tensor.detach().cpu().tolist()]

        print(
            "[rl_pose_debug] "
            f"step={count} env={env_i} "
            f"hand_pos_E={_vals(hand_pos_local[env_i])} "
            f"hand_quat_wxyz={_vals(hand_quat_w[env_i])} "
            f"object_pos_E={_vals(object_pos_local[env_i])} "
            f"object_quat_wxyz={_vals(object_quat_w[env_i])} "
            f"target_pos_E={_vals(target_pos_local[env_i])} "
            f"target_quat_wxyz={_vals(target_quat[env_i])} "
            f"distance_to_goal={float(self.metrics['distance_to_goal'][env_i].detach().cpu()):.6f} "
            f"rot_to_goal={float(self.metrics['rot_to_goal'][env_i].detach().cpu()):.6f}",
            flush=True,
        )

    @property
    def command(self) -> torch.Tensor:
        """The generated command."""
        return self._command

    @command.setter
    def command(self, value: torch.Tensor):
        """Set the command."""
        self._command = value.clone()


@configclass
class StablePoseCommandCfg(CommandTermCfg):
    """Configuration for stable pose command generator."""

    class_type: type = StablePoseCommand
    
    # Curriculum-controllable parameters
    xy_offset_range: float = 0.15
    """Target XY offset range configured from ``RLCfg.object_pose_sampling``."""
    
    initial_position_range: float = 0.15
    """Initial XY offset range configured from ``RLCfg.object_pose_sampling``."""
    
    # Visualization
    debug_vis: bool = False
    """Whether to visualize the target poses."""

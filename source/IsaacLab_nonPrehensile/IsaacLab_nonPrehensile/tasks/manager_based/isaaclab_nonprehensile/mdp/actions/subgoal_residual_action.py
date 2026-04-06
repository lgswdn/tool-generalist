# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Subgoal residual action with variable impedance control for non-prehensile manipulation.

This module implements the action space described in Kim et al. (2023):
- End-effector subgoal residuals (6D): translation + axis-angle rotation
- Joint-space proportional gains (7D)
- Joint-space damping ratios (7D)

The action computes joint torques using:
τ = kp(qtarget - qt) - kd * q̇t, where kd = ρ * √kp
qtarget = qt + IK(ΔTee) using damped least squares
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import apply_delta_pose, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SubgoalResidualAction(ActionTerm):
    """Subgoal residual action with variable impedance control.
    
    This action term implements the variable impedance control described in Kim et al. (2023).
    The action space consists of:
    - End-effector subgoal residuals (6D): [Δt(3), Δr(3)]
    - Joint proportional gains (7D): [kp(7)]
    - Joint damping ratios (7D): [ρ(7)]
    
    The controller computes joint torques using:
    τ = kp(qtarget - qt) - kd * q̇t
    where:
    - qtarget = qt + IK(ΔTee) using damped least squares
    - kd = ρ * √kp
    """

    def __init__(self, cfg: SubgoalResidualActionCfg, env: ManagerBasedEnv) -> None:
        # Initialize the action term
        super().__init__(cfg, env)

        # Resolve the robot asset
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Create robot entity configuration for unified joint and body lookup
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self._robot_entity_cfg.resolve(env.scene)
        
        # Get joint indices from the resolved configuration
        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)
        
        # Check that we have exactly 7 joints (for Franka Panda)
        if self._num_joints != 7:
            raise ValueError(f"Expected 7 joints, got {self._num_joints}. This action is designed for 7-DOF arms.")

        # Get end-effector body index from the resolved configuration
        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0]

        # Set up differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=True, 
            ik_method="dls",
        )

        self._ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

        # Action dimensions: 6 (ee residual) + 7 (kp) + 7 (damping ratio) = 20
        self._action_dim = 6 + self._num_joints + self._num_joints

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions from the last call to `process_actions()`."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (joint torques) applied to the robot."""
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the raw actions and compute joint torques.
        
        Args:
            actions: Raw actions of shape (num_envs, 20).
                     Format: [Δt(3), Δr(3), kp(7), ρ(7)]
        """
        # Store raw actions
        self._raw_actions = actions.clone()

        # Parse actions with config-driven linear mapping
        actions = actions.clone()
        # 0-2: End-effector position delta [meters], map [-1,1] to [-translation_scale, translation_scale]
        ee_pos_raw = torch.clamp(actions[:, 0:3], -1.0, 1.0)
        ee_pos_residual = ee_pos_raw * self.cfg.translation_scale
        # 3-5: End-effector axis-angle rotation delta [radians], map [-1,1] to [-rotation_scale, rotation_scale]
        ee_rot_raw = torch.clamp(actions[:, 3:6], -1.0, 1.0)
        ee_rot_residual = ee_rot_raw * self.cfg.rotation_scale
        # 6-12: Joint KP, range [min_kp, max_kp]
        kp_raw = torch.clamp(actions[:, 6:13], -1.0, 1.0)
        min_kp, max_kp = self.cfg.kp_scale
        kp = min_kp + (kp_raw + 1.0) * 0.5 * (max_kp - min_kp)  # shape: (num_envs, 7)
        # 13-19: Joint damping ratio, range [min_ratio, max_ratio]
        damping_ratio_raw = torch.clamp(actions[:, 13:20], -1.0, 1.0)
        min_ratio, max_ratio = self.cfg.damping_ratio_scale
        damping_ratio = min_ratio + (damping_ratio_raw + 1.0) * 0.5 * (max_ratio - min_ratio)  # shape: (num_envs, 7)
        
        # Compute actual damping coefficients: kd = ρ * √kp
        kd = damping_ratio * torch.sqrt(kp)

        # Store current gains
        self._current_kp = kp
        self._current_kd = kd

        # Get current robot state
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        current_joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        
        # Get current end-effector pose
        ee_pose_w = self._robot.data.body_state_w[:, self._robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w
        
        # Convert to robot base frame for IK
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Apply residual to current end-effector pose (in base frame)
        # Combine position and rotation residuals into 6D delta pose
        delta_pose = torch.cat([ee_pos_residual, ee_rot_residual], dim=-1)  # (num_envs, 6)


        # Set IK command (relative delta pose in robot base frame)
        self._ik_controller.set_command(delta_pose, ee_pos_b, ee_quat_b)

        # Get Jacobian for IK
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._joint_ids]

        # Solve IK to get target joint positions using damped least squares
        joint_pos_target = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        # Compute joint torques using impedance control
        # τ = kp(qtarget - qt) - kd * q̇t
        joint_pos_error = joint_pos_target - current_joint_pos
        joint_torques = kp * joint_pos_error - kd * current_joint_vel

        # Store processed actions (joint torques)
        self._processed_actions = joint_torques

    def apply_actions(self) -> None:
        """Apply the processed actions to the robot.
        
        This method is called by the action manager after process_actions().
        It applies the computed joint torques to the robot.
        """
        # Apply the computed joint torques to the robot
        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)

        # Reset IK controller
        self._ik_controller.reset(env_ids)


    """
    Helper methods for observations.
    """

    def get_current_impedance_gains(self, gain_type: str = "kp") -> torch.Tensor:
        """Get current impedance gains for observation.
        
        Args:
            gain_type: Type of gain to return ("kp" or "kd").
            
        Returns:
            Current impedance gains of shape (num_envs, num_joints).
        """
        if gain_type == "kp":
            return self._current_kp.clone()
        elif gain_type == "kd":
            return self._current_kd.clone()
        else:
            raise ValueError(f"Invalid gain type: {gain_type}. Must be 'kp' or 'kd'.")


@configclass
class SubgoalResidualActionCfg(ActionTermCfg):
    """Configuration for subgoal residual action with variable impedance control."""

    class_type: type[ActionTerm] = SubgoalResidualAction

    # Asset and joint configuration
    asset_name: str = "robot"
    """Name of the robot asset in the scene."""
    
    joint_names: list[str] = ["panda_joint.*"]
    """List of joint names or regex patterns for the controlled joints."""
    
    ee_frame_name: str = "ee_frame"
    """Name of the end-effector frame for pose control."""

    # Action scaling parameters
    translation_scale: float = 0.05
    """Maximum translation residual per step [m]."""
    
    rotation_scale: float = 0.2
    """Maximum rotation residual per step [rad]."""
    
    kp_scale: tuple[float, float] = (50.0, 300.0)
    """Proportional gain range [N⋅m/rad]: (min, max)."""
    
    damping_ratio_scale: tuple[float, float] = (0.3, 1.5)
    """Damping ratio range (dimensionless): (min, max)."""

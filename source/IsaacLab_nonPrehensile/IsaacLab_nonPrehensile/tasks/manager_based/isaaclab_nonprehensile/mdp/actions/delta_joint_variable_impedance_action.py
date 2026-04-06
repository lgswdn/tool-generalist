from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DeltaJointVariableImpedanceAction(ActionTerm):
    """Delta-joint position + variable Kp/rho action with custom impedance control implementation.

    - Action dimension: 3n = [Δq(n), Kp(n), rho(n)]
    - Control mode: joint-space impedance with variable stiffness and damping
    - Damping coefficient kd = sqrt(Kp) * rho
    - Optional inertial and gravity compensation
    """

    def __init__(self, cfg: DeltaJointVariableImpedanceActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names)
        self._robot_entity_cfg.resolve(env.scene)

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)

        # Use soft joint limits as controller clipping range for joint positions
        self._dof_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]

        # Build per-joint delta-q scale (supports scalar or per-joint sequence)
        if isinstance(cfg.joint_scale, (float, int)):
            self._joint_scale = torch.full((self.num_envs, self._num_joints), float(cfg.joint_scale), device=self.device)
        else:
            js = torch.tensor(cfg.joint_scale, device=self.device, dtype=torch.float32)
            if js.numel() != self._num_joints:
                raise ValueError(f"joint_scale length {js.numel()} != num_joints {self._num_joints}")
            self._joint_scale = js.view(1, -1).repeat(self.num_envs, 1)

        # Optional per-joint torque limits
        if cfg.torque_limits is not None:
            if isinstance(cfg.torque_limits, Sequence):
                torque_limits = torch.tensor(cfg.torque_limits, device=self.device, dtype=torch.float32)
                if torque_limits.numel() != self._num_joints:
                    raise ValueError(
                        f"torque_limits length {torque_limits.numel()} != num_joints {self._num_joints}"
                    )
                self._torque_limits = torque_limits
            else:
                raise ValueError("torque_limits must be a sequence of per-joint limits or None")
        else:
            # Default Franka Panda torque limits if not specified
            self._torque_limits = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=self.device)

        # Buffers for dynamics (optional)
        self._mass_matrix: torch.Tensor | None = None
        self._gravity: torch.Tensor | None = None

        # Action dimension: 3n = [Δq(n), Kp(n), rho(n)]
        self._action_dim = 3 * self._num_joints
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _maybe_update_dynamics(self):
        """Update dynamics quantities if compensation is enabled."""
        if self.cfg.use_inertial_compensation:
            full_mass = self._robot.root_physx_view.get_generalized_mass_matrices()
            # PhysX returns (num_envs, total_dofs, total_dofs)
            # Extract sub-matrix for controlled joints - keep as 3D for proper batch matrix multiplication
            self._mass_matrix = full_mass[:, self._joint_ids, :][:, :, self._joint_ids]
        else:
            self._mass_matrix = None
            
        if self.cfg.use_gravity_comp:
            self._gravity = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        else:
            self._gravity = None

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw actions
        self._raw_actions = actions.clone()

        # Clamp actions to [-1, 1]
        a = torch.clamp(actions, -1.0, 1.0)
        n = self._num_joints

        # Parse and map actions
        # Map delta q with per-joint scale
        delta_q = a[:, 0:n] * self._joint_scale
        
        # Map Kp from [-1, 1] to physical range
        kp_min, kp_max = self.cfg.kp_scale
        kp = kp_min + (a[:, n:2*n] + 1.0) * 0.5 * (kp_max - kp_min)
        
        # Map damping ratio rho from [-1, 1] to controller limits
        dr_min, dr_max = self.cfg.damping_ratio_scale
        rho = dr_min + (a[:, 2*n:3*n] + 1.0) * 0.5 * (dr_max - dr_min)

        # Get current robot state
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        current_joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        # Compute desired position with joint limits
        desired_joint_pos = current_joint_pos + delta_q
        desired_joint_pos = desired_joint_pos.clip_(
            min=self._dof_pos_limits[..., 0], 
            max=self._dof_pos_limits[..., 1]
        )

        # Compute errors
        joint_pos_error = desired_joint_pos - current_joint_pos
        joint_vel_error = -current_joint_vel

        # Compute acceleration using variable gains
        # kd = sqrt(kp) * rho (critical damping * damping ratio)
        kd = torch.sqrt(kp) * rho
        des_joint_acc = kp * joint_pos_error + kd * joint_vel_error

        # Compute torques
        if self.cfg.use_inertial_compensation:
            self._maybe_update_dynamics()
            # Use batch matrix multiplication: (batch, n, n) @ (batch, n, 1) -> (batch, n, 1)
            des_joint_acc_unsqueezed = des_joint_acc.unsqueeze(-1)
            joint_torques = torch.bmm(self._mass_matrix, des_joint_acc_unsqueezed).squeeze(-1)
        else:
            # Decoupled control without inertial compensation
            joint_torques = des_joint_acc

        # Add gravity compensation
        if self.cfg.use_gravity_comp:
            if self._gravity is None:
                self._maybe_update_dynamics()
            joint_torques += self._gravity

        # Clamp per-joint torques
        joint_torques = torch.max(torch.min(joint_torques, self._torque_limits), -self._torque_limits)

        self._processed_actions = joint_torques

    def apply_actions(self) -> None:
        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return


@configclass
class DeltaJointVariableImpedanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DeltaJointVariableImpedanceAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]

    # [-1, 1] → Δq scaling (radians). Supports float or per-joint list of length n.
    joint_scale: float | Sequence[float] = 1.0

    # Variable impedance ranges
    kp_scale: tuple[float, float] = (50.0, 300.0)
    damping_ratio_scale: tuple[float, float] = (0.1, 2.0)

    # Compensation flags
    use_inertial_compensation: bool = True
    use_gravity_comp: bool = True

    # Optional torque limits; set None to disable, otherwise pass per-joint list
    torque_limits: Sequence[float] | None = None 
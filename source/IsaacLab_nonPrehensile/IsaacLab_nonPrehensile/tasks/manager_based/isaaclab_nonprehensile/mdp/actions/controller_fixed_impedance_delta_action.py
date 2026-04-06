from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.controllers.joint_impedance import (
    JointImpedanceController,
    JointImpedanceControllerCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DeltaJointFixedImpedanceControllerAction(ActionTerm):
    """Delta-joint position action using JointImpedanceController with fixed Kp/rho.

    - Action dimension: n = [Δq(n)]
    - Controller: `impedance_mode="fixed"`, `command_type="p_rel"`
    - Gains (Kp, rho) are fixed from the config; controller computes kd = 2 * sqrt(Kp) * rho
    - Optional inertial and gravity compensation
    """

    def __init__(self, cfg: DeltaJointFixedImpedanceControllerActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names)
        self._robot_entity_cfg.resolve(env.scene)

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)

        # Use soft joint limits for controller clipping
        dof_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]

        # Build per-joint delta-q scale (supports scalar or per-joint sequence)
        if isinstance(cfg.joint_scale, (float, int)):
            self._joint_scale = torch.full((self.num_envs, self._num_joints), float(cfg.joint_scale), device=self.device)
        else:
            js = torch.tensor(cfg.joint_scale, device=self.device, dtype=torch.float32)
            if js.numel() != self._num_joints:
                raise ValueError(f"joint_scale length {js.numel()} != num_joints {self._num_joints}")
            self._joint_scale = js.view(1, -1).repeat(self.num_envs, 1)

        # Build controller with fixed gains
        ctrl_cfg = JointImpedanceControllerCfg(
            command_type="p_rel",
            impedance_mode="fixed",
            inertial_compensation=bool(cfg.use_inertial_compensation),
            gravity_compensation=bool(cfg.use_gravity_compensation),
            stiffness=cfg.fixed_kp,
            damping_ratio=cfg.fixed_damping_ratio,
            # Limits not used in fixed mode, but keep defaults
        )
        self._controller = JointImpedanceController(
            ctrl_cfg, num_robots=self.num_envs, dof_pos_limits=dof_pos_limits, device=self.device
        )

        # Optional dynamics buffers
        self._mass_matrix: torch.Tensor | None = None
        self._gravity: torch.Tensor | None = None

        # Optional torque limits
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
            self._torque_limits = None

        # Dimensions and buffers
        self._action_dim = self._num_joints
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
        if self._controller.cfg.inertial_compensation:
            full_mass = self._robot.root_physx_view.get_generalized_mass_matrices()
            mm = full_mass
            # reshape to (num_envs, total_dofs, total_dofs) if flattened to (num_envs*total_dofs, total_dofs)
            if mm.dim() == 2:
                total_dofs = self._robot.data.joint_pos.shape[1]
                if mm.shape[0] == self.num_envs * total_dofs and mm.shape[1] == total_dofs:
                    mm = mm.contiguous().view(self.num_envs, total_dofs, total_dofs)
            self._mass_matrix = mm[:, self._joint_ids, :][:, :, self._joint_ids]
        else:
            self._mass_matrix = None
        if self._controller.cfg.gravity_compensation:
            self._gravity = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        else:
            self._gravity = None

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw actions
        self._raw_actions = actions.clone()

        # Clamp and map Δq with per-joint scale
        a = torch.clamp(actions, -1.0, 1.0)
        delta_q = a * self._joint_scale

        # Set controller command: [Δq]
        self._controller.set_command(delta_q)

        # Compute torques
        self._maybe_update_dynamics()
        q = self._robot.data.joint_pos[:, self._joint_ids]
        qd = self._robot.data.joint_vel[:, self._joint_ids]
        tau = self._controller.compute(q, qd, mass_matrix=self._mass_matrix, gravity=self._gravity)

        # Optional torque clamping
        if self._torque_limits is not None:
            tau = torch.max(torch.min(tau, self._torque_limits), -self._torque_limits)

        self._processed_actions = tau

    def apply_actions(self) -> None:
        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return


@configclass
class DeltaJointFixedImpedanceControllerActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DeltaJointFixedImpedanceControllerAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]

    # [-1, 1] → Δq scaling (radians). Supports float or per-joint list of length n.
    joint_scale: float | Sequence[float] = 1.0

    # Fixed gains
    fixed_kp: float | Sequence[float] = 200.0
    fixed_damping_ratio: float | Sequence[float] = 1.0

    # Compensation flags
    use_inertial_compensation: bool = False
    use_gravity_compensation: bool = True

    # Optional torque limits; set None to disable, otherwise pass per-joint list
    torque_limits: Sequence[float] | None = None 
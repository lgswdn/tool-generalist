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


class DeltaJointVariableImpedanceControllerAction(ActionTerm):
    """Delta-joint position + variable Kp/rho action using JointImpedanceController.

    - Action dimension: 3n = [Δq(n), Kp(n), rho(n)]
    - Control mode: joint-space impedance with controller `impedance_mode="variable"` and `command_type="p_rel"`
    - Controller uses kd = sqrt(Kp) * rho internally (without coefficient 2)
    - Optional inertial and gravity compensation
    """

    def __init__(self, cfg: DeltaJointVariableImpedanceControllerActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names)
        self._robot_entity_cfg.resolve(env.scene)

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)

        # Use soft joint limits as controller clipping range for joint positions
        dof_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]

        # Build per-joint delta-q scale (supports scalar or per-joint sequence)
        if isinstance(cfg.joint_scale, (float, int)):
            self._joint_scale = torch.full((self.num_envs, self._num_joints), float(cfg.joint_scale), device=self.device)
        else:
            js = torch.tensor(cfg.joint_scale, device=self.device, dtype=torch.float32)
            if js.numel() != self._num_joints:
                raise ValueError(f"joint_scale length {js.numel()} != num_joints {self._num_joints}")
            self._joint_scale = js.view(1, -1).repeat(self.num_envs, 1)

        # Controller configuration
        # Initialize Kp/damping ratio at mid-range for stable startup
        kp_min, kp_max = cfg.kp_scale
        kp_init = 0.5 * (kp_min + kp_max)
        dr_min, dr_max = cfg.damping_ratio_scale
        dr_init = 0.5 * (dr_min + dr_max)

        ctrl_cfg = JointImpedanceControllerCfg(
            command_type="p_rel",
            impedance_mode="variable",
            inertial_compensation=bool(cfg.use_inertial_compensation),
            gravity_compensation=bool(cfg.use_gravity_comp),
            stiffness=kp_init,
            damping_ratio=dr_init,
            stiffness_limits=(kp_min, kp_max),
            damping_ratio_limits=(dr_min, dr_max),
        )

        # Properly instantiate the controller
        self._controller = JointImpedanceController(
            ctrl_cfg, num_robots=self.num_envs, dof_pos_limits=dof_pos_limits, device=self.device
        )

        # Buffers for dynamics (optional)
        self._mass_matrix: torch.Tensor | None = None
        self._gravity: torch.Tensor | None = None

        # Optional per-joint torque limits. Set to None to disable clamping.
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
        # Fetch dynamics only if compensation is enabled
        if self._controller.cfg.inertial_compensation:
            full_mass = self._robot.root_physx_view.get_generalized_mass_matrices()
            
            # PhysX returns (num_envs, total_dofs, total_dofs)
            # Extract sub-matrix for controlled joints - keep as 3D for proper batch matrix multiplication
            self._mass_matrix = full_mass[:, self._joint_ids, :][:, :, self._joint_ids]
        else:
            self._mass_matrix = None
        if self._controller.cfg.gravity_compensation:
            self._gravity = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        else:
            self._gravity = None

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw actions
        self._raw_actions = actions.clone()

        # Clamp actions to [-1, 1]
        a = torch.clamp(actions, -1.0, 1.0)
        n = self._num_joints

        # Parse and map
        # Map delta q with per-joint scale
        delta_q = a[:, 0:n] * self._joint_scale
        # Map Kp from [-1, 1] to physical range
        kp_min, kp_max = self.cfg.kp_scale
        kp = kp_min + (a[:, n:2*n] + 1.0) * 0.5 * (kp_max - kp_min)
        # Map damping ratio rho from [-1, 1] to controller limits
        dr_min, dr_max = self._controller.cfg.damping_ratio_limits
        rho = dr_min + (a[:, 2*n:3*n] + 1.0) * 0.5 * (dr_max - dr_min)

        # Set controller command: [Δq, Kp, rho]
        ctrl_cmd = torch.cat([delta_q, kp, rho], dim=-1)
        self._controller.set_command(ctrl_cmd)

        # Get current state
        q = self._robot.data.joint_pos[:, self._joint_ids]
        qd = self._robot.data.joint_vel[:, self._joint_ids]

        # Update dynamics parameters if compensation is needed
        if self._controller.cfg.inertial_compensation or self._controller.cfg.gravity_compensation:
            self._maybe_update_dynamics()

        # Use controller's compute method instead of reimplementing
        tau = self._controller.compute(
            dof_pos=q,
            dof_vel=qd,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity
        )

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
class DeltaJointVariableImpedanceControllerActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DeltaJointVariableImpedanceControllerAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]

    # [-1, 1] → Δq scaling (radians). Supports float or per-joint list of length n.
    joint_scale: float | Sequence[float] = 1.0

    # Physical range for Kp (used to map from [-1, 1])
    kp_scale: tuple[float, float] = (50.0, 300.0)

    # Damping ratio clamp range for controller and mapping from [-1, 1]
    damping_ratio_scale: tuple[float, float] = (0.1, 2.0)

    # Compensation flags
    use_inertial_compensation: bool = True
    use_gravity_comp: bool = True

    # Optional torque limits; set None to disable, otherwise pass per-joint list
    torque_limits: Sequence[float] | None = None 
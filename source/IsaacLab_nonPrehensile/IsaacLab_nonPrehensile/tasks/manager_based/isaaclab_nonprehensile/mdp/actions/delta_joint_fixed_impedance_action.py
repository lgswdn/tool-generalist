from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DeltaJointFixedImpedanceAction(ActionTerm):
    def __init__(self, cfg: DeltaJointFixedImpedanceActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names)
        self._robot_entity_cfg.resolve(env.scene)

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)

        if self._num_joints != 7:
            raise ValueError(f"Expected 7 joints, got {self._num_joints}.")

        # Fixed impedance gains
        self._fixed_kp = torch.full((env.num_envs, self._num_joints), cfg.fixed_kp, device=env.device)
        self._fixed_kd = torch.full((env.num_envs, self._num_joints), cfg.fixed_kd, device=env.device)

        # Franka Panda joint limits (radians)
        self._joint_limits_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=env.device,
        )
        self._joint_limits_max = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=env.device,
        )

        # Franka Panda continuous torque limits (Nm)
        self._torque_limits = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=env.device)

        self._action_dim = 7

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()

        # Scale actions in [-1, 1] to delta joint positions (radians)
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        delta_joint_pos = clamped_actions * self.cfg.joint_scale

        # Current joint states
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        current_joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        # Target joints with limits
        joint_pos_target = current_joint_pos + delta_joint_pos
        joint_pos_target = torch.clamp(joint_pos_target, self._joint_limits_min, self._joint_limits_max)

        # PD impedance torque
        joint_pos_error = joint_pos_target - current_joint_pos
        joint_torques = self._fixed_kp * joint_pos_error - self._fixed_kd * current_joint_vel

        # Per-joint torque clamping
        joint_torques = torch.max(torch.min(joint_torques, self._torque_limits), -self._torque_limits)

        self._processed_actions = joint_torques

    def apply_actions(self) -> None:
        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # No internal state to reset for this action
        return


@configclass
class DeltaJointFixedImpedanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DeltaJointFixedImpedanceAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]

    # Scale mapping from normalized action [-1, 1] to radians
    joint_scale: float = 1.0

    # Fixed impedance gains
    fixed_kp: float = 600.0
    fixed_kd: float = 50.0 
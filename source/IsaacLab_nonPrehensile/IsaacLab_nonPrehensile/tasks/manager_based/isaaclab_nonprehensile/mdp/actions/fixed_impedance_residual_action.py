from __future__ import annotations

import torch
from typing import TYPE_CHECKING


from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import apply_delta_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class FixedImpedanceResidualAction(ActionTerm):
    def __init__(self, cfg: FixedImpedanceResidualActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names, body_names=["panda_hand"])
        self._robot_entity_cfg.resolve(env.scene)

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)

        if self._num_joints != 7:
            raise ValueError(f"Expected 7 joints, got {self._num_joints}.")

        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0]

        # Franka Panda continuous torque limits (Nm)
        self._torque_limits = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=env.device)

        # 重力补偿配置
        self._use_gravity_comp: bool = bool(cfg.use_gravity_comp)
        self._gravity_scale = torch.clamp(torch.tensor(cfg.gravity_scale, device=env.device), min=0.0)

        # 设置固定kp和kd
        self._fixed_kp = torch.full((env.num_envs, self._num_joints), cfg.fixed_kp, device=env.device)
        self._fixed_kd = torch.full((env.num_envs, self._num_joints), cfg.fixed_kd, device=env.device)

        # 设置差分IK控制器，如subgoal
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        )
        self._ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

        self._action_dim = 6

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

        ee_pos_raw = torch.clamp(actions[:, 0:3], -1.0, 1.0)
        ee_pos_residual = ee_pos_raw * self.cfg.translation_scale
        ee_rot_raw = torch.clamp(actions[:, 3:6], -1.0, 1.0)
        ee_rot_residual = ee_rot_raw * self.cfg.rotation_scale

        current_joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        current_joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        ee_pose_w = self._robot.data.body_state_w[:, self._robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        delta_pose = torch.cat([ee_pos_residual, ee_rot_residual], dim=-1)

        self._ik_controller.set_command(delta_pose, ee_pos_b, ee_quat_b)

        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._joint_ids]

        joint_pos_target = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        joint_pos_error = joint_pos_target - current_joint_pos
        joint_torques = self._fixed_kp * joint_pos_error - self._fixed_kd * current_joint_vel

        # 可选：加入重力补偿偏置（feed-forward）
        if self._use_gravity_comp:
            gravity_vec = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
            joint_torques = joint_torques + self._gravity_scale * gravity_vec

        # 按物理极限裁剪
        joint_torques = torch.max(torch.min(joint_torques, self._torque_limits), -self._torque_limits)

        self._processed_actions = joint_torques

    def get_current_impedance_gains(self, gain_type: str = "kp") -> torch.Tensor:
        if gain_type == "kp":
            return self._fixed_kp.clone()
        elif gain_type == "kd":
            return self._fixed_kd.clone()
        else:
            raise ValueError(f"Invalid gain type: {gain_type}.")

    def _get_jacobian(self):
        return self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._joint_ids]

    def apply_actions(self) -> None:
        self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._ik_controller.reset(env_ids)


@configclass
class FixedImpedanceResidualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FixedImpedanceResidualAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]
    ee_frame_name: str = "ee_frame"

    translation_scale: float = 0.06
    rotation_scale: float = 0.1
    fixed_kp: float = 600.0  # 固定kp值，假设为标量，可扩展为tuple
    fixed_kd: float = 50.0   # 固定kd值

    # 重力补偿配置
    use_gravity_comp: bool = True
    gravity_scale: float = 1.0
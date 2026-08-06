from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SymmetricGeneratedGripperAction(ActionTerm):
    """One-dimensional absolute openness command for generated gripper fingers."""

    def __init__(self, cfg: SymmetricGeneratedGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._env = env
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._fixed_base_contract_checked = False
        self._robot_entity_cfg = SceneEntityCfg(cfg.asset_name, joint_names=cfg.joint_names)
        try:
            self._robot_entity_cfg.resolve(env.scene)
        except Exception as exc:
            available = tuple(getattr(self._robot.data, "joint_names", ()))
            raise RuntimeError(
                "generated_gripper requires the manifest finger joints "
                f"{tuple(cfg.joint_names)!r}; available joints are {available!r}"
            ) from exc

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)
        if self._num_joints != 2:
            raise RuntimeError(
                "SymmetricGeneratedGripperAction expected exactly two generated finger joints, "
                f"resolved {self._num_joints}: {self._joint_names!r}"
            )
        if cfg.open_joint_pos <= cfg.closed_joint_pos:
            raise ValueError(
                "SymmetricGeneratedGripperActionCfg.open_joint_pos must be greater "
                "than closed_joint_pos"
            )
        if cfg.clip[1] <= cfg.clip[0]:
            raise ValueError("SymmetricGeneratedGripperActionCfg.clip must be increasing")

        self._action_dim = 1
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        if cfg.semantic_closure:
            env._generated_gripper_commanded_closure = torch.zeros(
                self.num_envs, 1, device=self.device
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _validate_fixed_base_contract(self) -> None:
        if self._fixed_base_contract_checked:
            return
        root_physx_view = getattr(self._robot, "root_physx_view", None)
        shared_metatype = getattr(root_physx_view, "shared_metatype", None)
        if shared_metatype is None or not hasattr(shared_metatype, "fixed_base"):
            raise RuntimeError(
                "generated_gripper could not verify fixed-base contract because Isaac "
                "has not populated robot.root_physx_view.shared_metatype.fixed_base. "
                "This check must run after the articulation PhysX view is initialized."
            )
        if not bool(shared_metatype.fixed_base):
            raise RuntimeError(
                "generated_gripper requires fixed-base generated robot USDs. "
                "Regenerate USDs with gripper/convert_urdf.py after the fix_base=True change."
            )
        self._fixed_base_contract_checked = True

    def process_actions(self, actions: torch.Tensor) -> None:
        self._validate_fixed_base_contract()
        if actions.shape[-1] != self._action_dim:
            raise ValueError(f"Expected generated gripper action shape (*, 1), got {tuple(actions.shape)}")

        self._raw_actions = actions.clone()
        clipped = torch.clamp(actions[:, :1], self.cfg.clip[0], self.cfg.clip[1])
        normalized = (clipped - self.cfg.clip[0]) / (self.cfg.clip[1] - self.cfg.clip[0])
        if self.cfg.semantic_closure:
            # Cross-embodiment convention: -1=open, +1=closed, matching the
            # one-DoF Robotiq action adapter.
            target = self.cfg.open_joint_pos + normalized * (
                self.cfg.closed_joint_pos - self.cfg.open_joint_pos
            )
            self._env._generated_gripper_commanded_closure = normalized.detach()
        else:
            # Preserve the historical generated-gripper convention.
            target = self.cfg.closed_joint_pos + normalized * (
                self.cfg.open_joint_pos - self.cfg.closed_joint_pos
            )
        self._processed_actions = target.expand(-1, self._num_joints)

    def apply_actions(self) -> None:
        self._validate_fixed_base_contract()
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.cfg.semantic_closure:
            target = self._env._generated_gripper_commanded_closure
            if env_ids is None:
                target.zero_()
            else:
                target[env_ids] = 0.0
        return


@configclass
class SymmetricGeneratedGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SymmetricGeneratedGripperAction

    asset_name: str = "robot"
    joint_names: list[str] = []
    closed_joint_pos: float = 0.0
    open_joint_pos: float = 0.04
    clip: tuple[float, float] = (-1.0, 1.0)
    semantic_closure: bool = False

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SymmetricPandaGripperAction(ActionTerm):
    """One-dimensional absolute openness command for the official Panda gripper.

    The raw scalar action is interpreted in ``[-1, 1]`` and mapped to the same
    target position for both finger joints.  This keeps the policy action space
    at 7 arm deltas + 1 symmetric gripper openness value.
    """

    def __init__(self, cfg: SymmetricPandaGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._robot_entity_cfg = SceneEntityCfg(cfg.asset_name, joint_names=cfg.joint_names)
        try:
            self._robot_entity_cfg.resolve(env.scene)
        except Exception as exc:
            available = tuple(getattr(self._robot.data, "joint_names", ()))
            raise RuntimeError(
                "official_panda_gripper requires the official Panda finger joints "
                f"{tuple(cfg.joint_names)!r}; available joints are {available!r}"
            ) from exc

        self._joint_ids = self._robot_entity_cfg.joint_ids
        self._joint_names = self._robot_entity_cfg.joint_names
        self._num_joints = len(self._joint_ids)
        if self._num_joints != 2:
            raise RuntimeError(
                "SymmetricPandaGripperAction expected exactly two Panda finger joints, "
                f"resolved {self._num_joints}: {self._joint_names!r}"
            )

        self._action_dim = 1
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

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self._action_dim:
            raise ValueError(f"Expected gripper action shape (*, 1), got {tuple(actions.shape)}")

        self._raw_actions = actions.clone()
        clipped = torch.clamp(actions[:, :1], self.cfg.clip[0], self.cfg.clip[1])
        normalized = (clipped - self.cfg.clip[0]) / max(self.cfg.clip[1] - self.cfg.clip[0], 1e-6)
        target = self.cfg.closed_joint_pos + normalized * (
            self.cfg.open_joint_pos - self.cfg.closed_joint_pos
        )
        self._processed_actions = target.expand(-1, self._num_joints)

    def apply_actions(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return


@configclass
class SymmetricPandaGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SymmetricPandaGripperAction

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_finger_joint.*"]
    closed_joint_pos: float = 0.0
    open_joint_pos: float = 0.04
    clip: tuple[float, float] = (-1.0, 1.0)

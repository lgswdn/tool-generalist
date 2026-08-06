from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SemanticOneDofGripperAction(ActionTerm):
    """Map one normalized closure command onto an embodiment's physical joint targets.

    The policy convention is fixed across families: ``-1`` is fully open and
    ``+1`` is fully closed. A gripper may have multiple driven joints, but it
    still exposes exactly one policy control degree of freedom.
    """

    def __init__(self, cfg: SemanticOneDofGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._fixed_base_contract_checked = False
        entity_cfg = SceneEntityCfg(cfg.asset_name, joint_names=cfg.joint_names)
        try:
            entity_cfg.resolve(env.scene)
        except Exception as exc:
            available = tuple(getattr(self._robot.data, "joint_names", ()))
            raise RuntimeError(
                f"one_dof_gripper requires joints {tuple(cfg.joint_names)!r}; available={available!r}"
            ) from exc
        self._joint_ids = entity_cfg.joint_ids
        self._joint_names = entity_cfg.joint_names
        joint_count = len(self._joint_ids)
        if joint_count == 0:
            raise RuntimeError("SemanticOneDofGripperAction resolved no physical joints")
        if len(cfg.open_joint_positions) != joint_count or len(cfg.closed_joint_positions) != joint_count:
            raise ValueError("Open/closed joint-position vectors must match resolved physical joints")
        if cfg.clip[1] <= cfg.clip[0]:
            raise ValueError("SemanticOneDofGripperActionCfg.clip must be increasing")

        self._action_dim = 1
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, joint_count, device=self.device)
        self._open = torch.tensor(cfg.open_joint_positions, dtype=torch.float32, device=self.device).view(1, -1)
        self._closed = torch.tensor(cfg.closed_joint_positions, dtype=torch.float32, device=self.device).view(1, -1)
        if torch.allclose(self._open, self._closed):
            raise ValueError("One-DoF gripper open and closed configurations must differ")
        env._one_dof_gripper_commanded_closure = torch.zeros(self.num_envs, 1, device=self.device)

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
        self._validate_fixed_base_contract()
        if actions.shape != (self.num_envs, 1):
            raise ValueError(f"Expected one-DoF gripper action {(self.num_envs, 1)}, got {tuple(actions.shape)}")
        self._raw_actions = actions.clone()
        clipped = torch.clamp(actions, self.cfg.clip[0], self.cfg.clip[1])
        closure = (clipped - self.cfg.clip[0]) / (self.cfg.clip[1] - self.cfg.clip[0])
        self._processed_actions = self._open + closure * (self._closed - self._open)
        self._env._one_dof_gripper_commanded_closure = closure.detach()

    def apply_actions(self) -> None:
        self._validate_fixed_base_contract()
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._robot.write_data_to_sim()

    def _validate_fixed_base_contract(self) -> None:
        if self._fixed_base_contract_checked:
            return
        root_view = getattr(self._robot, "root_physx_view", None)
        metatype = getattr(root_view, "shared_metatype", None)
        if metatype is None or not hasattr(metatype, "fixed_base"):
            raise RuntimeError(
                "one_dof_gripper could not verify the fixed-base articulation after PhysX initialization"
            )
        if not bool(metatype.fixed_base):
            raise RuntimeError(
                "one_dof_gripper requires a fixed-base USD; rerun scripts/convert_one_dof_gripper.py"
            )
        self._fixed_base_contract_checked = True

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        target = self._env._one_dof_gripper_commanded_closure
        if env_ids is None:
            target.zero_()
        else:
            target[env_ids] = 0.0


@configclass
class SemanticOneDofGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SemanticOneDofGripperAction

    asset_name: str = "robot"
    joint_names: list[str] = []
    open_joint_positions: list[float] = []
    closed_joint_positions: list[float] = []
    clip: tuple[float, float] = (-1.0, 1.0)

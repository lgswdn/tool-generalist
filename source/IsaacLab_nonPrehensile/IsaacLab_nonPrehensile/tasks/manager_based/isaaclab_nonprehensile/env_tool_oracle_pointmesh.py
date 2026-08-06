"""Generated-gripper environment exposing privileged unsigned mesh distance."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    _PHYSICS_OBSERVATION_FIELDS,
    _RL_CONTRACT,
    _RL_RUNTIME_SPEC,
    _USE_BARE_FRANKA,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool_unstable import (
    NonPrehensileUnstableEnv,
    NonPrehensileUnstableEnvCfg,
)


def _validate_runtime() -> None:
    backend = str(_RL_RUNTIME_SPEC.get("policy_params", {}).get("encoder_backend", ""))
    layout = tuple(_RL_RUNTIME_SPEC.get("observation_layout", ()))
    if backend != "oracle_pointmesh_pointnet":
        raise RuntimeError(
            "generated-gripper-oracle-pointmesh-v0 requires "
            "encoder_backend=oracle_pointmesh_pointnet"
        )
    if not bool(
        getattr(_RL_CONTRACT.observation, "include_oracle_mesh_unsigned_distance", False)
    ):
        raise RuntimeError(
            "generated-gripper-oracle-pointmesh-v0 requires "
            "include_oracle_mesh_unsigned_distance=True"
        )
    if len(layout) < 3 or layout[2] != "oracle_mesh_unsigned_distance":
        raise RuntimeError(
            "generated-gripper-oracle-pointmesh-v0 requires "
            "oracle_mesh_unsigned_distance at layout index 2"
        )


_validate_runtime()


class OraclePointMeshNonPrehensileUnstableEnv(NonPrehensileUnstableEnv):
    """Random-pose task exposing exact unsigned distance to opposite meshes."""


@configclass
class OraclePointMeshObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        object_cloud = ObsTerm(
            func=mdp.get_object_pointcloud_in_env_frame,
        ) if _RL_CONTRACT.observation.include_object_cloud else None

        tool_cloud = ObsTerm(
            func=mdp.get_tool_pointcloud_in_env_frame,
        ) if (_RL_CONTRACT.observation.include_tool_cloud and not _USE_BARE_FRANKA) else None

        oracle_mesh_unsigned_distance = ObsTerm(
            func=mdp.get_oracle_mesh_unsigned_distance
        )

        object_bbox_center = ObsTerm(
            func=mdp.get_obj_bbox_center,
        ) if _RL_CONTRACT.observation.include_bbox_centers else None

        tool_bbox_center = ObsTerm(
            func=mdp.get_tool_bbox_center,
        ) if (
            _RL_CONTRACT.observation.include_bbox_centers
            and _RL_CONTRACT.observation.include_tool_cloud
            and not _USE_BARE_FRANKA
        ) else None

        hand_state = ObsTerm(
            func=mdp.hand_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        robot_state = ObsTerm(
            func=mdp.robot_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        previous_action = ObsTerm(func=mdp.last_action)
        relative_goal_pose = ObsTerm(
            func=mdp.rel_pose_goal,
            params={"command_name": "target_object_pose"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        task_embedding = ObsTerm(
            func=mdp.target_pose_task_embedding,
            params={"command_name": "target_object_pose"},
        ) if _RL_CONTRACT.observation.task_embedding_dim > 0 else None
        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"field_names": _PHYSICS_OBSERVATION_FIELDS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class OraclePointMeshNonPrehensileUnstableEnvCfg(NonPrehensileUnstableEnvCfg):
    observations: OraclePointMeshObservationsCfg = OraclePointMeshObservationsCfg()

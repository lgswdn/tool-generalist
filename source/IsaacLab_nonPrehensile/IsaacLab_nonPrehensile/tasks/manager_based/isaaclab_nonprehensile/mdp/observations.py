# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from scipy.spatial.transform import Rotation as R
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from utils.assets import (
    GeneratedGripperAsset,
    PrismaticJointSpec,
    RigidTransformSpec,
)

def profile_obs(fn):
    return fn

_OFFICIAL_PANDA_GRIPPER_MODE = "official_panda_gripper"
_OFFICIAL_PANDA_FINGERTIP_BODY_NAMES = ("panda_leftfingertip", "panda_rightfingertip")
_OFFICIAL_PANDA_FINGER_BODY_NAMES = ("panda_leftfinger", "panda_rightfinger")
_OFFICIAL_PANDA_PALM_BODY_NAMES = ("panda_hand",)
_OFFICIAL_PANDA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
_OFFICIAL_GRIPPER_NUM_BUCKETS = 64
_OFFICIAL_GRIPPER_OPEN_JOINT_POS = 0.04
_OFFICIAL_GRIPPER_CLOUD_SOURCE = "official_panda_gripper_kinematic_mesh_rx90_v2"
_OFFICIAL_GRIPPER_CLOUD_CACHE: dict[tuple[str, str, str, int], torch.Tensor] = {}
_OFFICIAL_GRIPPER_FINGER_MOUNT_OFFSET_Y = 0.0584
_OFFICIAL_GRIPPER_FINGER_TIP_OFFSET_XYZ = (0.0, 0.0, 0.045)
_GENERATED_GRIPPER_MODE = "generated_gripper"
_GENERATED_GRIPPER_CLOUD_SOURCE = "gripper_cloud_cache_v1"
_GENERATED_GRIPPER_CLOUD_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_ONE_DOF_GRIPPER_MODE = "one_dof_gripper"
_ONE_DOF_GRIPPER_CLOUD_SOURCE = "gripper_cloud_cache_v1"
_ONE_DOF_GRIPPER_STATE_CLOUD_CACHE: dict[tuple[str, str, int], torch.Tensor] = {}
_ONE_DOF_CANONICAL_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_ORACLE_MESH_SDF_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_ORACLE_GRIPPER_LINK_MESH_CACHE: dict[tuple, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

_HAND_GOAL_MEAN = torch.tensor([0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # z mean = 0.15
_HAND_GOAL_STD = torch.tensor([0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def _dbg(env: "ManagerBasedRLEnv", name: str, tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _dbg_cloud(env: "ManagerBasedRLEnv", name: str, cloud_env: torch.Tensor) -> None:
    return None


def _bbox_center_env(pointcloud_env: torch.Tensor) -> torch.Tensor:
    bbox_min = pointcloud_env.min(dim=1).values
    bbox_max = pointcloud_env.max(dim=1).values
    return (bbox_min + bbox_max) * 0.5


def _bbox_extent_env(pointcloud_env: torch.Tensor) -> torch.Tensor:
    bbox_min = pointcloud_env.min(dim=1).values
    bbox_max = pointcloud_env.max(dim=1).values
    return bbox_max - bbox_min


def _is_official_panda_gripper(env: "ManagerBasedRLEnv") -> bool:
    return getattr(env.cfg, "robot_mode", "tool") == _OFFICIAL_PANDA_GRIPPER_MODE


def _is_generated_gripper(env: "ManagerBasedRLEnv") -> bool:
    return getattr(env.cfg, "robot_mode", "tool") == _GENERATED_GRIPPER_MODE


def _is_one_dof_gripper(env: "ManagerBasedRLEnv") -> bool:
    return getattr(env.cfg, "robot_mode", "tool") == _ONE_DOF_GRIPPER_MODE


def _resolve_robot_bodies(
    env: "ManagerBasedRLEnv",
    *,
    attr_name: str,
    body_names: tuple[str, ...],
) -> SceneEntityCfg:
    cfg = getattr(env, attr_name, None)
    if cfg is not None:
        return cfg

    cfg = SceneEntityCfg("robot", body_names=list(body_names))
    try:
        cfg.resolve(env.scene)
    except Exception as exc:
        robot = env.scene["robot"]
        available = tuple(getattr(robot.data, "body_names", ()))
        raise RuntimeError(
            f"{_OFFICIAL_PANDA_GRIPPER_MODE} requires official Panda bodies "
            f"{body_names!r}; available robot bodies are {available!r}"
        ) from exc

    setattr(env, attr_name, cfg)
    return cfg


def _resolve_generated_robot_bodies(
    env: "ManagerBasedRLEnv",
    *,
    body_names: tuple[str, ...],
    expected_count: int,
) -> SceneEntityCfg:
    cache = getattr(env, "_generated_gripper_body_cfg_cache", None)
    if cache is None:
        cache = {}
        env._generated_gripper_body_cfg_cache = cache
    key = tuple(body_names)
    cached = cache.get(key)
    if cached is not None:
        return cached

    cfg = SceneEntityCfg("robot", body_names=list(body_names))
    try:
        cfg.resolve(env.scene)
    except Exception as exc:
        robot = env.scene["robot"]
        available = tuple(getattr(robot.data, "body_names", ()))
        raise RuntimeError(
            f"{_GENERATED_GRIPPER_MODE} requires manifest robot bodies "
            f"{body_names!r}; available robot bodies are {available!r}"
        ) from exc

    if len(cfg.body_ids) != expected_count:
        raise RuntimeError(
            f"{_GENERATED_GRIPPER_MODE} expected {expected_count} bodies for "
            f"{body_names!r}, resolved {len(cfg.body_ids)}: {cfg.body_names!r}"
        )
    cache[key] = cfg
    return cfg


def _resolve_generated_robot_joints(
    env: "ManagerBasedRLEnv",
    *,
    joint_names: tuple[str, ...],
    expected_count: int,
) -> SceneEntityCfg:
    cache = getattr(env, "_generated_gripper_joint_cfg_cache", None)
    if cache is None:
        cache = {}
        env._generated_gripper_joint_cfg_cache = cache
    key = tuple(joint_names)
    cached = cache.get(key)
    if cached is not None:
        return cached

    cfg = SceneEntityCfg("robot", joint_names=list(joint_names))
    try:
        cfg.resolve(env.scene)
    except Exception as exc:
        robot = env.scene["robot"]
        available = tuple(getattr(robot.data, "joint_names", ()))
        raise RuntimeError(
            f"{_GENERATED_GRIPPER_MODE} requires manifest robot joints "
            f"{joint_names!r}; available robot joints are {available!r}"
        ) from exc

    if len(cfg.joint_ids) != expected_count:
        raise RuntimeError(
            f"{_GENERATED_GRIPPER_MODE} expected {expected_count} joints for "
            f"{joint_names!r}, resolved {len(cfg.joint_ids)}: {cfg.joint_names!r}"
        )
    cache[key] = cfg
    return cfg


def _resolve_one_dof_robot_bodies(
    env: "ManagerBasedRLEnv", *, body_names: tuple[str, ...]
) -> SceneEntityCfg:
    cache = getattr(env, "_one_dof_gripper_body_cfg_cache", None)
    if cache is None:
        cache = {}
        env._one_dof_gripper_body_cfg_cache = cache
    key = tuple(body_names)
    if key not in cache:
        cfg = SceneEntityCfg("robot", body_names=list(body_names))
        try:
            cfg.resolve(env.scene)
        except Exception as exc:
            available = tuple(getattr(env.scene["robot"].data, "body_names", ()))
            raise RuntimeError(
                f"one_dof_gripper requires bodies {body_names!r}; available={available!r}"
            ) from exc
        if len(cfg.body_ids) != len(body_names):
            raise RuntimeError(f"one_dof_gripper could not resolve bodies exactly: {body_names!r}")
        cache[key] = cfg
    return cache[key]


def _resolve_one_dof_robot_joints(
    env: "ManagerBasedRLEnv", *, joint_names: tuple[str, ...]
) -> SceneEntityCfg:
    cache = getattr(env, "_one_dof_gripper_joint_cfg_cache", None)
    if cache is None:
        cache = {}
        env._one_dof_gripper_joint_cfg_cache = cache
    key = tuple(joint_names)
    if key not in cache:
        cfg = SceneEntityCfg("robot", joint_names=list(joint_names))
        try:
            cfg.resolve(env.scene)
        except Exception as exc:
            available = tuple(getattr(env.scene["robot"].data, "joint_names", ()))
            raise RuntimeError(
                f"one_dof_gripper requires joints {joint_names!r}; available={available!r}"
            ) from exc
        if len(cfg.joint_ids) != len(joint_names):
            raise RuntimeError(f"one_dof_gripper could not resolve joints exactly: {joint_names!r}")
        cache[key] = cfg
    return cache[key]


def _try_resolve_robot_bodies(
    env: "ManagerBasedRLEnv",
    *,
    attr_name: str,
    missing_attr_name: str,
    body_names: tuple[str, ...],
) -> SceneEntityCfg | None:
    cfg = getattr(env, attr_name, None)
    if cfg is not None:
        return cfg
    if getattr(env, missing_attr_name, False):
        return None

    cfg = SceneEntityCfg("robot", body_names=list(body_names))
    try:
        cfg.resolve(env.scene)
    except Exception:
        setattr(env, missing_attr_name, True)
        return None

    setattr(env, attr_name, cfg)
    return cfg


def _resolve_robot_joints(
    env: "ManagerBasedRLEnv",
    *,
    attr_name: str,
    joint_names: tuple[str, ...] | list[str],
    expected_count: int | None = None,
) -> SceneEntityCfg:
    cfg = getattr(env, attr_name, None)
    if cfg is not None:
        return cfg

    cfg = SceneEntityCfg("robot", joint_names=list(joint_names))
    try:
        cfg.resolve(env.scene)
    except Exception as exc:
        robot = env.scene["robot"]
        available = tuple(getattr(robot.data, "joint_names", ()))
        raise RuntimeError(
            f"{_OFFICIAL_PANDA_GRIPPER_MODE} requires official Panda joints "
            f"{tuple(joint_names)!r}; available robot joints are {available!r}"
        ) from exc

    if expected_count is not None and len(cfg.joint_ids) != expected_count:
        raise RuntimeError(
            f"{_OFFICIAL_PANDA_GRIPPER_MODE} expected {expected_count} joints for "
            f"{tuple(joint_names)!r}, resolved {len(cfg.joint_ids)}: {cfg.joint_names!r}"
        )

    setattr(env, attr_name, cfg)
    return cfg


def get_official_panda_fingertip_center_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return midpoint of the official Panda fingertip positions."""

    robot = env.scene["robot"]
    fingertip_cfg = _try_resolve_robot_bodies(
        env,
        attr_name="_official_panda_fingertip_body_cfg",
        missing_attr_name="_official_panda_fingertip_body_cfg_missing",
        body_names=_OFFICIAL_PANDA_FINGERTIP_BODY_NAMES,
    )
    if fingertip_cfg is not None:
        if len(fingertip_cfg.body_ids) != 2:
            raise RuntimeError(
                f"{_OFFICIAL_PANDA_GRIPPER_MODE} expected two fingertip bodies "
                f"{_OFFICIAL_PANDA_FINGERTIP_BODY_NAMES!r}, resolved {fingertip_cfg.body_ids!r}"
            )
        fingertip_pos_w = robot.data.body_state_w[:, fingertip_cfg.body_ids, :3]
        return fingertip_pos_w.mean(dim=1)

    finger_pos_w, finger_quat_w = _get_official_panda_finger_poses_w(env)
    tip_offset = torch.tensor(
        _OFFICIAL_GRIPPER_FINGER_TIP_OFFSET_XYZ,
        dtype=finger_pos_w.dtype,
        device=finger_pos_w.device,
    )
    num_envs = finger_pos_w.shape[0]
    finger_rot_w = matrix_from_quat(finger_quat_w.reshape(-1, 4)).reshape(num_envs, 2, 3, 3)
    fingertip_pos_w = (
        finger_pos_w
        + torch.bmm(
            finger_rot_w.reshape(-1, 3, 3),
            tip_offset.view(1, 3, 1).expand(num_envs * 2, -1, -1),
        ).reshape(num_envs, 2, 3)
    )
    return fingertip_pos_w.mean(dim=1)


def _get_official_panda_finger_poses_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    fingers_cfg = _resolve_robot_bodies(
        env,
        attr_name="_official_panda_finger_body_cfg",
        body_names=_OFFICIAL_PANDA_FINGER_BODY_NAMES,
    )
    if len(fingers_cfg.body_ids) != 2:
        raise RuntimeError(
            f"{_OFFICIAL_PANDA_GRIPPER_MODE} expected two finger bodies "
            f"{_OFFICIAL_PANDA_FINGER_BODY_NAMES!r}, resolved {fingers_cfg.body_ids!r}"
        )
    state_w = robot.data.body_state_w[:, fingers_cfg.body_ids, :]
    return state_w[..., :3], state_w[..., 3:7]


def _get_official_panda_palm_pose_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    palm_cfg = _resolve_robot_bodies(
        env,
        attr_name="_official_panda_palm_body_cfg",
        body_names=_OFFICIAL_PANDA_PALM_BODY_NAMES,
    )
    palm_id = palm_cfg.body_ids[0]
    return robot.data.body_state_w[:, palm_id, :3], robot.data.body_state_w[:, palm_id, 3:7]


def _generated_gripper_env_groups(env: "ManagerBasedRLEnv") -> dict[int, list[int]]:
    cached = getattr(env, "_generated_gripper_env_groups_cache", None)
    if cached is not None:
        return cached

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_generated_gripper_index_for_env,
    )

    groups: dict[int, list[int]] = {}
    for env_id in range(env.num_envs):
        gripper_index = get_generated_gripper_index_for_env(env_id)
        groups.setdefault(gripper_index, []).append(env_id)
    env._generated_gripper_env_groups_cache = groups
    return groups


def _generated_gripper_runtime_metadata(env: "ManagerBasedRLEnv") -> dict[str, object]:
    """Materialize immutable per-env generated-gripper metadata on the GPU once."""

    cached = getattr(env, "_generated_gripper_runtime_metadata_cache", None)
    if cached is not None:
        return cached

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        GENERATED_GRIPPER_ASSET_INDICES_BY_ENV,
        GENERATED_GRIPPER_DATA,
    )

    num_envs = int(env.num_envs)
    if len(GENERATED_GRIPPER_ASSET_INDICES_BY_ENV) != num_envs:
        raise RuntimeError(
            "Generated-gripper assignment length does not match the active environment: "
            f"{len(GENERATED_GRIPPER_ASSET_INDICES_BY_ENV)} != {num_envs}"
        )
    if not GENERATED_GRIPPER_DATA:
        raise RuntimeError("Generated-gripper runtime metadata is empty")

    assigned_asset_indices = [
        int(value) for value in GENERATED_GRIPPER_ASSET_INDICES_BY_ENV
    ]
    if any(
        value < 0 or value >= len(GENERATED_GRIPPER_DATA)
        for value in assigned_asset_indices
    ):
        raise RuntimeError(
            "Generated-gripper assignment contains an out-of-range asset index"
        )
    active_asset_indices = sorted(set(assigned_asset_indices))
    compact_index_by_asset = {
        asset_index: compact_index
        for compact_index, asset_index in enumerate(active_asset_indices)
    }
    assets = tuple(
        GENERATED_GRIPPER_DATA[asset_index]
        for asset_index in active_asset_indices
    )
    compact_assignment = [
        compact_index_by_asset[asset_index]
        for asset_index in assigned_asset_indices
    ]

    palm_body_ids: list[int] = []
    finger_body_ids: list[list[int]] = []
    fingertip_body_ids: list[list[int]] = []
    finger_joint_ids: list[list[int]] = []
    fingertip_local_offsets: list[tuple[tuple[float, float, float], ...]] = []
    fingertip_from_body: list[bool] = []
    open_joint_positions: list[float] = []

    for gripper in assets:
        palm_cfg = _resolve_generated_robot_bodies(
            env,
            body_names=(gripper.palm_body_name,),
            expected_count=1,
        )
        finger_cfg = _resolve_generated_robot_bodies(
            env,
            body_names=gripper.finger_body_names,
            expected_count=2,
        )
        joint_cfg = _resolve_generated_robot_joints(
            env,
            joint_names=gripper.finger_joint_names,
            expected_count=2,
        )
        palm_body_ids.append(int(palm_cfg.body_ids[0]))
        finger_body_ids.append([int(value) for value in finger_cfg.body_ids])
        finger_joint_ids.append([int(value) for value in joint_cfg.joint_ids])
        open_joint_positions.append(float(gripper.open_joint_pos))

        if gripper.fingertip_body_names is not None:
            tip_cfg = _resolve_generated_robot_bodies(
                env,
                body_names=gripper.fingertip_body_names,
                expected_count=2,
            )
            fingertip_body_ids.append([int(value) for value in tip_cfg.body_ids])
            fingertip_local_offsets.append(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            fingertip_from_body.append(True)
        else:
            if gripper.fingertip_local_offsets is None:
                raise RuntimeError(
                    f"{_GENERATED_GRIPPER_MODE} gripper {gripper.gripper_id!r} must provide "
                    "fingertip_body_names or fingertip_local_offsets"
                )
            # Valid placeholder ids keep the mixed-metadata gather branch tensor-only.
            fingertip_body_ids.append([int(value) for value in finger_cfg.body_ids])
            fingertip_local_offsets.append(gripper.fingertip_local_offsets)
            fingertip_from_body.append(False)

    device = env.device
    asset_indices = torch.as_tensor(
        compact_assignment,
        dtype=torch.long,
        device=device,
    )
    env_indices = torch.arange(num_envs, dtype=torch.long, device=device)

    def per_env(values, *, dtype):
        by_asset = torch.as_tensor(values, dtype=dtype, device=device)
        return by_asset[asset_indices].contiguous()

    fingertip_from_body_by_env = per_env(fingertip_from_body, dtype=torch.bool)
    cached = {
        "assets": assets,
        "asset_indices": asset_indices,
        "env_indices": env_indices,
        "palm_body_ids": per_env(palm_body_ids, dtype=torch.long),
        "finger_body_ids": per_env(finger_body_ids, dtype=torch.long),
        "fingertip_body_ids": per_env(fingertip_body_ids, dtype=torch.long),
        "finger_joint_ids": per_env(finger_joint_ids, dtype=torch.long),
        "fingertip_local_offsets": per_env(
            fingertip_local_offsets,
            dtype=torch.float32,
        ),
        "fingertip_from_body": fingertip_from_body_by_env,
        "has_body_fingertips": bool(any(fingertip_from_body)),
        "all_body_fingertips": bool(all(fingertip_from_body)),
        "open_joint_positions": per_env(open_joint_positions, dtype=torch.float32),
    }
    env._generated_gripper_runtime_metadata_cache = cached
    return cached


def _one_dof_gripper_env_groups(env: "ManagerBasedRLEnv") -> dict[int, list[int]]:
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_one_dof_gripper_index_for_env,
    )

    groups: dict[int, list[int]] = {}
    for env_id in range(env.num_envs):
        index = get_one_dof_gripper_index_for_env(env_id)
        groups.setdefault(index, []).append(env_id)
    return groups


def get_one_dof_gripper_interaction_center_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the physical interaction center for every one-DoF gripper."""

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_one_dof_gripper_data_for_env,
    )

    robot = env.scene["robot"]
    out = torch.empty((env.num_envs, 3), device=env.device, dtype=robot.data.body_state_w.dtype)
    for _, env_ids in _one_dof_gripper_env_groups(env).items():
        asset = get_one_dof_gripper_data_for_env(env_ids[0])
        indices = torch.tensor(env_ids, dtype=torch.long, device=env.device)
        if asset.category == "two_finger_revolute":
            body_cfg = _resolve_one_dof_robot_bodies(
                env, body_names=("left_top_link", "right_top_link")
            )
            states = robot.data.body_state_w[indices][:, body_cfg.body_ids, :]
            tip_length = (
                float(asset.params["tip_length"])
                if asset.params["tip_shape"] != "none"
                else 0.0
            )
            local_tip = torch.tensor(
                [0.0, 0.0, float(asset.params["top_size"][2]) + tip_length],
                dtype=states.dtype,
                device=env.device,
            )
            rotations = matrix_from_quat(states[..., 3:7].reshape(-1, 4)).reshape(
                len(env_ids), 2, 3, 3
            )
            tips_w = states[..., :3] + torch.matmul(
                rotations,
                local_tip.view(1, 1, 3, 1),
            ).squeeze(-1)
            out[indices] = tips_w.mean(dim=1)
            continue
        body_cfg = _resolve_one_dof_robot_bodies(
            env, body_names=(asset.grasp_frame_body_name,)
        )
        state = robot.data.body_state_w[indices, body_cfg.body_ids[0], :]
        offset = torch.tensor(
            asset.grasp_frame_offset.translation,
            dtype=state.dtype,
            device=env.device,
        )
        rotation = matrix_from_quat(state[:, 3:7])
        out[indices] = state[:, :3] + torch.bmm(
            rotation, offset.view(1, 3, 1).expand(len(env_ids), -1, -1)
        ).squeeze(-1)
    return out


def _get_generated_gripper_palm_pose_w(
    env: "ManagerBasedRLEnv",
    gripper: GeneratedGripperAsset,
    env_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    palm_cfg = _resolve_generated_robot_bodies(
        env,
        body_names=(gripper.palm_body_name,),
        expected_count=1,
    )
    palm_id = palm_cfg.body_ids[0]
    state_w = robot.data.body_state_w[env_indices, palm_id, :]
    return state_w[:, :3], state_w[:, 3:7]


def get_generated_gripper_fingertip_center_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return midpoint of generated gripper fingertips from explicit metadata."""

    robot = env.scene["robot"]
    metadata = _generated_gripper_runtime_metadata(env)
    env_indices = metadata["env_indices"]

    if metadata["all_body_fingertips"]:
        fingertip_state_w = robot.data.body_state_w[
            env_indices.unsqueeze(1),
            metadata["fingertip_body_ids"],
        ]
        return fingertip_state_w[..., :3].mean(dim=1)

    finger_state_w = robot.data.body_state_w[
        env_indices.unsqueeze(1),
        metadata["finger_body_ids"],
    ]
    finger_pos_w = finger_state_w[..., :3]
    finger_rot_w = matrix_from_quat(
        finger_state_w[..., 3:7].reshape(-1, 4)
    ).reshape(env.num_envs, 2, 3, 3)
    offsets = metadata["fingertip_local_offsets"].to(dtype=finger_pos_w.dtype)
    local_fingertip_pos_w = finger_pos_w + torch.matmul(
        finger_rot_w,
        offsets.unsqueeze(-1),
    ).squeeze(-1)
    local_center_w = local_fingertip_pos_w.mean(dim=1)

    if not metadata["has_body_fingertips"]:
        return local_center_w

    fingertip_state_w = robot.data.body_state_w[
        env_indices.unsqueeze(1),
        metadata["fingertip_body_ids"],
    ]
    body_center_w = fingertip_state_w[..., :3].mean(dim=1)
    return torch.where(
        metadata["fingertip_from_body"].unsqueeze(1),
        body_center_w,
        local_center_w,
    )


def get_head_area_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the head area center in world space for every environment.

    Uses the tool body (link_coacd_convex_piece_0) pose from the ee_frame sensor
    plus the per-env local offset stored in env._head_area_offsets (computed once
    in post_reset via OBJ bounding-box queries).

    Returns:
        torch.Tensor: Shape (num_envs, 3) – world-space positions.
    """
    if _is_official_panda_gripper(env):
        fingertip_center_w = get_official_panda_fingertip_center_pos_w(env)
        if getattr(env.cfg, "visualize_head_area_center", True):
            _visualize_head_area_center(env, fingertip_center_w)
        return fingertip_center_w
    if _is_generated_gripper(env):
        fingertip_center_w = get_generated_gripper_fingertip_center_pos_w(env)
        if getattr(env.cfg, "visualize_head_area_center", True):
            _visualize_head_area_center(env, fingertip_center_w)
        return fingertip_center_w
    if _is_one_dof_gripper(env):
        center_w = get_one_dof_gripper_interaction_center_pos_w(env)
        if getattr(env.cfg, "visualize_head_area_center", True):
            _visualize_head_area_center(env, center_w)
        return center_w

    ee_frame = env.scene["ee_frame"]
    tool_pos_w = ee_frame.data.target_pos_w[..., 0, :]   # (N, 3)

    if not (hasattr(env, "_head_area_offsets") and env._head_area_offsets is not None):
        return tool_pos_w

    tool_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (N, 4)
    R = matrix_from_quat(tool_quat_w)                      # (N, 3, 3)
    offset = env._head_area_offsets                         # (N, 3)
    head_pos_w = tool_pos_w + torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)

    if getattr(env.cfg, "visualize_head_area_center", True):
        _visualize_head_area_center(env, head_pos_w)

    return head_pos_w


def _visualize_head_area_center(env: "ManagerBasedRLEnv", head_pos_w: torch.Tensor):
    """Draw a small red sphere at the head area center for each environment."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    if not hasattr(env, "_head_area_visualizer"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/HeadAreaCenter",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        env._head_area_visualizer = VisualizationMarkers(marker_cfg)

    num_envs = head_pos_w.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=head_pos_w.device).expand(num_envs, -1)
    env._head_area_visualizer.visualize(head_pos_w, orientations)


@profile_obs
def hand_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hand state observation (9D: position[3] + rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9) containing [x,y,z, r11,r12,r13, r21,r22,r23]
    """
    ee_frame = env.scene[ee_frame_cfg.name]

    # Head area world position (tool body + rotated per-env local offset)
    ee_pos_w = get_head_area_pos_w(env)          # (num_envs, 3)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    # Convert to environment coordinates
    ee_pos_env = ee_pos_w - env.scene.env_origins
    
    # Rotation as 6D
    rot_matrix = matrix_from_quat(ee_quat_w)  # (num_envs, 3, 3)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    hand_state_9d = torch.cat([ee_pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters (similar to corn config)
        device = hand_state_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        hand_state_9d = (hand_state_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "hand_state", hand_state_9d)


def _cross_embodiment_generated_robot_state(env, asset) -> torch.Tensor:
    """Return the same 14D arm + 4D semantic gripper state used by Robotiq ranks."""

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        GENERATED_GRIPPER_FINGER_JOINT_NAMES,
        GENERATED_GRIPPER_OPEN_JOINT_POS,
    )

    arm_cfg = _resolve_robot_joints(
        env,
        attr_name="_cross_embodiment_generated_arm_joint_cfg",
        joint_names=["panda_joint.*"],
        expected_count=7,
    )
    finger_cfg = _resolve_generated_robot_joints(
        env,
        joint_names=GENERATED_GRIPPER_FINGER_JOINT_NAMES,
        expected_count=2,
    )
    arm_pos = asset.data.joint_pos[:, arm_cfg.joint_ids]
    arm_vel = asset.data.joint_vel[:, arm_cfg.joint_ids]
    if getattr(env.cfg, "normalize_observations", True):
        defaults = asset.data.default_joint_pos[:, arm_cfg.joint_ids]
        limits = asset.data.soft_joint_pos_limits[:, arm_cfg.joint_ids, :]
        half_ranges = torch.clamp((limits[..., 1] - limits[..., 0]) * 0.5, min=1e-6)
        arm_pos = torch.clamp((arm_pos - defaults) / half_ranges, -1.0, 1.0)
        velocity_limits = torch.clamp(
            asset.data.soft_joint_vel_limits[:, arm_cfg.joint_ids], min=1e-6
        )
        arm_vel = (torch.clamp(arm_vel / velocity_limits, -1.0, 1.0) + 1.0) * 0.5

    finger_ids = finger_cfg.joint_ids
    span = max(float(GENERATED_GRIPPER_OPEN_JOINT_POS), 1e-6)
    mean_pos = asset.data.joint_pos[:, finger_ids].mean(dim=1)
    mean_vel = asset.data.joint_vel[:, finger_ids].mean(dim=1)
    closure = torch.clamp(
        (float(GENERATED_GRIPPER_OPEN_JOINT_POS) - mean_pos) / span, 0.0, 1.0
    )
    closure_vel = torch.clamp(-mean_vel / span, -1.0, 1.0)
    commanded = getattr(env, "_generated_gripper_commanded_closure", None)
    target = closure if commanded is None else commanded[:, 0]
    tracking_error = torch.clamp(target - closure, -1.0, 1.0)
    mean_effort = asset.data.applied_torque[:, finger_ids].mean(dim=1)
    effort_limit = torch.clamp(
        asset.data.joint_effort_limits[:, finger_ids].mean(dim=1), min=1e-6
    )
    effort = torch.clamp(mean_effort / effort_limit, -1.0, 1.0)
    semantic = torch.stack(
        (closure, (closure_vel + 1.0) * 0.5, tracking_error, effort), dim=1
    )
    return torch.cat((arm_pos, arm_vel, semantic), dim=1)


@profile_obs
def robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot state observation.

    Tool-only robots use 14D arm state. Gripper robots use 18D arm+finger state.
    
    Returns:
        torch.Tensor: Shape (num_envs, 14) or (num_envs, 18)
    """
    asset = env.scene[asset_cfg.name]

    if (
        _is_generated_gripper(env)
        and getattr(env.cfg, "requested_robot_mode", "") == "cross_embodiment_gripper"
    ):
        return _dbg(
            env,
            "robot_state",
            _cross_embodiment_generated_robot_state(env, asset),
        )

    if _is_one_dof_gripper(env):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_one_dof_gripper_data_for_env,
        )

        arm_cfg = _resolve_one_dof_robot_joints(env, joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)))
        arm_pos = asset.data.joint_pos[:, arm_cfg.joint_ids]
        arm_vel = asset.data.joint_vel[:, arm_cfg.joint_ids]
        normalize = getattr(env.cfg, "normalize_observations", True)
        if normalize:
            defaults = asset.data.default_joint_pos[:, arm_cfg.joint_ids]
            limits = asset.data.soft_joint_pos_limits[:, arm_cfg.joint_ids, :]
            half_ranges = torch.clamp((limits[..., 1] - limits[..., 0]) * 0.5, min=1e-6)
            arm_pos = torch.clamp((arm_pos - defaults) / half_ranges, -1.0, 1.0)
            velocity_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, arm_cfg.joint_ids], min=1e-6)
            arm_vel = (torch.clamp(arm_vel / velocity_limits, -1.0, 1.0) + 1.0) * 0.5

        semantic = torch.empty((env.num_envs, 4), dtype=arm_pos.dtype, device=env.device)
        commanded = getattr(env, "_one_dof_gripper_commanded_closure", None)
        for _, env_ids in _one_dof_gripper_env_groups(env).items():
            gripper = get_one_dof_gripper_data_for_env(env_ids[0])
            indices = torch.tensor(env_ids, dtype=torch.long, device=env.device)
            gripper_cfg = _resolve_one_dof_robot_joints(
                env, joint_names=gripper.actuated_joint_names
            )
            open_pos = torch.tensor(
                gripper.open_joint_positions,
                dtype=arm_pos.dtype,
                device=env.device,
            )
            closed_pos = torch.tensor(
                gripper.closed_joint_positions,
                dtype=arm_pos.dtype,
                device=env.device,
            )
            span = closed_pos - open_pos
            joint_pos = asset.data.joint_pos[indices][:, gripper_cfg.joint_ids]
            joint_vel = asset.data.joint_vel[indices][:, gripper_cfg.joint_ids]
            joint_closure = torch.clamp(
                (joint_pos - open_pos.unsqueeze(0)) / span.unsqueeze(0),
                0.0,
                1.0,
            )
            closure = joint_closure.mean(dim=1)
            closure_vel = torch.clamp(
                (joint_vel / span.unsqueeze(0)).mean(dim=1),
                -1.0,
                1.0,
            )
            target = closure if commanded is None else commanded[indices, 0]
            tracking_error = torch.clamp(target - closure, -1.0, 1.0)
            applied = asset.data.applied_torque[indices][:, gripper_cfg.joint_ids]
            effort_limit = torch.clamp(
                asset.data.joint_effort_limits[indices][:, gripper_cfg.joint_ids],
                min=1e-6,
            )
            effort = torch.clamp(applied / effort_limit, -1.0, 1.0).mean(dim=1)
            semantic[indices] = torch.stack(
                (closure, (closure_vel + 1.0) * 0.5, tracking_error, effort), dim=1
            )
        return _dbg(env, "robot_state", torch.cat((arm_pos, arm_vel, semantic), dim=1))

    if _is_official_panda_gripper(env):
        arm_cfg = _resolve_robot_joints(
            env,
            attr_name="_official_panda_arm_joint_cfg",
            joint_names=["panda_joint.*"],
            expected_count=7,
        )
        finger_cfg = _resolve_robot_joints(
            env,
            attr_name="_official_panda_finger_joint_cfg",
            joint_names=_OFFICIAL_PANDA_FINGER_JOINT_NAMES,
            expected_count=2,
        )
        joint_ids = list(arm_cfg.joint_ids) + list(finger_cfg.joint_ids)

        joint_pos = asset.data.joint_pos[:, joint_ids]
        joint_vel = asset.data.joint_vel[:, joint_ids]

        normalize = getattr(env.cfg, 'normalize_observations', True)
        if normalize:
            default_pos = asset.data.default_joint_pos[:, joint_ids]
            soft_limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
            mins = soft_limits[..., 0]
            maxs = soft_limits[..., 1]
            centers = default_pos
            half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
            pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

            vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, joint_ids], min=1e-6)
            vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
            vel_norm = (vel_norm + 1.0) * 0.5
            return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))

        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))

    if _is_generated_gripper(env):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            GENERATED_GRIPPER_FINGER_JOINT_NAMES,
        )

        arm_cfg = _resolve_robot_joints(
            env,
            attr_name="_generated_gripper_arm_joint_cfg",
            joint_names=["panda_joint.*"],
            expected_count=7,
        )
        finger_cfg = _resolve_generated_robot_joints(
            env,
            joint_names=GENERATED_GRIPPER_FINGER_JOINT_NAMES,
            expected_count=2,
        )
        joint_ids = list(arm_cfg.joint_ids) + list(finger_cfg.joint_ids)

        joint_pos = asset.data.joint_pos[:, joint_ids]
        joint_vel = asset.data.joint_vel[:, joint_ids]

        normalize = getattr(env.cfg, 'normalize_observations', True)
        if normalize:
            default_pos = asset.data.default_joint_pos[:, joint_ids]
            soft_limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
            mins = soft_limits[..., 0]
            maxs = soft_limits[..., 1]
            centers = default_pos
            half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
            pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

            vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, joint_ids], min=1e-6)
            vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
            vel_norm = (vel_norm + 1.0) * 0.5
            return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))

        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))
    
    # Get joint positions and velocities
    joint_pos = asset.data.joint_pos[:, :7]
    joint_vel = asset.data.joint_vel[:, :7]
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Normalize joint positions using soft limits around default pos -> [-1,1]
        default_pos = asset.data.default_joint_pos[:, :7]
        soft_limits = asset.data.soft_joint_pos_limits[:, :7, :]
        mins = soft_limits[..., 0]
        maxs = soft_limits[..., 1]
        centers = default_pos
        half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
        pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

        # Normalize joint velocities to [0,1] using soft velocity limits
        vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, :7], min=1e-6)
        vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
        vel_norm = (vel_norm + 1.0) * 0.5
        
        return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))
    else:
        # Return raw joint states without normalization
        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))


@profile_obs
def object_root_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root linear and angular velocity: [vx, vy, vz, wx, wy, wz]."""

    object: RigidObject = env.scene[object_cfg.name]
    return _dbg(
        env,
        "object_root_velocity",
        torch.cat(
            [
                object.data.root_lin_vel_w[:, :3],
                object.data.root_ang_vel_w[:, :3],
            ],
            dim=1,
        ),
    )


@profile_obs
def abs_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
) -> torch.Tensor:
    """Absolute pose goal observation (9D: target position[3] + target rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9)
    """
    from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
    
    target_goal = env.command_manager.get_command(command_name)
    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    # Command now directly contains quaternions, convert to rotation matrix
    rot_matrix = matrix_from_quat(target_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    goal_9d = torch.cat([target_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use goal-specific normalization parameters (similar to corn config)
        device = goal_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        goal_9d = (goal_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "abs_goal", goal_9d)


def rel_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Relative pose goal observation (9D: goal relative to current object pose)."""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate, matrix_from_quat

    target_goal = env.command_manager.get_command(command_name)  # (num_envs, 7)
    object_pose_7d = object_pose_in_env_frame(env, object_cfg, normalize=False)
    obj_pos_env = object_pose_7d[:, :3]
    obj_quat_w = object_pose_7d[:, 3:7]

    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    rel_pos = target_pos - obj_pos_env
    current_quat_inv = quat_conjugate(obj_quat_w)
    rel_quat = quat_mul(target_quat, current_quat_inv)
    rot_matrix = matrix_from_quat(rel_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine relative position and rotation
    rel_pose_9d = torch.cat([rel_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for relative pose goal
        device = rel_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        rel_pose_9d = (rel_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return rel_pose_9d


PHYS_PARAM_FIELD_NAMES = (
    "object_mass",
    "object_static_friction",
    "object_dynamic_friction",
    "object_restitution",
    "tool_mass",
    "tool_static_friction",
    "tool_dynamic_friction",
    "tool_restitution",
    "ground_static_friction",
    "ground_dynamic_friction",
    "ground_restitution",
    "table_static_friction",
    "table_dynamic_friction",
    "table_restitution",
)


def _compute_phys_params_tensor(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    field_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Read the requested physical parameters from PhysX/configuration."""

    device = env.scene[object_cfg.name].data.root_pos_w.device
    object: RigidObject = env.scene[object_cfg.name]
    if field_names is None:
        field_names = getattr(env.cfg, "physics_observation_fields", ())
    field_names = tuple(field_names)
    env._phys_param_field_names = field_names
    reference = object.data.root_pos_w[:, 0]
    values: dict[str, torch.Tensor] = {}

    # PhysX view reads can synchronize/copy large buffers, so only request the
    # groups that are actually part of this policy's observation contract.
    if "object_mass" in field_names:
        object_mass = object.root_physx_view.get_masses().squeeze(-1)
        values["object_mass"] = object_mass.to(device=device)

    object_material_fields = {
        "object_static_friction",
        "object_dynamic_friction",
        "object_restitution",
    }
    if object_material_fields.intersection(field_names):
        object_material_props = object.root_physx_view.get_material_properties()
        values.update(
            {
                "object_static_friction": object_material_props[:, :, 0].mean(dim=1).to(device=device),
                "object_dynamic_friction": object_material_props[:, :, 1].mean(dim=1).to(device=device),
                "object_restitution": object_material_props[:, :, 2].mean(dim=1).to(device=device),
            }
        )

    tool_fields_requested = any(name.startswith("tool_") for name in field_names)
    if tool_fields_requested:
        if getattr(env.cfg, "robot_mode", "tool") != "tool":
            raise ValueError("phys_params requested tool_* fields, but robot_mode is not tool")
        hand: RigidObject = env.scene[hand_cfg.name]
        if not hasattr(env, "_tool_body_idx"):
            tool_body_cfg = SceneEntityCfg(hand_cfg.name, body_names=["link_coacd_convex_piece_0"])
            tool_body_cfg.resolve(env.scene)
            env._tool_body_idx = tool_body_cfg.body_ids[0]

        tool_idx = env._tool_body_idx
        robot_masses = None
        if "tool_mass" in field_names:
            robot_masses = hand.root_physx_view.get_masses()
            values["tool_mass"] = robot_masses[:, tool_idx].to(device=device)

        tool_material_fields = {
            "tool_static_friction",
            "tool_dynamic_friction",
            "tool_restitution",
        }
        if tool_material_fields.intersection(field_names):
            if robot_masses is None:
                robot_masses = hand.root_physx_view.get_masses()
            robot_material_props = hand.root_physx_view.get_material_properties()
            num_shapes = robot_material_props.shape[1]
            num_bodies = robot_masses.shape[1]
            shapes_per_body = num_shapes // num_bodies
            tool_shape_start = tool_idx * shapes_per_body
            tool_shape_end = min((tool_idx + 1) * shapes_per_body, num_shapes)
            tool_material = robot_material_props[:, tool_shape_start:tool_shape_end, :]
            values.update(
                {
                    "tool_static_friction": tool_material[:, :, 0].mean(dim=1).to(device=device),
                    "tool_dynamic_friction": tool_material[:, :, 1].mean(dim=1).to(device=device),
                    "tool_restitution": tool_material[:, :, 2].mean(dim=1).to(device=device),
                }
            )

    ground_fields_requested = any(name.startswith("ground_") for name in field_names)
    if ground_fields_requested:
        ground_static_value = None
        ground_dynamic_value = None
        ground_restitution_value = None
        if bool(getattr(env.cfg, "table_enabled", False)):
            raise ValueError(
                "phys_params requested ground_* fields, but table is enabled and "
                "the ground terrain is not part of the scene."
            )
        try:
            terrain = env.scene["terrain"]
        except Exception as exc:
            raise ValueError(
                "phys_params requested ground_* fields, but scene terrain is unavailable."
            ) from exc
        terrain_prim_path = terrain.cfg.prim_path + "/terrain"
        physics_material_path = f"{terrain_prim_path}/physicsMaterial"

        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdPhysics

        physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)
        if (
            physics_material_prim
            and physics_material_prim.IsValid()
            and physics_material_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            physics_material = UsdPhysics.MaterialAPI(physics_material_prim)
            ground_static_value = physics_material.GetStaticFrictionAttr().Get()
            ground_dynamic_value = physics_material.GetDynamicFrictionAttr().Get()
            ground_restitution_value = physics_material.GetRestitutionAttr().Get()
        values.update(
            {
                "ground_static_friction": torch.full_like(
                    reference, 1.0 if ground_static_value is None else float(ground_static_value)
                ),
                "ground_dynamic_friction": torch.full_like(
                    reference, 1.0 if ground_dynamic_value is None else float(ground_dynamic_value)
                ),
                "ground_restitution": torch.full_like(
                    reference, 0.0 if ground_restitution_value is None else float(ground_restitution_value)
                ),
            }
        )

    table_fields_requested = any(name.startswith("table_") for name in field_names)
    if table_fields_requested:
        table_material = getattr(env.cfg, "table_material", None)
        sampled_table = getattr(env, "_sampled_table_material", None)
        table_static_default = getattr(table_material, "static_friction", 0.8)
        table_dynamic_default = getattr(table_material, "dynamic_friction", 0.8)
        table_restitution_default = getattr(table_material, "restitution", 0.0)
        values.update(
            {
                "table_static_friction": torch.full_like(
                    reference,
                    float(getattr(sampled_table, "static_friction", table_static_default)),
                ),
                "table_dynamic_friction": torch.full_like(
                    reference,
                    float(getattr(sampled_table, "dynamic_friction", table_dynamic_default)),
                ),
                "table_restitution": torch.full_like(
                    reference,
                    float(getattr(sampled_table, "restitution", table_restitution_default)),
                ),
            }
        )

    missing = [name for name in field_names if name not in values]
    if missing:
        raise ValueError(f"Unknown phys_params fields: {missing}")

    if not field_names:
        phys_params_tensor = torch.empty((env.num_envs, 0), device=device)
    else:
        phys_params_tensor = torch.stack(
            [values[name].to(device=device) for name in field_names], dim=1
        )

    return phys_params_tensor


def refresh_phys_params_cache(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | list[int] | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    field_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Refresh all or selected rows of the physical-parameter cache."""

    if field_names is None:
        field_names = getattr(env.cfg, "physics_observation_fields", ())
    field_names = tuple(field_names)
    cache_key = (object_cfg.name, hand_cfg.name, field_names)
    fresh = _compute_phys_params_tensor(
        env,
        object_cfg=object_cfg,
        hand_cfg=hand_cfg,
        field_names=field_names,
    ).detach()

    cache = getattr(env, "_phys_params_cache", None)
    if (
        env_ids is None
        or getattr(env, "_phys_params_cache_key", None) != cache_key
        or cache is None
        or cache.shape != fresh.shape
    ):
        env._phys_params_cache = fresh
        env._phys_params_cache_key = cache_key
        return fresh

    indices = torch.as_tensor(env_ids, device=fresh.device, dtype=torch.long)
    if indices.numel() > 0:
        cache[indices] = fresh[indices]
    return cache


@profile_obs
def phys_params(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    field_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Return physical parameters cached until the corresponding env resets."""

    if field_names is None:
        field_names = getattr(env.cfg, "physics_observation_fields", ())
    field_names = tuple(field_names)
    cache_key = (object_cfg.name, hand_cfg.name, field_names)
    cache = getattr(env, "_phys_params_cache", None)
    if (
        cache is None
        or getattr(env, "_phys_params_cache_key", None) != cache_key
        or cache.shape != (env.num_envs, len(field_names))
    ):
        cache = refresh_phys_params_cache(
            env,
            object_cfg=object_cfg,
            hand_cfg=hand_cfg,
            field_names=field_names,
        )

    return _dbg(env, "phys_params", cache)


@profile_obs
def target_pose_task_embedding(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
) -> torch.Tensor:
    """Two-way task label for the sampled target pose: [stable, secondary]."""

    command_term = env.command_manager.get_term(command_name)
    task_index = getattr(command_term, "target_pose_task_index", None)
    if task_index is None:
        task_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    task_index = task_index.to(device=env.device, dtype=torch.long).clamp_(0, 1)
    return torch.nn.functional.one_hot(task_index, num_classes=2).to(dtype=torch.float32)


def object_pose_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    normalize: bool | None = None,
) -> torch.Tensor:
    """The pose of the object in the environment coordinate frame.

    Returns:
        torch.Tensor: Shape (num_envs, 7) containing [x, y, z, qw, qx, qy, qz] in environment coordinates
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_quat_w = object.data.root_quat_w       # (num_envs, 4)
    object_pos_env = object_pos_w - env.scene.env_origins  # (num_envs, 3)
    pose_7d = torch.cat([object_pos_env, object_quat_w], dim=1)  # (num_envs, 7)

    if env.cfg.visualize_current_object_pose:
        visualize_object_pose_in_env(env, pose_7d)

    # Check normalization setting from environment config unless an internal
    # caller needs the raw pose for pose arithmetic.
    if normalize is None:
        normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for object pose
        device = pose_7d.device
        # For 7D pose: position [x,y,z] + quaternion [qw,qx,qy,qz]
        # Use position normalization from hand_goal params
        pos_mean = _HAND_GOAL_MEAN[:3].to(device).view(1, 3)  # [x, y, z] mean
        pos_std = _HAND_GOAL_STD[:3].to(device).view(1, 3)    # [x, y, z] std
        # For quaternion, use simple normalization (quaternions are already normalized)
        quat_mean = torch.zeros(4, device=device).view(1, 4)  # [qw, qx, qy, qz] mean
        quat_std = torch.ones(4, device=device).view(1, 4)    # [qw, qx, qy, qz] std
        
        # Normalize position and quaternion separately
        pos_norm = (pose_7d[:, :3] - pos_mean) / torch.clamp(pos_std, min=1e-6)
        quat_norm = (pose_7d[:, 3:7] - quat_mean) / torch.clamp(quat_std, min=1e-6)
        
        pose_7d = torch.cat([pos_norm, quat_norm], dim=1)
        
    return pose_7d


def object_pose_9d_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The object's current pose in the environment frame as 9D [x,y,z, r11,r12,r13, r21,r22,r23]."""
    pose_7d = object_pose_in_env_frame(env, object_cfg)
    pos_env = pose_7d[:, :3]
    quat_wxyz = pose_7d[:, 3:7]

    # Convert to rotation matrix
    rot_matrix = matrix_from_quat(quat_wxyz)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    object_pose_9d = torch.cat([pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand/goal-specific normalization parameters for object pose too
        device = object_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        object_pose_9d = (object_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "cur_pose", object_pose_9d)


def visualize_object_pose_in_env(
    env: ManagerBasedRLEnv,
    object_pose_7d: torch.Tensor,
    marker_scale: tuple = (0.08, 0.08, 0.08),
) -> None:
    """Visualize the object's current pose in environment coordinates.
    
    This function creates visualization markers to show the object's current pose
    in the environment coordinate frame, using the same approach as the target pose visualization.
    """
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import quat_from_euler_xyz
    
    # Create visualization markers if they don't exist (similar to target pose visualization)
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"  # Different path from target pose
        marker_cfg.markers["frame"].scale = marker_scale  # Make frames visible but distinct
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)

    # Extract position and quaternion (same as target pose visualization)
    local_positions = object_pose_7d[:, :3]  # (num_envs, 3)
    quaternions = object_pose_7d[:, 3:7]  # (num_envs, 4)

    # Convert local positions to world positions by adding environment origins (same as target)
    world_positions = local_positions + env.scene.env_origins
    
    # Visualize current pose frames using world positions (same method as target)
    env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def create_object_pose_visualizer(env: ManagerBasedRLEnv, marker_scale: tuple = (0.08, 0.08, 0.08)) -> None:
    """Initialize the object pose visualizer. Call this once during environment setup."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = marker_scale
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)


def update_object_pose_visualization(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> None:
    """Update the object pose visualization. Call this during environment step/reset."""
    from isaaclab.utils.math import quat_from_euler_xyz
    
    if hasattr(env, '_current_object_pose_visualizer'):
        # Get object pose in environment frame
        object_pose_7d = object_pose_in_env_frame(env, object_cfg)
        
        # Extract position and euler angles
        local_positions = object_pose_7d[:, :3]
        world_positions = local_positions + env.scene.env_origins
        
        quaternions = object_pose_7d[:, 3:7]
        
        # Update visualization (same method as target pose)
        env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def object_pose_with_visualization(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object pose in environment frame and update visualization as a side effect.
    
    This function serves dual purpose:
    1. Returns the object pose for observations
    2. Triggers visualization update each time it's called
    
    Returns:
        torch.Tensor: Shape (num_envs, 6) containing [x, y, z, roll, pitch, yaw] in environment coordinates
    """
    # Initialize visualizer if not already done
    if not hasattr(env, '_current_object_pose_visualizer'):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)
    
    # Get object pose
    object_pose_7d = object_pose_in_env_frame(env, object_cfg)
    
    # Update visualization
    update_object_pose_visualization(env, object_cfg)
    
    return object_pose_7d


def visualize_object_pointcloud(
    env: ManagerBasedRLEnv,
    pointcloud_tensor: torch.Tensor,
    point_size: float = 0.005,
    color: tuple = (0.0, 1.0, 0.0),  # Green color for point cloud
) -> None:
    """Visualize the object's point cloud for debugging purposes.
    
    The point cloud is displayed in world coordinates, showing the actual 
    transformed points at the object's current position and orientation.
    
    Args:
        env: The RL environment
        pointcloud_tensor: Pre-computed point cloud tensor, shape (num_envs, num_points*3)
        point_size: Size of the visualization spheres
        color: RGB color tuple for the point cloud visualization
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils
    
    # Create visualization markers if they don't exist
    if not hasattr(env, '_pointcloud_visualizer'):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        
        env._pointcloud_visualizer = VisualizationMarkers(marker_cfg)
    
    # Reshape flattened point cloud back to (num_envs, num_points, 3) for visualization
    num_envs = pointcloud_tensor.shape[0]
    points_per_env = pointcloud_tensor.shape[1] // 3
    pointcloud_reshaped = pointcloud_tensor.view(num_envs, points_per_env, 3)
    
    # Flatten all envs into a single set of points for visualization
    all_points = pointcloud_reshaped.reshape(-1, 3)  # Shape: (num_envs * num_points, 3)
    
    # Create identity quaternions for all points (spheres don't need rotation)
    total_points = all_points.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=all_points.device).expand(total_points, -1)
    
    # Visualize the points in world coordinates
    env._pointcloud_visualizer.visualize(
        translations=all_points,
        orientations=orientations
    )


@profile_obs
def get_object_pointcloud(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud in world coordinates.

    Canonical point clouds are initialized once, scaled per environment, and
    transformed for all environments in one batched GPU operation.

    Returns:
        torch.Tensor: shape (num_envs, num_points*3) in world coordinates.
    """
    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_object_pointcloud_points_l"):
        raise RuntimeError(
            "Object point clouds were not preloaded. The prestartup "
            "preload_object_pointclouds event must run before observations are initialized."
        )

    num_envs = object.data.root_pos_w.shape[0]
    all_pointclouds_w = _transform_local_cloud(
        env._object_pointcloud_points_l,
        object.data.root_pos_w[:, :3],
        object.data.root_quat_w,
    )
    all_pointclouds = all_pointclouds_w.reshape(num_envs, -1)

    # Optional visualization for debugging
    if env.cfg.visualize_object_pointcloud:
        visualize_object_pointcloud(env, all_pointclouds.float())

    return all_pointclouds

@profile_obs
def get_object_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud in the environment frame.

    The point cloud remains in env-frame coordinates for the observation.  A
    separate bbox-center term is cached for the policy context.

    Returns:
        Tensor (num_envs, num_points*3) in env-frame coordinates.
    """
    pointcloud_w = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_w.shape
    num_points = flat_dim // 3
    pointcloud_w_reshaped = pointcloud_w.view(num_envs, num_points, 3)
    # Convert to env frame (subtract world env origin)
    pointcloud_env = pointcloud_w_reshaped - env.scene.env_origins.unsqueeze(1)  # (N, P, 3)
    env._obs_object_cloud_E = pointcloud_env.detach()

    obj_bbox_center = _bbox_center_env(pointcloud_env)
    env._obs_obj_bbox_center = obj_bbox_center.detach()

    return pointcloud_env.reshape(num_envs, num_points * 3)


def get_obj_bbox_center(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object cloud AABB center in the environment frame (3D per env).

    Must be listed AFTER ``object_cloud`` in the observation config so the
    bbox-center cache (``env._obs_obj_bbox_center``) is populated first.

    Returns:
        Tensor (num_envs, 3)
    """
    if hasattr(env, "_obs_obj_bbox_center"):
        return env._obs_obj_bbox_center
    # Fallback: recompute (should not happen in normal operation)
    pointcloud_w = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_w.shape
    num_points = flat_dim // 3
    pc_env = pointcloud_w.view(num_envs, num_points, 3) - env.scene.env_origins.unsqueeze(1)
    bbox_center = _bbox_center_env(pc_env)
    env._obs_obj_bbox_center = bbox_center.detach()
    return bbox_center


def visualize_tool_pointcloud(
    env: ManagerBasedRLEnv,
    pointcloud_tensor: torch.Tensor,
    point_size: float = 0.005,
    color: tuple = (0.0, 0.0, 1.0),  # Blue color for tool point cloud
) -> None:
    """Visualize the tool's point cloud for debugging purposes.

    The point cloud is displayed in world coordinates, showing the actual
    transformed points at the tool's current position and orientation.
    Only the first environment is visualized to avoid clutter.

    Args:
        env: The RL environment
        pointcloud_tensor: Pre-computed point cloud tensor, shape (num_envs, num_points*3)
        point_size: Size of the visualization spheres
        color: RGB color tuple for the point cloud visualization
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    # Create visualization markers if they don't exist
    if not hasattr(env, '_tool_pointcloud_visualizer'):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ToolPointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        env._tool_pointcloud_visualizer = VisualizationMarkers(marker_cfg)

    # Reshape flattened point cloud back to (num_envs, num_points, 3) for visualization
    num_envs = pointcloud_tensor.shape[0]
    points_per_env = pointcloud_tensor.shape[1] // 3
    pointcloud_reshaped = pointcloud_tensor.view(num_envs, points_per_env, 3)

    # Flatten all envs into a single set of points for visualization
    all_points = pointcloud_reshaped.reshape(-1, 3)  # Shape: (num_envs * num_points, 3)

    # Create identity quaternions for all points (spheres don't need rotation)
    total_points = all_points.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=all_points.device).expand(total_points, -1)

    # Visualize the points in world coordinates
    env._tool_pointcloud_visualizer.visualize(
        translations=all_points,
        orientations=orientations
    )


def _deterministic_resample_points(points: torch.Tensor, target_count: int) -> torch.Tensor:
    if target_count <= 0:
        return points.new_zeros((0, 3))
    if points.shape[0] == target_count:
        return points
    if points.shape[0] > target_count:
        indices = torch.linspace(
            0,
            points.shape[0] - 1,
            target_count,
            device=points.device,
        ).round().long()
        return points[indices]

    repeats = target_count // points.shape[0] + 1
    return points.repeat((repeats, 1))[:target_count]


def _parse_obj_face_index(token: str, num_vertices: int) -> int:
    raw = token.split("/", 1)[0]
    idx = int(raw)
    if idx < 0:
        return num_vertices + idx
    return idx - 1


def _load_obj_surface_points(
    path: Path,
    *,
    target_count: int,
    device: torch.device,
    label: str = _OFFICIAL_PANDA_GRIPPER_MODE,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} gripper cloud mesh not found: {path}"
        )
    if target_count <= 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if not line.startswith("f "):
                continue
            parts = line.split()[1:]
            if len(parts) < 3:
                continue
            face_indices = [_parse_obj_face_index(token, len(vertices)) for token in parts]
            for i in range(1, len(face_indices) - 1):
                triangles.append((face_indices[0], face_indices[i], face_indices[i + 1]))

    if not vertices:
        raise RuntimeError(f"{label} mesh has no OBJ vertices: {path}")

    verts = torch.tensor(vertices, dtype=torch.float32)
    if not triangles:
        return _deterministic_resample_points(verts, target_count).to(device=device)

    tri_indices = torch.tensor(triangles, dtype=torch.long)
    valid = (tri_indices >= 0).all(dim=1) & (tri_indices < verts.shape[0]).all(dim=1)
    tri_indices = tri_indices[valid]
    if tri_indices.numel() == 0:
        return _deterministic_resample_points(verts, target_count).to(device=device)

    tris = verts[tri_indices]
    edge_a = tris[:, 1] - tris[:, 0]
    edge_b = tris[:, 2] - tris[:, 0]
    areas = 0.5 * torch.linalg.norm(torch.cross(edge_a, edge_b, dim=1), dim=1)
    nonzero = areas > 1e-12
    tris = tris[nonzero]
    areas = areas[nonzero]
    if tris.shape[0] == 0:
        return _deterministic_resample_points(verts, target_count).to(device=device)

    cumulative = torch.cumsum(areas, dim=0)
    total_area = cumulative[-1]
    quantiles = (torch.arange(target_count, dtype=torch.float32) + 0.5) / float(target_count)
    face_ids = torch.searchsorted(cumulative, quantiles * total_area)
    face_ids = torch.clamp(face_ids, 0, tris.shape[0] - 1)
    selected = tris[face_ids]

    sample_ids = torch.arange(target_count, dtype=torch.float32) + 1.0
    r1 = torch.remainder(sample_ids * 0.7548776662466927, 1.0).clamp(1e-6, 1.0 - 1e-6)
    r2 = torch.remainder(sample_ids * 0.5698402909980532, 1.0)
    sqrt_r1 = torch.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    points = (
        selected[:, 0] * w0.unsqueeze(1)
        + selected[:, 1] * w1.unsqueeze(1)
        + selected[:, 2] * w2.unsqueeze(1)
    )
    return points.to(device=device)


def _transform_points_by_spec(
    points: torch.Tensor,
    transform: RigidTransformSpec,
    *,
    device: torch.device,
) -> torch.Tensor:
    if points.shape[0] == 0:
        return points.to(device=device)
    quat = torch.tensor(transform.quat_wxyz, dtype=points.dtype, device=device).view(1, 4)
    pos = torch.tensor(transform.translation, dtype=points.dtype, device=device)
    rot = matrix_from_quat(quat)[0]
    return points.to(device=device) @ rot.T + pos


def _rpy_matrix(
    rpy: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rot_np = R.from_euler("xyz", list(rpy)).as_matrix()
    return torch.tensor(rot_np, dtype=dtype, device=device)


def _apply_generated_joint_pose(
    points_child: torch.Tensor,
    joint: PrismaticJointSpec,
    opening: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if points_child.shape[0] == 0:
        return points_child.to(device=device)
    rot = _rpy_matrix(joint.origin_rpy, device=device, dtype=points_child.dtype)
    origin = torch.tensor(joint.origin_xyz, dtype=points_child.dtype, device=device)
    axis = torch.tensor(joint.axis_xyz, dtype=points_child.dtype, device=device)
    translation = origin + rot @ (axis * float(opening))
    return points_child.to(device=device) @ rot.T + translation


def _official_panda_mesh_points_to_hand_frame(points_mesh: torch.Tensor) -> torch.Tensor:
    """Map eef_panda OBJ mesh coordinates into official Panda hand coordinates."""

    return torch.stack(
        [
            points_mesh[..., 0],
            -points_mesh[..., 2],
            points_mesh[..., 1],
        ],
        dim=-1,
    )


def _get_official_panda_gripper_bucket_clouds(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        OFFICIAL_PANDA_GRIPPER_PROPS_DIR,
    )

    num_points = int(getattr(env.cfg, "num_points", 512))
    mesh_dir = Path(OFFICIAL_PANDA_GRIPPER_PROPS_DIR)
    cache_key = (_OFFICIAL_GRIPPER_CLOUD_SOURCE, str(mesh_dir), str(env.device), num_points)
    cached = _OFFICIAL_GRIPPER_CLOUD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    hand_count = num_points // 2
    left_count = (num_points - hand_count) // 2
    right_count = num_points - hand_count - left_count

    hand = _load_obj_surface_points(
        mesh_dir / "panda_hand.obj",
        target_count=hand_count,
        device=env.device,
    )
    left_base = _load_obj_surface_points(
        mesh_dir / "panda_leftfinger.obj",
        target_count=left_count,
        device=env.device,
    )
    right_base = _load_obj_surface_points(
        mesh_dir / "panda_rightfinger.obj",
        target_count=right_count,
        device=env.device,
    )

    bucket_clouds = []
    for bucket_id in range(_OFFICIAL_GRIPPER_NUM_BUCKETS):
        opening = _OFFICIAL_GRIPPER_OPEN_JOINT_POS * bucket_id / float(_OFFICIAL_GRIPPER_NUM_BUCKETS - 1)
        left_offset = torch.tensor(
            [0.0, _OFFICIAL_GRIPPER_FINGER_MOUNT_OFFSET_Y, -opening],
            dtype=torch.float32,
            device=env.device,
        )
        right_offset = torch.tensor(
            [0.0, _OFFICIAL_GRIPPER_FINGER_MOUNT_OFFSET_Y, opening],
            dtype=torch.float32,
            device=env.device,
        )
        points_mesh = torch.cat([hand, left_base + left_offset, right_base + right_offset], dim=0)
        bucket_clouds.append(_official_panda_mesh_points_to_hand_frame(points_mesh))

    clouds = torch.stack(bucket_clouds, dim=0).contiguous()
    _OFFICIAL_GRIPPER_CLOUD_CACHE[cache_key] = clouds
    return clouds


def _transform_local_cloud(
    points_l: torch.Tensor,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    num_envs = pos_w.shape[0]
    rot_w = matrix_from_quat(quat_w)
    if points_l.dim() == 3:
        return torch.bmm(points_l, rot_w.transpose(1, 2)) + pos_w.unsqueeze(1)
    return torch.bmm(
        rot_w,
        points_l.T.unsqueeze(0).expand(num_envs, -1, -1),
    ).transpose(1, 2) + pos_w.unsqueeze(1)


def _official_panda_gripper_bucket_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    finger_cfg = _resolve_robot_joints(
        env,
        attr_name="_official_panda_finger_joint_cfg",
        joint_names=_OFFICIAL_PANDA_FINGER_JOINT_NAMES,
        expected_count=2,
    )
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids]
    openness = torch.clamp(
        finger_pos.mean(dim=1) / _OFFICIAL_GRIPPER_OPEN_JOINT_POS,
        0.0,
        1.0,
    )
    return torch.clamp(
        torch.round(openness * (_OFFICIAL_GRIPPER_NUM_BUCKETS - 1)).long(),
        0,
        _OFFICIAL_GRIPPER_NUM_BUCKETS - 1,
    )


def get_official_panda_gripper_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    num_envs = env.num_envs
    bucket_ids = _official_panda_gripper_bucket_ids(env)
    env._obs_gripper_bucket_ids = bucket_ids.detach()

    bucket_clouds_l = _get_official_panda_gripper_bucket_clouds(env)
    points_l = bucket_clouds_l[bucket_ids].float()
    palm_pos_w, palm_quat_w = _get_official_panda_palm_pose_w(env)
    pts_world = _transform_local_cloud(points_l, palm_pos_w, palm_quat_w)

    pointcloud_env = pts_world - env.scene.env_origins.unsqueeze(1)
    env._obs_tool_bbox_center = _bbox_center_env(pointcloud_env).detach()
    env._obs_tool_bbox_extent = _bbox_extent_env(pointcloud_env).detach()
    env._obs_gripper_cloud_source = _OFFICIAL_GRIPPER_CLOUD_SOURCE

    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, pts_world.reshape(num_envs, -1).float())

    return pointcloud_env.reshape(num_envs, -1)


def _get_generated_gripper_state_clouds(
    env: ManagerBasedRLEnv,
    gripper: GeneratedGripperAsset,
) -> torch.Tensor:
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_generated_gripper_cloud_cache_dir,
    )
    from utils.geometry.gripper_cloud_cache import (
        cache_path_for_asset,
        load_gripper_cloud_cache,
    )

    cache_key = (gripper.gripper_id, str(env.device))
    cached = _GENERATED_GRIPPER_CLOUD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache = load_gripper_cloud_cache(
        cache_path_for_asset(
            gripper, get_generated_gripper_cloud_cache_dir()
        ),
        expected_gripper_id=gripper.gripper_id,
        expected_source_asset_root=gripper.root_dir,
    )
    states = cache.state_clouds_palm.to(env.device).contiguous()
    _GENERATED_GRIPPER_CLOUD_CACHE[cache_key] = states
    return states


def _get_generated_gripper_state_clouds_by_asset(
    env: ManagerBasedRLEnv,
    metadata: dict[str, object],
) -> torch.Tensor:
    """Stack canonical clouds once so per-step selection is one GPU gather."""

    cached = getattr(env, "_generated_gripper_state_clouds_by_asset_cache", None)
    if cached is not None:
        return cached

    assets = metadata["assets"]
    state_clouds = [
        _get_generated_gripper_state_clouds(env, gripper)
        for gripper in assets
    ]
    expected_shape = tuple(state_clouds[0].shape)
    for gripper, states in zip(assets, state_clouds):
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                "Generated-gripper cloud caches must have one uniform shape for "
                f"vectorized lookup; {gripper.gripper_id!r} has {tuple(states.shape)}, "
                f"expected {expected_shape}"
            )

    stacked = torch.stack(state_clouds, dim=0).contiguous()
    # Replace the individual allocations with views into the stacked tensor.
    # This keeps legacy callers working without retaining a second full copy.
    device_key = str(env.device)
    for asset_index, gripper in enumerate(assets):
        _GENERATED_GRIPPER_CLOUD_CACHE[(gripper.gripper_id, device_key)] = stacked[asset_index]
    env._generated_gripper_state_clouds_by_asset_cache = stacked
    return stacked


def _generated_gripper_cache_bin_ids(
    env: ManagerBasedRLEnv,
    gripper: GeneratedGripperAsset,
    env_indices: torch.Tensor,
) -> torch.Tensor:
    robot = env.scene["robot"]
    finger_cfg = _resolve_generated_robot_joints(
        env,
        joint_names=gripper.finger_joint_names,
        expected_count=2,
    )
    finger_pos = robot.data.joint_pos[env_indices][:, finger_cfg.joint_ids]
    openness = torch.clamp(
        finger_pos.mean(dim=1) / gripper.open_joint_pos,
        0.0,
        1.0,
    )
    return torch.clamp(
        torch.round(openness * 127).long(),
        0,
        127,
    )


def get_generated_gripper_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    num_envs = env.num_envs
    num_points = int(getattr(env.cfg, "num_points", 512))
    robot = env.scene["robot"]
    metadata = _generated_gripper_runtime_metadata(env)
    env_indices = metadata["env_indices"]

    finger_pos = robot.data.joint_pos[
        env_indices.unsqueeze(1),
        metadata["finger_joint_ids"],
    ]
    openness = torch.clamp(
        finger_pos.mean(dim=1) / metadata["open_joint_positions"],
        0.0,
        1.0,
    )
    state_clouds_by_asset = _get_generated_gripper_state_clouds_by_asset(env, metadata)
    num_buckets = int(state_clouds_by_asset.shape[1])
    all_bucket_ids = torch.clamp(
        torch.round(openness * (num_buckets - 1)).long(),
        0,
        num_buckets - 1,
    )
    points_l = state_clouds_by_asset[
        metadata["asset_indices"],
        all_bucket_ids,
    ].float()
    if tuple(points_l.shape[1:]) != (num_points, 3):
        raise RuntimeError(
            "Generated-gripper point cloud has invalid vectorized shape "
            f"{tuple(points_l.shape)}; expected ({num_envs}, {num_points}, 3)"
        )

    palm_state_w = robot.data.body_state_w[
        env_indices,
        metadata["palm_body_ids"],
    ]
    out_world = _transform_local_cloud(
        points_l,
        palm_state_w[:, :3],
        palm_state_w[:, 3:7],
    )

    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)
    env._obs_tool_cloud_E = pointcloud_env.detach()
    env._obs_gripper_bucket_ids = all_bucket_ids.detach()
    env._obs_tool_bbox_center = _bbox_center_env(pointcloud_env).detach()
    env._obs_tool_bbox_extent = _bbox_extent_env(pointcloud_env).detach()
    env._obs_gripper_cloud_source = _GENERATED_GRIPPER_CLOUD_SOURCE

    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(num_envs, -1).float())

    return pointcloud_env.reshape(num_envs, -1)


def get_generated_gripper_kinematic_state_clouds(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return strict closed, half-open, and open clouds for either generated family."""

    if _is_one_dof_gripper(env):
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
            get_one_dof_gripper_data_for_env,
        )
        from utils.geometry.gripper_cloud_cache import (
            cache_path_for_asset,
            load_gripper_cloud_cache,
        )

        num_points = int(getattr(env.cfg, "num_points", 512))
        out = torch.empty(
            (env.num_envs, 3, num_points, 3),
            device=env.device,
            dtype=torch.float32,
        )
        for _, env_ids_list in _one_dof_gripper_env_groups(env).items():
            gripper = get_one_dof_gripper_data_for_env(env_ids_list[0])
            key = (gripper.gripper_id, str(env.device), num_points)
            states = _ONE_DOF_GRIPPER_STATE_CLOUD_CACHE.get(key)
            if states is None:
                cache = load_gripper_cloud_cache(
                    cache_path_for_asset(gripper),
                    expected_gripper_id=gripper.gripper_id,
                    expected_source_manifest=gripper.manifest_path,
                    expected_source_asset_root=gripper.root_dir,
                )
                states = torch.stack(
                    [
                        cache.cloud_at_fraction(fraction)
                        for fraction in (0.0, 0.5, 1.0)
                    ]
                ).to(env.device)
                if states.shape != (3, num_points, 3):
                    raise RuntimeError(
                        "One-DoF kinematic cloud has invalid shape "
                        f"{tuple(states.shape)} for {gripper.gripper_id}"
                    )
                _ONE_DOF_GRIPPER_STATE_CLOUD_CACHE[key] = states
            indices = torch.tensor(
                env_ids_list, device=env.device, dtype=torch.long
            )
            out[indices] = states.unsqueeze(0)
        return out.reshape(env.num_envs, -1)

    if not _is_generated_gripper(env):
        raise RuntimeError(
            "Kinematic gripper clouds require generated_gripper or one_dof_gripper "
            f"runtime mode, got {getattr(env.cfg, 'robot_mode', None)!r}"
        )

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_generated_gripper_data_for_env,
    )

    num_points = int(getattr(env.cfg, "num_points", 512))
    out = torch.empty(
        (env.num_envs, 3, num_points, 3),
        device=env.device,
        dtype=torch.float32,
    )
    for _, env_ids_list in _generated_gripper_env_groups(env).items():
        gripper = get_generated_gripper_data_for_env(env_ids_list[0])
        env_indices = torch.tensor(
            env_ids_list, device=env.device, dtype=torch.long
        )
        buckets = _get_generated_gripper_state_clouds(env, gripper).float()
        states = torch.stack(
            (buckets[0], buckets[64], buckets[-1]),
            dim=0,
        )
        out[env_indices] = states.unsqueeze(0)
    return out.reshape(env.num_envs, -1)


def get_one_dof_gripper_pointcloud_in_env_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select the nearest canonical cache bin from the measured joint state."""

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_one_dof_gripper_data_for_env,
    )
    from utils.geometry.gripper_cloud_cache import (
        cache_path_for_asset,
        load_gripper_cloud_cache,
    )

    robot = env.scene["robot"]
    num_points = int(getattr(env.cfg, "num_points", 512))
    out_world = torch.empty((env.num_envs, num_points, 3), dtype=torch.float32, device=env.device)
    all_bin_ids = torch.empty(
        (env.num_envs,), dtype=torch.long, device=env.device
    )
    for _, env_ids in _one_dof_gripper_env_groups(env).items():
        gripper = get_one_dof_gripper_data_for_env(env_ids[0])
        indices = torch.tensor(env_ids, dtype=torch.long, device=env.device)
        cache_key = (gripper.gripper_id, str(env.device))
        states = _ONE_DOF_CANONICAL_CACHE.get(cache_key)
        if states is None:
            cache = load_gripper_cloud_cache(
                cache_path_for_asset(gripper),
                expected_gripper_id=gripper.gripper_id,
                expected_source_manifest=gripper.manifest_path,
                expected_source_asset_root=gripper.root_dir,
            )
            states = cache.state_clouds_palm.to(env.device).contiguous()
            _ONE_DOF_CANONICAL_CACHE[cache_key] = states
        if states.shape != (128, num_points, 3):
            raise RuntimeError(
                "Canonical gripper cache does not match the RL observation: "
                f"cache={tuple(states.shape)} "
                f"expected=(128, {num_points}, 3) asset={gripper.gripper_id}"
            )
        joint_cfg = _resolve_one_dof_robot_joints(
            env, joint_names=gripper.actuated_joint_names
        )
        open_pos = torch.tensor(
            gripper.open_joint_positions,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )
        closed_pos = torch.tensor(
            gripper.closed_joint_positions,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )
        joint_pos = robot.data.joint_pos[indices][:, joint_cfg.joint_ids]
        opening = torch.clamp(
            ((joint_pos - closed_pos) / (open_pos - closed_pos)).mean(dim=1),
            0.0,
            1.0,
        )
        bin_ids = torch.round(opening * 127).long()
        all_bin_ids[indices] = bin_ids
        points_palm = states[bin_ids]
        palm_cfg = _resolve_one_dof_robot_bodies(
            env, body_names=(gripper.palm_body_name,)
        )
        palm_state = robot.data.body_state_w[
            indices, palm_cfg.body_ids[0], :
        ]
        out_world[indices] = _transform_local_cloud(
            points_palm, palm_state[:, :3], palm_state[:, 3:7]
        )

    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)
    env._obs_tool_cloud_E = pointcloud_env.detach()
    env._obs_gripper_bucket_ids = all_bin_ids.detach()
    env._obs_tool_bbox_center = _bbox_center_env(pointcloud_env).detach()
    env._obs_tool_bbox_extent = _bbox_extent_env(pointcloud_env).detach()
    env._obs_gripper_cloud_source = _ONE_DOF_GRIPPER_CLOUD_SOURCE
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(env.num_envs, -1))
    return pointcloud_env.reshape(env.num_envs, -1)


def _load_oracle_prepared_mesh(path: Path, *, device: torch.device):
    key = ("mesh", str(path.resolve()), str(device))
    cached = _ORACLE_MESH_SDF_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("oracle_patch exact mesh SDF requires trimesh") from exc
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise RuntimeError(f"oracle_patch mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geometries)
    vertices = torch.as_tensor(loaded.vertices, dtype=torch.float32, device=device).contiguous()
    faces = torch.as_tensor(loaded.faces, dtype=torch.long, device=device).contiguous()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"oracle_patch requires a triangular mesh: {path}")
    # Warp builds and caches its own BVH; no dense face-vertex expansion is needed.
    prepared = (vertices, faces, None)
    _ORACLE_MESH_SDF_CACHE[key] = prepared
    return prepared


def _prepare_oracle_component(
    path: Path,
    *,
    device: torch.device,
    body_transform: RigidTransformSpec,
    joint: PrismaticJointSpec | None = None,
    opening: float = 0.0,
    post_body_transform: RigidTransformSpec | None = None,
):
    raw_v, faces, _ = _load_oracle_prepared_mesh(path, device=device)
    vertices = _transform_points_by_spec(raw_v, body_transform, device=device)
    if post_body_transform is not None:
        vertices = _transform_points_by_spec(vertices, post_body_transform, device=device)
    if joint is not None:
        vertices = _apply_generated_joint_pose(vertices, joint, opening, device=device)
    return vertices, faces, None


def _get_oracle_generated_gripper_link_meshes(
    env: "ManagerBasedRLEnv",
    gripper: GeneratedGripperAsset,
):
    key = (gripper.gripper_id, str(env.device))
    cached = _ORACLE_GRIPPER_LINK_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    meshes = {
        "plank": _prepare_oracle_component(
            gripper.plank_mesh,
            device=env.device,
            body_transform=gripper.mesh_to_body_frame["plank"],
        ),
        "finger": _prepare_oracle_component(
            gripper.finger_mesh,
            device=env.device,
            body_transform=gripper.mesh_to_body_frame["finger"],
        ),
    }
    if gripper.has_tip:
        if gripper.finger_tip_mesh is None or gripper.finger_tip_to_finger_frame is None:
            raise RuntimeError(f"generated gripper {gripper.gripper_id!r} has invalid tip metadata")
        meshes["tip_body"] = _prepare_oracle_component(
            gripper.finger_tip_mesh,
            device=env.device,
            body_transform=gripper.mesh_to_body_frame["finger_tip"],
        )
        meshes["tip_in_finger"] = _prepare_oracle_component(
            gripper.finger_tip_mesh,
            device=env.device,
            body_transform=gripper.mesh_to_body_frame["finger_tip"],
            post_body_transform=gripper.finger_tip_to_finger_frame,
        )
    _ORACLE_GRIPPER_LINK_MESH_CACHE[key] = meshes
    return meshes


def _query_oracle_prepared_mesh(
    points: torch.Tensor,
    prepared,
    *,
    signed: bool,
) -> torch.Tensor:
    from utils.geometry.warp_sdf import (
        signed_distance_points_to_prepared_mesh_warp,
        unsigned_distance_points_to_prepared_mesh_warp,
    )

    vertices, faces, _ = prepared
    query = (
        signed_distance_points_to_prepared_mesh_warp
        if signed
        else unsigned_distance_points_to_prepared_mesh_warp
    )
    return query(
        points,
        mesh_v=vertices,
        mesh_f=faces,
    )


def _get_oracle_mesh_distance(
    env: "ManagerBasedRLEnv",
    *,
    signed: bool,
) -> torch.Tensor:
    """Exact privileged mesh distance for every object/tool cloud point.

    Output order is ``object-points-to-gripper`` followed by
    ``tool-points-to-object``. Negative is inside the opposite mesh. There is
    deliberately no point-cloud-distance or unsigned fallback.
    """
    if not _is_generated_gripper(env):
        raise RuntimeError("oracle_patch exact mesh SDF currently requires generated_gripper mode")
    if not hasattr(env, "_obs_object_cloud_E") or not hasattr(env, "_obs_tool_cloud_E"):
        raise RuntimeError("oracle mesh SDF observation must run after object_cloud and tool_cloud")

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_generated_gripper_data_for_env,
        get_object_asset_cfg_for_env,
        get_object_index_for_env,
    )

    object_points_E = env._obs_object_cloud_E
    tool_points_E = env._obs_tool_cloud_E
    batch_size, num_points, _ = object_points_E.shape
    if tool_points_E.shape != (batch_size, num_points, 3):
        raise RuntimeError("oracle mesh SDF requires matching object/tool point counts")
    object_points_W = object_points_E + env.scene.env_origins.unsqueeze(1)
    tool_points_W = tool_points_E + env.scene.env_origins.unsqueeze(1)

    # Tool points queried against each environment's exact object mesh.
    object_asset = env.scene["object"]
    object_pos_W = object_asset.data.root_pos_w[:, :3]
    object_rot_W = matrix_from_quat(object_asset.data.root_quat_w)
    object_scales = env._object_pointcloud_scales
    if not bool(torch.allclose(object_scales, object_scales[:, :1].expand_as(object_scales), atol=1e-6, rtol=1e-6)):
        raise RuntimeError("oracle mesh SDF currently requires uniform object scale")
    object_groups: dict[int, list[int]] = {}
    for env_id in range(batch_size):
        object_groups.setdefault(get_object_index_for_env(env_id), []).append(env_id)
    tool_sdf = torch.empty((batch_size, num_points), device=env.device, dtype=torch.float32)
    for env_ids in object_groups.values():
        indices = torch.tensor(env_ids, device=env.device, dtype=torch.long)
        scale = object_scales[indices, 0]
        query_scaled_local = torch.matmul(
            tool_points_W[indices] - object_pos_W[indices].unsqueeze(1),
            object_rot_W[indices],
        )
        query_raw_local = query_scaled_local / scale[:, None, None]
        mesh_path = Path(get_object_asset_cfg_for_env(env_ids[0]).obj_path)
        prepared = _load_oracle_prepared_mesh(mesh_path, device=env.device)
        distance_raw = _query_oracle_prepared_mesh(
            query_raw_local.reshape(-1, 3), prepared, signed=signed
        )
        tool_sdf[indices] = distance_raw.reshape(len(env_ids), num_points) * scale.unsqueeze(1)

    # Object points queried against the exact union of live articulated link meshes.
    # Every query uses current body poses; no openness bucket approximation is used.
    object_sdf = torch.empty_like(tool_sdf)
    robot = env.scene["robot"]
    for _, env_ids in _generated_gripper_env_groups(env).items():
        gripper = get_generated_gripper_data_for_env(env_ids[0])
        indices = torch.tensor(env_ids, device=env.device, dtype=torch.long)
        meshes = _get_oracle_generated_gripper_link_meshes(env, gripper)
        component_queries = []

        palm_pos_W, palm_quat_W = _get_generated_gripper_palm_pose_w(env, gripper, indices)
        component_queries.append((meshes["plank"], palm_pos_W, palm_quat_W))

        finger_cfg = _resolve_generated_robot_bodies(
            env,
            body_names=gripper.finger_body_names,
            expected_count=2,
        )
        finger_state_W = robot.data.body_state_w[indices][:, finger_cfg.body_ids, :]
        for finger_index in range(2):
            component_queries.append(
                (
                    meshes["finger"],
                    finger_state_W[:, finger_index, :3],
                    finger_state_W[:, finger_index, 3:7],
                )
            )

        if gripper.has_tip:
            if gripper.fingertip_body_names is not None:
                tip_cfg = _resolve_generated_robot_bodies(
                    env,
                    body_names=gripper.fingertip_body_names,
                    expected_count=2,
                )
                tip_state_W = robot.data.body_state_w[indices][:, tip_cfg.body_ids, :]
                for tip_index in range(2):
                    component_queries.append(
                        (
                            meshes["tip_body"],
                            tip_state_W[:, tip_index, :3],
                            tip_state_W[:, tip_index, 3:7],
                        )
                    )
            else:
                for finger_index in range(2):
                    component_queries.append(
                        (
                            meshes["tip_in_finger"],
                            finger_state_W[:, finger_index, :3],
                            finger_state_W[:, finger_index, 3:7],
                        )
                    )

        component_sdf = []
        for prepared, body_pos_W, body_quat_W in component_queries:
            body_rot_W = matrix_from_quat(body_quat_W)
            query_body = torch.matmul(
                object_points_W[indices] - body_pos_W.unsqueeze(1),
                body_rot_W,
            )
            component_sdf.append(
                _query_oracle_prepared_mesh(
                    query_body.reshape(-1, 3), prepared, signed=signed
                ).reshape(
                    len(env_ids), num_points
                )
            )
        object_sdf[indices] = torch.stack(component_sdf, dim=0).min(dim=0).values

    if not bool(torch.isfinite(object_sdf).all()) or not bool(torch.isfinite(tool_sdf).all()):
        raise RuntimeError("oracle exact mesh SDF produced non-finite values")
    return torch.cat((object_sdf, tool_sdf), dim=1)


@profile_obs
def get_oracle_mesh_signed_sdf(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Exact signed point-to-opposite-mesh distance; negative means inside."""

    return _get_oracle_mesh_distance(env, signed=True)


@profile_obs
def get_oracle_mesh_unsigned_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Exact unsigned closest-triangle distance to the opposite mesh."""

    return _get_oracle_mesh_distance(env, signed=False)


@profile_obs
def get_tool_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get tool point cloud in the environment frame.

    Each env may have a different tool (via MultiUsdFileCfg).  The canonical
    mesh is transformed using Cloud.get_pointcloud() with the tool body's
    world-frame pose and a uniform scale of TOOL_SCALE, then env_origins
    are subtracted to get env-frame coordinates.

    This mirrors the pattern used by get_object_pointcloud_in_env_frame.

    Returns:
        torch.Tensor: shape (num_envs, num_points*3), float32
    """
    if _is_official_panda_gripper(env):
        return get_official_panda_gripper_pointcloud_in_env_frame(env)
    if _is_generated_gripper(env):
        return get_generated_gripper_pointcloud_in_env_frame(env)
    if _is_one_dof_gripper(env):
        return get_one_dof_gripper_pointcloud_in_env_frame(env)

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        from isaaclab.managers import SceneEntityCfg
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)

    out_tensor = None

    # Group envs by tool type for efficient batch processing
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for _tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])
        tool_cloud = get_cached_cloud(td["obj_path"])
        base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

        # Scale canonical points
        pts = base_pts * TOOL_SCALE  # (M, 3)

        # Compute the body-frame offset from base_center.
        # The USD body origin was placed at the base_center position (by adjust_meshes.py).
        # We must shift the point cloud by the same amount so it aligns with the body frame.
        pts = pts.clone()
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
            bbox_min_raw = base_pts.min(dim=0).values
            bbox_max_raw = base_pts.max(dim=0).values
            body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
            pts = pts - body_origin
        else:
            # Fallback: legacy Z-shift if base_center is not available
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Get per-env rotation matrices and positions
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        # Rotate points by body orientation and translate to body position
        # pts: (M, 3) -> (1, 3, M);  batch_rot: (B, 3, 3)
        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, M, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        # Allocate output on first iteration
        if out_tensor is None:
            num_points = pts_world.shape[1]
            out_tensor = torch.empty(
                (num_envs, num_points, 3),
                device=device,
                dtype=pts_world.dtype,
            )

        out_tensor[env_indices] = pts_world

    # Convert to env frame: subtract env_origins
    pointcloud_w = out_tensor  # (num_envs, M, 3) in world frame
    pointcloud_env = pointcloud_w - env.scene.env_origins.unsqueeze(1)  # (N, M, 3)

    tool_bbox_center = _bbox_center_env(pointcloud_env)
    env._obs_tool_bbox_center = tool_bbox_center.detach()

    # Flatten to (num_envs, M*3)
    pointcloud_env_flat = pointcloud_env.reshape(num_envs, -1)

    # Optional visualization (world frame for marker rendering)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_tensor.reshape(num_envs, -1).float())

    return pointcloud_env_flat


def get_tool_bbox_center(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return the tool cloud AABB center in the environment frame.

    Must be listed AFTER ``tool_cloud`` in the observation config so the
    bbox-center cache (``env._obs_tool_bbox_center``) is populated first.

    Returns:
        Tensor (num_envs, 3)
    """
    if hasattr(env, "_obs_tool_bbox_center"):
        return env._obs_tool_bbox_center
    # Fallback: recompute by calling the full function
    get_tool_pointcloud_in_env_frame(env)
    return env._obs_tool_bbox_center


# ---------------------------------------------------------------------------
# 7D point cloud observations: position + mass + velocity
# ---------------------------------------------------------------------------

def _check_object_robot_contact(
    env: ManagerBasedRLEnv,
    pointcloud_w: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if point cloud is in contact with robot (any body/link).

    Returns:
        torch.Tensor: (num_envs,) bool mask, True if any point is within threshold of any robot body.
    """
    robot = env.scene[robot_cfg.name]
    body_positions = robot.data.body_state_w[:, :, :3]

    pointcloud_expanded = pointcloud_w.unsqueeze(2)
    body_positions_expanded = body_positions.unsqueeze(1)

    distances = torch.norm(pointcloud_expanded - body_positions_expanded, dim=3)
    min_distances_per_point = distances.min(dim=2)[0]
    in_contact = (min_distances_per_point < contact_threshold).any(dim=1)

    return in_contact


def visualize_pointcloud_velocity_mass(
    env: ManagerBasedRLEnv,
    features_flat: torch.Tensor,
    pointcloud_w_for_viz: torch.Tensor,
    env_idx: int = 0,
    subsample: int = 16,
    velocity_scale: float = 0.1,
    max_velocity: float = 1.0,
    point_size: float = 0.003,
    prefix: str = "object",
) -> None:
    """Visualize 7D point cloud features (position + mass + velocity).

    Creates colored sphere markers for mass and arrow markers for velocity vectors.

    Args:
        env: The RL environment
        features_flat: (num_envs, num_points * 7) flat features tensor
        pointcloud_w_for_viz: (num_envs, num_points, 3) world-frame positions for visualization
        env_idx: Which environment to visualize
        subsample: Show every Nth point to reduce clutter
        velocity_scale: Scale factor for velocity arrow length
        max_velocity: Maximum velocity for color normalization
        point_size: Radius of the point spheres
        prefix: Marker namespace prefix (e.g., "object", "tool")
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    num_envs = features_flat.shape[0]
    num_points = features_flat.shape[1] // 7
    feat = features_flat.view(num_envs, num_points, 7).float()

    # Extract all envs, subsampled
    sub_points = pointcloud_w_for_viz[:, ::subsample, :].reshape(-1, 3).float()  # (num_envs*K, 3)
    num_viz = sub_points.shape[0]

    # 1. Point cloud visualization (small green spheres)
    marker_name = f"_{prefix}_pc_visualizer"
    if not hasattr(env, marker_name):
        marker_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/{prefix.title()}PC7D",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
            },
        )
        setattr(env, marker_name, VisualizationMarkers(marker_cfg))

    orientations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=sub_points.device
    ).expand(num_viz, -1)
    getattr(env, marker_name).visualize(sub_points, orientations)


@profile_obs
def get_object_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud with mass and velocity (7D per point).

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.
    Velocity is gated by a contact check: only non-zero when the object is
    in contact with the robot.

    Returns:
        torch.Tensor: shape (num_envs, 512*7) = (num_envs, 3584), dtype float16
    """
    object: RigidObject = env.scene[object_cfg.name]

    pointcloud_flat = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_flat.shape
    num_points = flat_dim // 3

    pointcloud_w = pointcloud_flat.view(num_envs, num_points, 3)
    pointcloud_env = pointcloud_w - env.scene.env_origins.unsqueeze(1)

    # Contact-gated velocity: only compute when object touches robot
    in_contact = _check_object_robot_contact(
        env, pointcloud_w, robot_cfg=SceneEntityCfg("robot")
    )
    point_velocities = torch.zeros_like(pointcloud_env)

    if in_contact.any():
        contact_mask = in_contact.unsqueeze(1).unsqueeze(2)

        root_pos = object.data.root_pos_w[:, :3].unsqueeze(1)
        root_lin_vel = object.data.root_lin_vel_w[:, :3].unsqueeze(1)
        root_ang_vel = object.data.root_ang_vel_w[:, :3].unsqueeze(1)

        rel_positions = pointcloud_w - root_pos
        point_velocities_computed = root_lin_vel + torch.cross(
            root_ang_vel.expand_as(rel_positions), rel_positions, dim=-1
        )

        point_velocities = point_velocities_computed * contact_mask

    # Mass feature: distribute total object mass evenly across points
    masses = object.root_physx_view.get_masses().view(num_envs, -1)
    object_mass = masses.sum(dim=1)
    mass_per_point = (
        (object_mass / float(num_points))
        .view(num_envs, 1, 1)
        .to(pointcloud_env.device, pointcloud_env.dtype)
    )
    mass_feature = mass_per_point.expand(-1, num_points, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_velocities], dim=-1
    )  # (num_envs, num_points, 7)

    # Optional visualization for debugging
    if (
        hasattr(env.cfg, "visualize_object_velocity_mass")
        and env.cfg.visualize_object_velocity_mass
    ):
        pointcloud_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, num_points * 7),
            pointcloud_w_for_viz,
            env_idx=0,
            subsample=16,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.003,
            prefix="object",
        )

    return features.reshape(num_envs, num_points * 7).to(torch.float16)


@profile_obs
def get_tool_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get tool point cloud with mass and velocity (7D per point).

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.
    Velocity is derived from the tool body's rigid body kinematics.
    Uses the same adjusted decomposed mesh contract as get_tool_pointcloud_in_env_frame.

    Returns:
        torch.Tensor: shape (num_envs, 512*7) = (num_envs, 3584), dtype float16
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)
    tool_lin_vel_w = robot.data.body_lin_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)
    tool_ang_vel_w = robot.data.body_ang_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)

    out_world = None  # (num_envs, M, 3) in world frame

    # Group envs by tool type for efficient batch processing
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for _tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])
        tool_cloud = get_cached_cloud(td["obj_path"])
        base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

        # Scale canonical points
        pts = base_pts * TOOL_SCALE  # (M, 3)

        # Preserve existing body-frame alignment behavior.
        pts = pts.clone()
        base_center_norm = td.get("base_center")
        if base_center_norm is not None:
            bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
            bbox_min_raw = base_pts.min(dim=0).values
            bbox_max_raw = base_pts.max(dim=0).values
            body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
            pts = pts - body_origin
        else:
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Get per-env rotation matrices and positions
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        # Rotate points by body orientation and translate to body position
        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, M, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        # Allocate output on first iteration
        if out_world is None:
            num_points = pts_world.shape[1]
            out_world = torch.empty(
                (num_envs, num_points, 3),
                device=device,
                dtype=pts_world.dtype,
            )

        out_world[env_indices] = pts_world

    # Convert to env frame
    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)  # (B, M, 3)

    # Velocity: v_point = v_lin + w x (p_world - p_link)
    rel_pos = out_world - tool_pos_w.unsqueeze(1)  # (B, M, 3)
    ang_vel_expanded = tool_ang_vel_w.unsqueeze(1).expand_as(rel_pos)
    point_vels = tool_lin_vel_w.unsqueeze(1) + torch.cross(
        ang_vel_expanded, rel_pos, dim=-1
    )  # (B, M, 3)

    # Mass feature: get tool mass from robot articulation PhysX view
    if not hasattr(env, "_tool_body_idx"):
        _tb_cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _tb_cfg.resolve(env.scene)
        env._tool_body_idx = _tb_cfg.body_ids[0]
    robot_masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies) on CPU
    tool_mass = robot_masses[:, env._tool_body_idx].to(device=device, dtype=pointcloud_env.dtype)  # (num_envs,)
    total_pts = pointcloud_env.shape[1]
    mass_per_point = (tool_mass / float(total_pts)).view(num_envs, 1, 1)
    mass_feature = mass_per_point.expand(-1, total_pts, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_vels], dim=-1
    )  # (B, M, 7)

    # Optional visualization: full tool cloud (blue spheres, position only)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(num_envs, -1).float())

    # Optional visualization: 7D features (velocity + mass)
    if getattr(env.cfg, "visualize_tool_velocity_mass", False):
        pts_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, -1),
            pts_w_for_viz,
            env_idx=0,
            subsample=4,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.003,
            prefix="tool",
        )

    # Flatten and return
    return features.reshape(num_envs, -1).to(torch.float16)


def get_tool_head_area_pointcloud_with_mass_velocity(
    env: ManagerBasedRLEnv,
    num_points: int = 512,
) -> torch.Tensor:
    """Get tool point cloud filtered to head area region, with mass and velocity (7D).

    Only includes points within the tool's head_area bounding box (the functional
    contact region of the tool). Points outside the head area are excluded.
    The remaining points are subsampled or padded to ``num_points``.

    Each point is [x, y, z, mass_per_point, vx, vy, vz] in environment frame.

    Args:
        env: The environment instance.
        num_points: Target number of points in the output cloud (default: 512).

    Returns:
        torch.Tensor: shape (num_envs, num_points*7), dtype float16
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud,
        get_tool_data_for_env,
        get_tool_index_for_env,
        TOOL_SCALE,
    )

    num_envs = env.num_envs
    device = env.device

    # Get tool body world pose from the robot articulation
    robot = env.scene["robot"]
    if not hasattr(env, "_tool_body_cfg_resolved"):
        _cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _cfg.resolve(env.scene)
        env._tool_body_cfg_resolved = _cfg
    tool_body_idx = env._tool_body_cfg_resolved.body_ids[0]
    tool_pos_w = robot.data.body_state_w[:, tool_body_idx, :3]    # (num_envs, 3)
    tool_quat_w = robot.data.body_state_w[:, tool_body_idx, 3:7]  # (num_envs, 4)
    tool_lin_vel_w = robot.data.body_lin_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)
    tool_ang_vel_w = robot.data.body_ang_vel_w[:, tool_body_idx, :3]  # (num_envs, 3)

    # Precompute per-tool head area masks on canonical points (cached)
    if not hasattr(env, "_tool_head_area_cache"):
        env._tool_head_area_cache = {}

    out_world = torch.empty((num_envs, num_points, 3), device=device, dtype=torch.float32)

    # Group envs by tool type
    tool_to_envs: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        tidx = get_tool_index_for_env(env_id)
        if tidx not in tool_to_envs:
            tool_to_envs[tidx] = []
        tool_to_envs[tidx].append(env_id)

    for tidx, env_ids_list in tool_to_envs.items():
        td = get_tool_data_for_env(env_ids_list[0])

        # Get or compute the head-area filtered canonical points for this tool
        if tidx not in env._tool_head_area_cache:
            tool_cloud = get_cached_cloud(td["obj_path"])
            base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)

            # Scale canonical points
            pts_canonical = base_pts * TOOL_SCALE  # (M, 3)
            pts_canonical = pts_canonical.clone()
            base_center_norm = td.get("base_center")
            if base_center_norm is not None:
                bc = torch.tensor(base_center_norm, device=device, dtype=torch.float32)
                bbox_min_raw = base_pts.min(dim=0).values
                bbox_max_raw = base_pts.max(dim=0).values
                body_origin = (bbox_min_raw + bc * (bbox_max_raw - bbox_min_raw)) * TOOL_SCALE
                pts_canonical = pts_canonical - body_origin
            else:
                pts_canonical[:, 2] = pts_canonical[:, 2] - pts_canonical[:, 2].min()

            head_area_norm = td.get("head_area")
            if head_area_norm is not None:
                # head_area_norm is [[x_min, y_min, z_min], [x_max, y_max, z_max]] in [0,1] normalized coords
                bbox_min = pts_canonical.min(dim=0).values
                bbox_max = pts_canonical.max(dim=0).values
                bbox_range = bbox_max - bbox_min

                # Convert normalized head area to absolute coordinates
                ha_min = bbox_min + torch.tensor(head_area_norm[0], device=device, dtype=torch.float32) * bbox_range
                ha_max = bbox_min + torch.tensor(head_area_norm[1], device=device, dtype=torch.float32) * bbox_range

                # Filter: keep points inside head area bounding box
                mask = (
                    (pts_canonical[:, 0] >= ha_min[0]) & (pts_canonical[:, 0] <= ha_max[0]) &
                    (pts_canonical[:, 1] >= ha_min[1]) & (pts_canonical[:, 1] <= ha_max[1]) &
                    (pts_canonical[:, 2] >= ha_min[2]) & (pts_canonical[:, 2] <= ha_max[2])
                )
                head_pts = pts_canonical[mask]

                if head_pts.shape[0] == 0:
                    # Fallback: if no points in head area, use all points
                    print(f"[WARNING] No points in head area for tool '{td['name']}', using full cloud")
                    head_pts = pts_canonical
            else:
                # No head_area defined, use all points
                print(f"[WARNING] No head_area for tool '{td['name']}', using full cloud as obstacle")
                head_pts = pts_canonical

            # Subsample or pad to target num_points
            n = head_pts.shape[0]
            if n >= num_points:
                # Uniform subsample
                idx = torch.linspace(0, n - 1, num_points, device=device).round().long()
                head_pts = head_pts[idx]
            else:
                # Repeat-pad
                repeats = (num_points + n - 1) // n
                head_pts = head_pts.repeat(repeats, 1)[:num_points]

            env._tool_head_area_cache[tidx] = head_pts  # (num_points, 3) canonical

        pts = env._tool_head_area_cache[tidx]  # (num_points, 3) canonical, already filtered

        env_indices = torch.tensor(env_ids_list, device=device, dtype=torch.long)

        # Transform to world frame
        batch_quat = tool_quat_w[env_indices].contiguous()  # (B, 4)
        batch_pos = tool_pos_w[env_indices].contiguous()     # (B, 3)
        batch_rot = matrix_from_quat(batch_quat)             # (B, 3, 3)

        pts_rotated = torch.bmm(
            batch_rot,
            pts.T.unsqueeze(0).expand(len(env_ids_list), -1, -1),
        ).transpose(1, 2)  # (B, num_points, 3)
        pts_world = pts_rotated + batch_pos.unsqueeze(1)

        out_world[env_indices] = pts_world

    # Convert to env frame
    pointcloud_env = out_world - env.scene.env_origins.unsqueeze(1)  # (B, num_points, 3)

    # Velocity: v_point = v_lin + w x (p_world - p_link)
    rel_pos = out_world - tool_pos_w.unsqueeze(1)  # (B, num_points, 3)
    ang_vel_expanded = tool_ang_vel_w.unsqueeze(1).expand_as(rel_pos)
    point_vels = tool_lin_vel_w.unsqueeze(1) + torch.cross(
        ang_vel_expanded, rel_pos, dim=-1
    )  # (B, num_points, 3)

    # Mass feature
    if not hasattr(env, "_tool_body_idx"):
        _tb_cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        _tb_cfg.resolve(env.scene)
        env._tool_body_idx = _tb_cfg.body_ids[0]
    robot_masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies) on CPU
    tool_mass = robot_masses[:, env._tool_body_idx].to(device=device, dtype=pointcloud_env.dtype)  # (num_envs,)
    mass_per_point = (tool_mass / float(num_points)).view(num_envs, 1, 1)
    mass_feature = mass_per_point.expand(-1, num_points, 1)

    features = torch.cat(
        [pointcloud_env, mass_feature, point_vels], dim=-1
    )  # (B, num_points, 7)

    # Optional visualization: head area cloud (blue spheres, position only)
    if getattr(env.cfg, "visualize_tool_pointcloud", False):
        visualize_tool_pointcloud(env, out_world.reshape(num_envs, -1).float())

    # Optional visualization: head area 7D features (orange spheres)
    if getattr(env.cfg, "visualize_tool_head_area", False):
        pts_w_for_viz = pointcloud_env + env.scene.env_origins.unsqueeze(1)
        visualize_pointcloud_velocity_mass(
            env,
            features.reshape(num_envs, -1),
            pts_w_for_viz,
            env_idx=0,
            subsample=4,
            velocity_scale=0.1,
            max_velocity=1.0,
            point_size=0.004,
            prefix="tool_head_area",
        )

    return features.reshape(num_envs, -1).to(torch.float16)

"""Strict forward kinematics for manifest-backed one-DoF grippers."""

from __future__ import annotations

from functools import lru_cache
import math
import xml.etree.ElementTree as ET

import torch

from utils.assets.one_dof_gripper_assets import OneDofGripperAsset


def _values(raw: str | None, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in (raw or "").split())
    if not values:
        return (0.0,) * count
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {values}")
    return values


def _rpy_matrix(rpy: tuple[float, float, float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=torch.float64,
    )


def _origin(node: ET.Element | None) -> torch.Tensor:
    xyz = _values(None if node is None else node.get("xyz"), 3)
    rpy = _values(None if node is None else node.get("rpy"), 3)
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return result


def _axis_angle(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    vector = torch.tensor(axis, dtype=torch.float64)
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= 0.0:
        raise ValueError("URDF revolute-joint axis cannot be zero")
    x, y, z = vector / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = torch.tensor(
        (
            (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
            (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
            (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
        ),
        dtype=torch.float64,
    )
    return result


@lru_cache(maxsize=256)
def _joint_tree(
    urdf_path: str,
) -> tuple[
    dict[
        str,
        tuple[
            tuple[
                str,
                str,
                str,
                torch.Tensor,
                tuple[float, float, float],
                str | None,
                float,
                float,
            ],
            ...,
        ],
    ],
    str,
]:
    root = ET.parse(urdf_path).getroot()
    links = {str(link.get("name")) for link in root.findall("link")}
    child_links: set[str] = set()
    children: dict[str, list[tuple]] = {}
    for joint in root.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            raise ValueError(f"URDF joint is missing parent/child: {urdf_path}")
        parent = str(parent_node.get("link"))
        child = str(child_node.get("link"))
        axis_node = joint.find("axis")
        axis = (
            (1.0, 0.0, 0.0)
            if axis_node is None
            else _values(axis_node.get("xyz"), 3)
        )
        mimic = joint.find("mimic")
        mimic_joint = None if mimic is None else str(mimic.get("joint"))
        mimic_multiplier = (
            1.0 if mimic is None else float(mimic.get("multiplier", "1"))
        )
        mimic_offset = 0.0 if mimic is None else float(mimic.get("offset", "0"))
        children.setdefault(parent, []).append(
            (
                str(joint.get("name")),
                str(joint.get("type")),
                child,
                _origin(joint.find("origin")),
                axis,
                mimic_joint,
                mimic_multiplier,
                mimic_offset,
            )
        )
        child_links.add(child)
    root_links = links.difference(child_links)
    if len(root_links) != 1:
        raise ValueError(
            f"URDF must contain exactly one root link, got {sorted(root_links)}: "
            f"{urdf_path}"
        )
    return (
        {parent: tuple(joints) for parent, joints in children.items()},
        next(iter(root_links)),
    )


def one_dof_body_poses(
    asset: OneDofGripperAsset,
    opening_fraction: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return every URDF body pose in the configured palm frame."""

    fraction = float(opening_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Opening fraction must be in [0, 1], got {fraction}")
    joint_positions = {
        name: closed + fraction * (opened - closed)
        for name, opened, closed in zip(
            asset.actuated_joint_names,
            asset.open_joint_positions,
            asset.closed_joint_positions,
        )
    }
    tree, root_link = _joint_tree(str(asset.urdf_path))
    joint_specs = {
        name: (mimic_joint, mimic_multiplier, mimic_offset)
        for joints in tree.values()
        for (
            name,
            _,
            _,
            _,
            _,
            mimic_joint,
            mimic_multiplier,
            mimic_offset,
        ) in joints
    }

    def joint_position(name: str, pending: frozenset[str] = frozenset()) -> float:
        if name in joint_positions:
            return joint_positions[name]
        if name in pending:
            raise ValueError(f"URDF mimic cycle contains joint {name!r}")
        mimic_joint, multiplier, offset = joint_specs[name]
        if mimic_joint is None:
            return 0.0
        if mimic_joint not in joint_specs:
            raise ValueError(
                f"URDF mimic joint {name!r} references unknown joint "
                f"{mimic_joint!r}"
            )
        return multiplier * joint_position(
            mimic_joint, pending | {name}
        ) + offset

    poses = {root_link: torch.eye(4, dtype=torch.float64)}
    pending = [root_link]
    while pending:
        parent = pending.pop()
        for (
            name,
            joint_type,
            child,
            origin,
            axis,
            _,
            _,
            _,
        ) in tree.get(parent, ()):
            pose = poses[parent] @ origin
            position = joint_position(name)
            if joint_type in {"revolute", "continuous"}:
                pose = pose @ _axis_angle(axis, position)
            elif joint_type == "prismatic":
                translation = torch.eye(4, dtype=torch.float64)
                translation[:3, 3] = torch.tensor(axis, dtype=torch.float64) * position
                pose = pose @ translation
            elif joint_type != "fixed":
                raise ValueError(
                    f"Unsupported joint type {joint_type!r} in {asset.urdf_path}"
                )
            poses[child] = pose
            pending.append(child)
    if asset.palm_body_name not in poses:
        raise ValueError(
            f"Configured palm {asset.palm_body_name!r} is absent from "
            f"{asset.urdf_path}"
        )
    missing = sorted(
        {part.body_name for part in asset.cloud_parts}.difference(poses)
    )
    if missing:
        raise ValueError(
            f"Cloud bodies are absent from the URDF tree: {missing}"
        )
    palm_inverse = torch.linalg.inv(poses[asset.palm_body_name])
    return {
        name: (palm_inverse @ pose).to(device=device, dtype=dtype)
        for name, pose in poses.items()
    }


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous transform to row-vector points."""

    return points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]

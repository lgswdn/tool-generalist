#!/usr/bin/env python3
"""Generate topology-stable GraspGenX-style gripper families for Franka RL.

The physical mechanisms have four or nine revolute joints, but every asset
exposes one normalized policy closure command through a joint-synergy manifest.
USD conversion remains a separate Isaac-enabled step:

    python scripts/convert_one_dof_gripper.py --manifest <manifest>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARM_TEMPLATE = ROOT / "gripper/franka_template/isaac.urdf"
DEFAULT_OUTPUT_ROOT = ROOT / "gripper/generated_graspgenx_matched_128"
IDENTITY_TRANSFORM = {
    "translation": [0.0, 0.0, 0.0],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
}
FRANKA_REAL_CLOSURE_TIME_S = 0.8
PARALLEL_MATCHED_OPEN_GAP_M = 0.08
REVOLUTE_OPEN_ANGLE_RANGE_RAD = (-0.45, -0.25)
REVOLUTE_TRAVEL_ANGLE_RAD = 0.65
THREE_FINGER_CLOSE_ANGLE_RANGE_RAD = (0.14, 0.32)
PANDA_GENERAL_FINGER_LENGTH_RANGE_M = (0.051, 0.235)
PANDA_GENERAL_FINGER_THICKNESS_RANGE_M = (0.006, 0.058)
PANDA_GENERAL_TIP_WIDTH_RANGE_M = (0.006, 0.035)
PANDA_GENERAL_PALM_DEPTH_RANGE_M = (0.015, 0.144)
PANDA_GENERAL_PALM_HEIGHT_RANGE_M = (0.015, 0.162)
PANDA_GENERAL_MAX_FINGER_ASPECT_RATIO = 26.5
PANDA_GENERAL_MAX_PALM_WIDTH_M = 0.272


def _box_inward_extent(
    angle: float,
    *,
    center_y: float,
    center_z: float,
    size_y: float,
    size_z: float,
) -> float:
    """Exact inward y projection of a left-finger box about its link pivot."""

    return max(
        z * math.sin(angle) - y * math.cos(angle)
        for y in (center_y - size_y * 0.5, center_y + size_y * 0.5)
        for z in (center_z - size_z * 0.5, center_z + size_z * 0.5)
    )


def _revolute_inward_extent(
    *,
    length_scale: float,
    mid_length: float,
    top_length: float,
    tip_length: float,
    mid_y: float,
    top_y: float,
    tip_width: float,
    tip_shape: str,
    add_outer: bool,
    outer_y: float,
    outer_length_ratio: float,
    mid_angle: float,
    top_angle: float,
) -> float:
    """Exact projected inward extent of every generated finger visual."""

    scaled_mid = mid_length * length_scale
    scaled_top = top_length * length_scale
    scaled_tip = tip_length * length_scale
    extents = [
        _box_inward_extent(
            mid_angle,
            center_y=0.0,
            center_z=scaled_mid * 0.5,
            size_y=mid_y,
            size_z=scaled_mid,
        )
    ]
    if add_outer:
        outer_length = scaled_mid * outer_length_ratio
        extents.append(
            _box_inward_extent(
                mid_angle,
                center_y=mid_y * 0.5 + outer_y * 0.5,
                center_z=scaled_mid * 0.5,
                size_y=outer_y,
                size_z=outer_length,
            )
        )

    top_base = scaled_mid * math.sin(mid_angle)
    extents.append(
        top_base
        + _box_inward_extent(
            top_angle,
            center_y=0.0,
            center_z=scaled_top * 0.5,
            size_y=top_y,
            size_z=scaled_top,
        )
    )
    if tip_shape == "square":
        extents.append(
            top_base
            + _box_inward_extent(
                top_angle,
                center_y=0.0,
                center_z=scaled_top + scaled_tip * 0.5,
                size_y=top_y,
                size_z=scaled_tip,
            )
        )
    elif tip_shape == "round":
        axial_inward = max(
            scaled_top * math.sin(top_angle),
            (scaled_top + scaled_tip) * math.sin(top_angle),
        )
        extents.append(
            top_base
            + axial_inward
            + tip_width * 0.5 * abs(math.cos(top_angle))
        )
    elif tip_shape != "none":
        raise ValueError(f"Unsupported revolute tip shape: {tip_shape!r}")
    return max(extents)


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def _xyz(values: Iterable[float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _arm_robot(name: str) -> ET.Element:
    """Copy the arm/base contract exactly from the generated parallel template."""

    source = ET.parse(ARM_TEMPLATE).getroot()
    robot = ET.Element("robot", {"name": name})
    for element in source:
        element_name = element.get("name", "")
        if element.tag == "link" and re.fullmatch(r"panda_link[0-8]", element_name):
            copied = copy.deepcopy(element)
            for mesh in copied.findall(".//mesh"):
                filename = Path(mesh.get("filename", ""))
                if not filename.is_absolute():
                    mesh.set("filename", str((ARM_TEMPLATE.parent / filename).resolve()))
            robot.append(copied)
        elif element.tag == "joint" and re.fullmatch(r"panda_joint[1-8]", element_name):
            robot.append(copy.deepcopy(element))
    return robot


def _origin(parent: ET.Element, xyz: Iterable[float], rpy: Iterable[float] = (0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _xyz(xyz), "rpy": _xyz(rpy)})


def _box_inertial(link: ET.Element, size: tuple[float, float, float], density: float = 550.0) -> None:
    x, y, z = size
    mass = max(x * y * z * density, 1.0e-4)
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", {"value": _fmt(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _fmt(mass * (y * y + z * z) / 12.0),
            "ixy": "0",
            "ixz": "0",
            "iyy": _fmt(mass * (x * x + z * z) / 12.0),
            "iyz": "0",
            "izz": _fmt(mass * (x * x + y * y) / 12.0),
        },
    )


def _empty_link(robot: ET.Element, name: str, inertial_size: tuple[float, float, float]) -> ET.Element:
    link = ET.SubElement(robot, "link", {"name": name})
    _box_inertial(link, inertial_size)
    return link


def _add_box_geometry(
    link: ET.Element,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    *,
    color: tuple[float, float, float, float],
) -> None:
    for kind in ("visual", "collision"):
        node = ET.SubElement(link, kind)
        _origin(node, center)
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "box", {"size": _xyz(size)})
        if kind == "visual":
            material = ET.SubElement(node, "material", {"name": "generated_gripper"})
            ET.SubElement(material, "color", {"rgba": _xyz(color)})


def _add_cylinder_geometry(
    link: ET.Element,
    radius: float,
    length: float,
    center: tuple[float, float, float],
    *,
    color: tuple[float, float, float, float],
) -> None:
    for kind in ("visual", "collision"):
        node = ET.SubElement(link, kind)
        _origin(node, center)
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(
            geometry,
            "cylinder",
            {"radius": _fmt(radius), "length": _fmt(length)},
        )
        if kind == "visual":
            material = ET.SubElement(node, "material", {"name": "generated_gripper_tip"})
            ET.SubElement(material, "color", {"rgba": _xyz(color)})


def _fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    *,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    _origin(joint, xyz)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def _revolute_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    *,
    xyz: tuple[float, float, float],
    axis: tuple[float, float, float],
    upper: float,
    velocity: float,
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": _xyz(axis)})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": "0",
            "upper": _fmt(upper),
            "effort": "40",
            "velocity": _fmt(velocity),
        },
    )


def _box_cloud_part(
    body_name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "body_name": body_name,
        "geometry_type": "box",
        "box_size": list(size),
        "geometry_to_body": {
            "translation": list(center),
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }


def _cylinder_cloud_part(
    body_name: str,
    radius: float,
    length: float,
    center: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "body_name": body_name,
        "geometry_type": "cylinder",
        "cylinder_radius": radius,
        "cylinder_length": length,
        "geometry_to_body": {
            "translation": list(center),
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }


def _add_grasp_target(robot: ET.Element, palm_height: float, reach: float) -> None:
    target = _empty_link(robot, "panda_grasptarget", (0.005, 0.005, 0.005))
    _fixed_joint(
        robot,
        "panda_grasptarget_joint",
        "gripper_palm",
        "panda_grasptarget",
        xyz=(0.0, 0.0, palm_height + reach),
    )


def _base_manifest_entry(
    *,
    gripper_id: str,
    category: str,
    topology_family: str,
    root_dir: str,
    joint_names: list[str],
    open_positions: list[float],
    closed_positions: list[float],
    velocity_limit: float,
    cloud_parts: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": gripper_id,
        "category": category,
        "topology_family": topology_family,
        "root_dir": root_dir,
        "urdf_path": "robot.urdf",
        "usd_path": "robot.usd",
        "palm_body_name": "gripper_palm",
        "ee_body_name": "panda_grasptarget",
        "grasp_frame": {
            "body_name": "panda_grasptarget",
            "offset": IDENTITY_TRANSFORM,
        },
        "control": {
            "command_dim": 1,
            "adapter": "joint_synergy",
            "actuated_joint_names": joint_names,
            "measured_joint_name": joint_names[0],
            "open_joint_positions": open_positions,
            "closed_joint_positions": closed_positions,
            "actuator": {
                "effort_limit": 40.0,
                "stiffness": 600.0,
                "damping": 50.0,
                "armature": 0.005,
                "velocity_limit": velocity_limit,
            },
        },
        "cloud_parts": cloud_parts,
        "params": {
            "generated": True,
            "generator": "graspgenx_procedural_v1",
            "policy_control_dof": 1,
            "nominal_closure_time_s": FRANKA_REAL_CLOSURE_TIME_S,
            **params,
        },
    }


def build_two_finger_revolute(
    rng: random.Random,
    *,
    gripper_id: str,
    root_dir: str,
) -> tuple[ET.Element, dict[str, Any], dict[str, Any]]:
    for _ in range(512):
        palm_x = rng.uniform(*PANDA_GENERAL_PALM_DEPTH_RANGE_M)
        palm_z = rng.uniform(*PANDA_GENERAL_PALM_HEIGHT_RANGE_M)
        sampled_finger_length = rng.uniform(*PANDA_GENERAL_FINGER_LENGTH_RANGE_M)
        mid_ratio = rng.uniform(0.30, 0.70)
        mid_length = sampled_finger_length * mid_ratio
        top_length = sampled_finger_length - mid_length
        mid_x = rng.uniform(*PANDA_GENERAL_FINGER_THICKNESS_RANGE_M)
        mid_y = rng.uniform(*PANDA_GENERAL_FINGER_THICKNESS_RANGE_M)
        top_x = rng.uniform(*PANDA_GENERAL_FINGER_THICKNESS_RANGE_M)
        top_y = rng.uniform(*PANDA_GENERAL_TIP_WIDTH_RANGE_M)
        minimum_section = sampled_finger_length / PANDA_GENERAL_MAX_FINGER_ASPECT_RATIO
        mid_x = max(mid_x, minimum_section)
        mid_y = max(mid_y, minimum_section)
        top_x = max(top_x, minimum_section)
        top_y = max(top_y, minimum_section)
        closure_mode = rng.choice(("parallel_tip", "pinch"))
        add_outer = rng.random() < 0.5
        tip_shape = rng.choice(("none", "square", "round"))
        tip_length = rng.uniform(0.006, 0.050)
        tip_width = rng.uniform(*PANDA_GENERAL_TIP_WIDTH_RANGE_M)
        outer_x = rng.uniform(0.004, 0.030)
        outer_y = rng.uniform(0.003, 0.025)
        outer_length_ratio = rng.uniform(0.35, 1.00)
        open_angle = rng.uniform(*REVOLUTE_OPEN_ANGLE_RANGE_RAD)
        closed_angle = open_angle + REVOLUTE_TRAVEL_ANGLE_RAD
        top_closed_angle = (
            0.0
            if closure_mode == "parallel_tip"
            else open_angle + 2.0 * REVOLUTE_TRAVEL_ANGLE_RAD
        )
        top_open_angle = 0.0 if closure_mode == "parallel_tip" else open_angle

        def inward_extent(length_scale: float, *, closed: bool) -> float:
            mid_angle = closed_angle if closed else open_angle
            top_angle = top_closed_angle if closed else top_open_angle
            return _revolute_inward_extent(
                length_scale=length_scale,
                mid_length=mid_length,
                top_length=top_length,
                tip_length=tip_length,
                mid_y=mid_y,
                top_y=top_y,
                tip_width=tip_width,
                tip_shape=tip_shape,
                add_outer=add_outer,
                outer_y=outer_y,
                outer_length_ratio=outer_length_ratio,
                mid_angle=mid_angle,
                top_angle=top_angle,
            )

        def open_gap_for_scale(length_scale: float) -> float:
            return 2.0 * (
                inward_extent(length_scale, closed=True)
                - inward_extent(length_scale, closed=False)
            )

        low, high = 1.0e-4, 1.0
        while open_gap_for_scale(high) < PARALLEL_MATCHED_OPEN_GAP_M and high < 64.0:
            high *= 2.0
        if (
            open_gap_for_scale(low) > PARALLEL_MATCHED_OPEN_GAP_M
            or open_gap_for_scale(high) < PARALLEL_MATCHED_OPEN_GAP_M
        ):
            continue
        for _ in range(64):
            middle = 0.5 * (low + high)
            if open_gap_for_scale(middle) < PARALLEL_MATCHED_OPEN_GAP_M:
                low = middle
            else:
                high = middle
        length_scale = 0.5 * (low + high)
        finger_length = sampled_finger_length * length_scale
        if not (
            PANDA_GENERAL_FINGER_LENGTH_RANGE_M[0]
            <= finger_length
            <= PANDA_GENERAL_FINGER_LENGTH_RANGE_M[1]
        ):
            continue
        minimum_final_section = finger_length / PANDA_GENERAL_MAX_FINGER_ASPECT_RATIO
        if min(mid_x, mid_y, top_x, top_y) < minimum_final_section:
            continue
        if tip_shape != "none" and tip_width < finger_length / 24.5:
            continue

        palm_side_margin = rng.uniform(0.006, 0.060)
        max_inward_extent = inward_extent(length_scale, closed=True)
        allowed_extent = (
            PANDA_GENERAL_MAX_PALM_WIDTH_M - mid_y - palm_side_margin
        ) * 0.5
        if max_inward_extent > allowed_extent:
            continue

        mid_length *= length_scale
        top_length *= length_scale
        tip_length *= length_scale
        break
    else:
        raise RuntimeError(
            "Could not sample a two-finger revolute geometry with an exact "
            f"{PARALLEL_MATCHED_OPEN_GAP_M:.3f} m open gap, zero closed gap, "
            "and the generated Panda geometry limits"
        )

    finger_envelope_y = max(
        mid_y,
        top_y,
        tip_width if tip_shape != "none" else 0.0,
    )
    closed_surface_gap = 0.0
    separation = 2.0 * max_inward_extent
    palm_y = separation + mid_y + palm_side_margin
    outer_size = (
        outer_x,
        outer_y,
        mid_length * outer_length_ratio,
    )

    params = {
        "family": "two_finger_revolute",
        "closure_mode": closure_mode,
        "top_to_mid_motion_ratio": -1.0 if closure_mode == "parallel_tip" else 1.0,
        "has_outer_finger": add_outer,
        "outer_size": list(outer_size),
        "tip_shape": tip_shape,
        "finger_length": finger_length,
        "mid_to_total_length_ratio": mid_ratio,
        "longitudinal_scale_for_aperture": length_scale,
        "palm_size": [palm_x, palm_y, palm_z],
        "finger_separation": separation,
        "open_surface_gap": PARALLEL_MATCHED_OPEN_GAP_M,
        "closed_surface_gap": closed_surface_gap,
        "finger_envelope_width": finger_envelope_y,
        "mid_size": [mid_x, mid_y, mid_length],
        "top_size": [top_x, top_y, top_length],
        "tip_length": tip_length,
        "tip_width": tip_width,
        "open_angle_rad": open_angle,
        "closed_angle_rad": closed_angle,
        "closed_top_global_angle_rad": top_closed_angle,
        "travel_angle_rad": REVOLUTE_TRAVEL_ANGLE_RAD,
    }
    robot = _arm_robot(gripper_id)
    palm = _empty_link(robot, "gripper_palm", (palm_x, palm_y, palm_z))
    _add_box_geometry(
        palm,
        (palm_x, palm_y, palm_z),
        (0.0, 0.0, palm_z * 0.5),
        color=(0.18, 0.22, 0.28, 1.0),
    )
    _fixed_joint(robot, "panda_hand_joint", "panda_link8", "gripper_palm")
    cloud_parts = [
        _box_cloud_part("gripper_palm", (palm_x, palm_y, palm_z), (0.0, 0.0, palm_z * 0.5))
    ]
    joint_names: list[str] = []
    for side, y_sign in (("left", 1.0), ("right", -1.0)):
        mid_name = f"{side}_mid_link"
        top_name = f"{side}_top_link"
        mid = _empty_link(robot, mid_name, (mid_x, mid_y, mid_length))
        top = _empty_link(robot, top_name, (top_x, top_y, top_length))
        _add_box_geometry(
            mid,
            (mid_x, mid_y, mid_length),
            (0.0, 0.0, mid_length * 0.5),
            color=(0.30, 0.34, 0.40, 1.0),
        )
        cloud_parts.append(
            _box_cloud_part(mid_name, (mid_x, mid_y, mid_length), (0.0, 0.0, mid_length * 0.5))
        )
        _add_box_geometry(
            top,
            (top_x, top_y, top_length),
            (0.0, 0.0, top_length * 0.5),
            color=(0.38, 0.42, 0.48, 1.0),
        )
        cloud_parts.append(
            _box_cloud_part(
                top_name,
                (top_x, top_y, top_length),
                (0.0, 0.0, top_length * 0.5),
            )
        )

        mid_axis = (y_sign, 0.0, 0.0)
        top_axis = (
            (-y_sign, 0.0, 0.0)
            if closure_mode == "parallel_tip"
            else mid_axis
        )
        top_zero_angle = -open_angle if closure_mode == "parallel_tip" else 0.0
        mid_joint = f"{side}_mid_joint"
        top_joint = f"{side}_top_joint"
        _revolute_joint(
            robot,
            mid_joint,
            "gripper_palm",
            mid_name,
            xyz=(0.0, y_sign * separation * 0.5, palm_z),
            axis=mid_axis,
            upper=1.10,
            velocity=REVOLUTE_TRAVEL_ANGLE_RAD / FRANKA_REAL_CLOSURE_TIME_S,
            rpy=(y_sign * open_angle, 0.0, 0.0),
        )
        _revolute_joint(
            robot,
            top_joint,
            mid_name,
            top_name,
            xyz=(0.0, 0.0, mid_length),
            axis=top_axis,
            upper=1.10,
            velocity=REVOLUTE_TRAVEL_ANGLE_RAD / FRANKA_REAL_CLOSURE_TIME_S,
            rpy=(y_sign * top_zero_angle, 0.0, 0.0),
        )
        joint_names.extend((mid_joint, top_joint))

        if add_outer:
            outer_center = (
                0.0,
                y_sign * (mid_y * 0.5 + outer_size[1] * 0.5),
                mid_length * 0.5,
            )
            _add_box_geometry(
                mid,
                outer_size,
                outer_center,
                color=(0.24, 0.28, 0.34, 1.0),
            )
            cloud_parts.append(_box_cloud_part(mid_name, outer_size, outer_center))

        tip_center = (0.0, 0.0, top_length + tip_length * 0.5)
        if tip_shape == "square":
            tip_size = (tip_width, top_y, tip_length)
            _add_box_geometry(top, tip_size, tip_center, color=(0.55, 0.58, 0.62, 1.0))
            cloud_parts.append(_box_cloud_part(top_name, tip_size, tip_center))
        elif tip_shape == "round":
            radius = tip_width * 0.5
            _add_cylinder_geometry(
                top,
                radius,
                tip_length,
                tip_center,
                color=(0.55, 0.58, 0.62, 1.0),
            )
            cloud_parts.append(
                _cylinder_cloud_part(top_name, radius, tip_length, tip_center)
            )

    _add_grasp_target(robot, palm_z, 0.65 * (mid_length + top_length))
    manifest = _base_manifest_entry(
        gripper_id=gripper_id,
        category="two_finger_revolute",
        topology_family="generated_two_finger_revolute_v1",
        root_dir=root_dir,
        joint_names=joint_names,
        open_positions=[0.0] * 4,
        closed_positions=[REVOLUTE_TRAVEL_ANGLE_RAD] * 4,
        velocity_limit=REVOLUTE_TRAVEL_ANGLE_RAD / FRANKA_REAL_CLOSURE_TIME_S,
        cloud_parts=cloud_parts,
        params=params,
    )
    return robot, manifest, params


def build_three_finger_high_dof(
    rng: random.Random,
    *,
    gripper_id: str,
    root_dir: str,
) -> tuple[ET.Element, dict[str, Any], dict[str, Any]]:
    palm_z = rng.uniform(*PANDA_GENERAL_PALM_HEIGHT_RANGE_M)
    wrist_ratio = rng.uniform(0.35, 0.70)
    side_rotation = rng.uniform(-0.30, 0.30)
    closed_angle = rng.uniform(*THREE_FINGER_CLOSE_ANGLE_RANGE_RAD)
    azimuths = (
        math.pi / 3.0 + side_rotation,
        2.0 * math.pi / 3.0 - side_rotation,
        -math.pi / 2.0,
    )

    # Match the aggressive generated Panda population at the whole-finger
    # scale, while retaining three independently randomized cubic links.
    finger_lengths = [
        rng.uniform(*PANDA_GENERAL_FINGER_LENGTH_RANGE_M) for _ in range(3)
    ]
    link_lengths: list[list[float]] = []
    link_widths: list[list[float]] = []
    link_depths: list[list[float]] = []
    for finger_length in finger_lengths:
        raw_ratios = [rng.uniform(0.20, 1.0) for _ in range(3)]
        ratio_sum = sum(raw_ratios)
        lengths = [finger_length * ratio / ratio_sum for ratio in raw_ratios]
        link_lengths.append(lengths)
        link_widths.append(
            [
                max(
                    rng.uniform(*PANDA_GENERAL_FINGER_THICKNESS_RANGE_M),
                    length / PANDA_GENERAL_MAX_FINGER_ASPECT_RATIO,
                )
                for length in lengths
            ]
        )
        link_depths.append(
            [
                max(
                    rng.uniform(*PANDA_GENERAL_FINGER_THICKNESS_RANGE_M),
                    length / PANDA_GENERAL_MAX_FINGER_ASPECT_RATIO,
                )
                for length in lengths
            ]
        )

    def inward_travels(lengths_by_finger: list[list[float]]) -> list[float]:
        return [
            sum(
                length * math.sin((index + 1) * closed_angle)
                for index, length in enumerate(lengths)
            )
            for lengths in lengths_by_finger
        ]

    inward_travel = inward_travels(link_lengths)
    # The fingertip centerlines must terminate at the common palm axis when
    # fully closed. The old 6--14 mm addition here was a designed air gap.
    base_radii = list(inward_travel)
    palm_side_margin = rng.uniform(0.006, 0.060)
    palm_half_extent = max(
        radius + max(max(widths), max(depths)) * 0.5
        for radius, widths, depths in zip(base_radii, link_widths, link_depths)
    )
    allowed_half_extent = 0.5 * (
        PANDA_GENERAL_MAX_PALM_WIDTH_M - palm_side_margin
    )
    longitudinal_scales = [1.0, 1.0, 1.0]
    for finger_index, (lengths, widths, depths) in enumerate(
        zip(link_lengths, link_widths, link_depths)
    ):
        cross_section_radius = 0.5 * max(max(widths), max(depths))
        if inward_travel[finger_index] + cross_section_radius <= allowed_half_extent:
            continue
        minimum_scale = min(
            1.0,
            PANDA_GENERAL_FINGER_LENGTH_RANGE_M[0] / sum(lengths),
        )
        low, high = minimum_scale, 1.0
        for _ in range(48):
            middle = 0.5 * (low + high)
            scaled_travel = sum(
                length
                * middle
                * math.sin((index + 1) * closed_angle)
                for index, length in enumerate(lengths)
            )
            if scaled_travel + cross_section_radius <= allowed_half_extent:
                low = middle
            else:
                high = middle
        longitudinal_scales[finger_index] = low
        link_lengths[finger_index] = [length * low for length in lengths]

    finger_lengths = [sum(lengths) for lengths in link_lengths]
    inward_travel = inward_travels(link_lengths)
    base_radii = list(inward_travel)

    palm_half_extent = max(
        radius + max(max(widths), max(depths)) * 0.5
        for radius, widths, depths in zip(base_radii, link_widths, link_depths)
    )
    palm_x = min(
        PANDA_GENERAL_MAX_PALM_WIDTH_M,
        2.0 * palm_half_extent + palm_side_margin,
    )
    palm_y = min(
        PANDA_GENERAL_MAX_PALM_WIDTH_M,
        2.0 * palm_half_extent + rng.uniform(0.006, 0.060),
    )

    params = {
        "family": "three_finger_high_dof",
        "physical_dof": 9,
        "graspgenx_joint_count_per_finger": 3,
        "palm_size": [palm_x, palm_y, palm_z],
        "wrist_to_palm_ratio": wrist_ratio,
        "base_radii": base_radii,
        "closed_tip_radial_clearance": [0.0, 0.0, 0.0],
        "closed_centerline_gap": 0.0,
        "top_finger_side_rotation_rad": side_rotation,
        "finger_azimuths_rad": list(azimuths),
        "finger_lengths": finger_lengths,
        "longitudinal_scales_for_palm_limit": longitudinal_scales,
        "link_lengths": link_lengths,
        "link_widths": link_widths,
        "link_depths": link_depths,
        "closed_angle_rad": closed_angle,
    }
    robot = _arm_robot(gripper_id)
    palm = _empty_link(robot, "gripper_palm", (palm_x, palm_y, palm_z))
    _add_box_geometry(
        palm,
        (palm_x, palm_y, palm_z),
        (0.0, 0.0, palm_z * 0.5),
        color=(0.16, 0.20, 0.27, 1.0),
    )
    wrist_size = (palm_x * wrist_ratio, palm_y * wrist_ratio, palm_z * 0.35)
    wrist_center = (0.0, 0.0, -wrist_size[2] * 0.5)
    _add_box_geometry(
        palm,
        wrist_size,
        wrist_center,
        color=(0.12, 0.16, 0.22, 1.0),
    )
    _fixed_joint(robot, "panda_hand_joint", "panda_link8", "gripper_palm")
    cloud_parts = [
        _box_cloud_part("gripper_palm", (palm_x, palm_y, palm_z), (0.0, 0.0, palm_z * 0.5)),
        _box_cloud_part("gripper_palm", wrist_size, wrist_center),
    ]

    joint_names: list[str] = []
    for finger_index, phi in enumerate(azimuths, start=1):
        parent = "gripper_palm"
        radial = (math.cos(phi), math.sin(phi), 0.0)
        inward_axis = (math.sin(phi), -math.cos(phi), 0.0)
        for link_index in range(1, 4):
            length = link_lengths[finger_index - 1][link_index - 1]
            width = link_widths[finger_index - 1][link_index - 1]
            depth = link_depths[finger_index - 1][link_index - 1]
            link_name = f"finger_{finger_index}_link_{link_index}"
            joint_name = f"finger_{finger_index}_joint_{link_index}"
            link = _empty_link(robot, link_name, (depth, width, length))
            _add_box_geometry(
                link,
                (depth, width, length),
                (0.0, 0.0, length * 0.5),
                color=(0.32 + 0.04 * link_index, 0.36, 0.43, 1.0),
            )
            if link_index == 1:
                joint_xyz = (
                    radial[0] * base_radii[finger_index - 1],
                    radial[1] * base_radii[finger_index - 1],
                    palm_z,
                )
            else:
                previous_length = link_lengths[finger_index - 1][link_index - 2]
                joint_xyz = (0.0, 0.0, previous_length)
            _revolute_joint(
                robot,
                joint_name,
                parent,
                link_name,
                xyz=joint_xyz,
                axis=inward_axis,
                upper=1.0,
                velocity=closed_angle / FRANKA_REAL_CLOSURE_TIME_S,
            )
            cloud_parts.append(
                _box_cloud_part(
                    link_name,
                    (depth, width, length),
                    (0.0, 0.0, length * 0.5),
                )
            )
            joint_names.append(joint_name)
            parent = link_name

    mean_reach = sum(sum(lengths) for lengths in link_lengths) / 3.0
    _add_grasp_target(robot, palm_z, 0.62 * mean_reach)
    manifest = _base_manifest_entry(
        gripper_id=gripper_id,
        category="three_finger_high_dof",
        topology_family="generated_three_finger_9dof_v1",
        root_dir=root_dir,
        joint_names=joint_names,
        open_positions=[0.0] * 9,
        closed_positions=[closed_angle] * 9,
        velocity_limit=closed_angle / FRANKA_REAL_CLOSURE_TIME_S,
        cloud_parts=cloud_parts,
        params=params,
    )
    return robot, manifest, params


def _write_asset(
    directory: Path,
    robot: ET.Element,
    params: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    urdf_path = directory / "robot.urdf"
    params_path = directory / "params.json"
    if not overwrite and (urdf_path.exists() or params_path.exists()):
        raise FileExistsError(
            f"Generated asset already exists: {directory}. Pass --overwrite to replace it."
        )
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    params_path.write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_family(
    output_root: Path,
    *,
    family: str,
    count: int,
    seed: int,
    overwrite: bool,
) -> Path:
    if count <= 0:
        raise ValueError("Generated family count must be positive")
    if family == "two_finger_revolute":
        builder = build_two_finger_revolute
    elif family == "three_finger_high_dof":
        builder = build_three_finger_high_dof
    else:
        raise ValueError(f"Unsupported family: {family}")

    entries = []
    for index in range(count):
        gripper_id = f"{family}_{index:06d}"
        relative_dir = Path(family) / f"{index:06d}"
        rng = random.Random((seed + 1) * 1_000_003 + index)
        robot, entry, params = builder(
            rng,
            gripper_id=gripper_id,
            root_dir=relative_dir.as_posix(),
        )
        _write_asset(
            output_root / relative_dir,
            robot,
            dict(entry["params"]),
            overwrite=overwrite,
        )
        entries.append(entry)

    manifest_path = output_root / f"{family}.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Generated manifest already exists: {manifest_path}. Pass --overwrite to replace it."
        )
    payload = {
        "schema_version": 1,
        "generator": "graspgenx_procedural_v1",
        "seed": seed,
        "family": family,
        "grippers": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from utils.assets import load_one_dof_gripper_manifest

    assets = load_one_dof_gripper_manifest(manifest_path, require_usd=False)
    signatures = {asset.topology_signature for asset in assets}
    if len(signatures) != 1:
        raise RuntimeError(
            f"Generated family {family!r} is not topology-homogeneous: {len(signatures)} signatures"
        )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--family",
        choices=("two_finger_revolute", "three_finger_high_dof", "all"),
        default="all",
    )
    parser.add_argument("--num-revolute", type=int, default=200)
    parser.add_argument("--num-three-finger", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    requested = (
        ("two_finger_revolute", "three_finger_high_dof")
        if args.family == "all"
        else (args.family,)
    )
    for family in requested:
        count = (
            args.num_revolute
            if family == "two_finger_revolute"
            else args.num_three_finger
        )
        manifest = generate_family(
            output_root,
            family=family,
            count=count,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"[generated] family={family} count={count} manifest={manifest}")


if __name__ == "__main__":
    main()

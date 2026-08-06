#!/usr/bin/env python3
"""Build RG2 and Robotiq 3F Franka URDFs from pinned, vendored descriptions."""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "thirdparty/rpdiff/src/rpdiff/descriptions/franka_panda_table"
ARM_SOURCE = ASSET_ROOT / "panda_2f140.urdf"
MOUNT_RPY = "0 0 3.926898163397"


def _arm_robot(name: str) -> ET.Element:
    source = ET.parse(ARM_SOURCE).getroot()
    robot = ET.Element("robot", {"name": name})
    for element in source:
        element_name = element.get("name", "")
        if element.tag == "link" and re.fullmatch(r"panda_link[0-8]", element_name):
            robot.append(copy.deepcopy(element))
        elif element.tag == "joint" and re.fullmatch(r"panda_joint[1-8]", element_name):
            robot.append(copy.deepcopy(element))
    return robot


def _origin(parent: ET.Element, xyz: str = "0 0 0", rpy: str = "0 0 0") -> None:
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _inertial(link: ET.Element, mass: float, inertia: float = 1.0e-3) -> None:
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial)
    ET.SubElement(inertial, "mass", {"value": f"{mass:.8g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia:.8g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{inertia:.8g}",
            "iyz": "0",
            "izz": f"{inertia:.8g}",
        },
    )


def _mesh_link(
    robot: ET.Element,
    name: str,
    visual_mesh: str,
    collision_mesh: str,
    *,
    mass: float,
) -> ET.Element:
    link = ET.SubElement(robot, "link", {"name": name})
    _inertial(link, mass)
    for geometry_kind, filename in (("visual", visual_mesh), ("collision", collision_mesh)):
        geometry_parent = ET.SubElement(link, geometry_kind)
        _origin(geometry_parent)
        geometry = ET.SubElement(geometry_parent, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": filename})
    return link


def _joint(
    robot: ET.Element,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    *,
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
    axis: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    mimic: tuple[str, float] | None = None,
) -> ET.Element:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": axis})
    if lower is not None and upper is not None:
        ET.SubElement(
            joint,
            "limit",
            {"lower": str(lower), "upper": str(upper), "effort": "24", "velocity": "2"},
        )
    if mimic is not None:
        ET.SubElement(
            joint,
            "mimic",
            {"joint": mimic[0], "multiplier": str(mimic[1]), "offset": "0"},
        )
    return joint


def _add_grasp_target(robot: ET.Element, *, height: float) -> None:
    target = ET.SubElement(robot, "link", {"name": "panda_grasptarget"})
    _inertial(target, 0.01, 1.0e-6)
    _joint(
        robot,
        "panda_grasptarget_hand",
        "fixed",
        "panda_link8",
        "panda_grasptarget",
        xyz=f"0 0 {height}",
    )


def _build_rg2() -> ET.Element:
    robot = _arm_robot("panda_onrobot_rg2")
    _joint(
        robot,
        "panda_hand_joint",
        "fixed",
        "panda_link8",
        "onrobot_rg2_base_link",
        rpy=MOUNT_RPY,
    )
    prefix = "meshes/onrobot_rg2"
    _mesh_link(
        robot,
        "onrobot_rg2_base_link",
        f"{prefix}/visual/base_link.stl",
        f"{prefix}/collision/base_link.stl",
        mass=0.7,
    )
    for side in ("left", "right"):
        for part in ("outer_knuckle", "inner_knuckle", "inner_finger"):
            _mesh_link(
                robot,
                f"{side}_{part}",
                f"{prefix}/visual/{part}.stl",
                f"{prefix}/collision/{part}.stl",
                mass=0.05,
            )
    _joint(
        robot,
        "finger_joint",
        "revolute",
        "onrobot_rg2_base_link",
        "left_outer_knuckle",
        xyz="0 -0.017178 0.125797",
        axis="-1 0 0",
        lower=-0.558505,
        upper=0.785398,
    )
    _joint(
        robot,
        "left_inner_knuckle_joint",
        "revolute",
        "onrobot_rg2_base_link",
        "left_inner_knuckle",
        xyz="0 -0.007678 0.142297",
        axis="1 0 0",
        lower=-0.785398,
        upper=0.785398,
        mimic=("finger_joint", -1.0),
    )
    _joint(
        robot,
        "left_inner_finger_joint",
        "revolute",
        "left_outer_knuckle",
        "left_inner_finger",
        xyz="0 -0.039592 0.038177",
        axis="1 0 0",
        lower=-0.872665,
        upper=0.872665,
        mimic=("finger_joint", 1.0),
    )
    _joint(
        robot,
        "right_outer_knuckle_joint",
        "revolute",
        "onrobot_rg2_base_link",
        "right_outer_knuckle",
        xyz="0 0.017178 0.125797",
        rpy="0 0 3.14159265359",
        axis="1 0 0",
        lower=-0.785398,
        upper=0.785398,
        mimic=("finger_joint", -1.0),
    )
    _joint(
        robot,
        "right_inner_knuckle_joint",
        "revolute",
        "onrobot_rg2_base_link",
        "right_inner_knuckle",
        xyz="0 0.007678 0.142297",
        rpy="0 0 -3.14159265359",
        axis="1 0 0",
        lower=-0.785398,
        upper=0.785398,
        mimic=("finger_joint", -1.0),
    )
    _joint(
        robot,
        "right_inner_finger_joint",
        "revolute",
        "right_outer_knuckle",
        "right_inner_finger",
        xyz="0 -0.039592 0.038177",
        axis="1 0 0",
        lower=-0.872665,
        upper=0.872665,
        mimic=("finger_joint", 1.0),
    )
    _add_grasp_target(robot, height=0.23)
    return robot


def _build_robotiq_3f() -> ET.Element:
    robot = _arm_robot("panda_robotiq_3f")
    source_path = ASSET_ROOT / "meshes/robotiq_3f/source/robotiq-3f-gripper_articulated.urdf"
    source = ET.parse(source_path).getroot()
    for element in source:
        if element.tag not in {"link", "joint"}:
            continue
        element = copy.deepcopy(element)
        name = element.get("name", "")
        if element.tag == "joint" and name in {"palm_finger_1_joint", "palm_finger_2_joint"}:
            element.set("type", "fixed")
            for child in list(element):
                if child.tag in {"axis", "limit"}:
                    element.remove(child)
        if element.tag == "joint" and re.fullmatch(r"finger_(?:1|2|middle)_joint_3", name):
            limit = element.find("limit")
            limit.set("lower", "-1.2217304764")
            limit.set("upper", "0")
            limit.set("effort", "20")
            limit.set("velocity", "2")
        elif element.tag == "joint" and re.fullmatch(r"finger_(?:1|2|middle)_joint_[12]", name):
            limit = element.find("limit")
            limit.set("lower", "0")
            limit.set("upper", "1.5707963268")
            limit.set("effort", "20")
            limit.set("velocity", "2")
        for mesh in element.findall(".//mesh"):
            old = mesh.get("filename", "")
            kind = "visual" if "/visual/" in old else "collision"
            source_name = Path(old).name
            if kind == "visual":
                source_name = f"{Path(source_name).stem}.dae"
            mesh.set("filename", f"meshes/robotiq_3f/{kind}/{source_name}")
        robot.append(element)
    palm = robot.find("link[@name='palm']")
    if palm.find("inertial") is None:
        _inertial(palm, 1.3, 6.0e-3)
    _joint(
        robot,
        "panda_hand_joint",
        "fixed",
        "panda_link8",
        "palm",
        # The upstream articulated model approaches along palm-local +Y.
        # Rotate that axis onto the generated-gripper Franka flange's +Z.
        rpy="1.57079632679 0 3.926898163397",
    )
    _add_grasp_target(robot, height=0.23)
    return robot


def _write(robot: ET.Element, path: Path) -> None:
    ET.indent(robot, space="  ")
    payload = ET.tostring(robot, encoding="utf-8", xml_declaration=True)
    if path.is_file() and path.read_bytes() == payload:
        print(f"[unchanged] {path}")
        return
    path.write_bytes(payload)
    print(f"[written] {path}")


def main() -> None:
    _write(_build_rg2(), ASSET_ROOT / "panda_rg2.urdf")
    _write(_build_robotiq_3f(), ASSET_ROOT / "panda_3f.urdf")


if __name__ == "__main__":
    main()

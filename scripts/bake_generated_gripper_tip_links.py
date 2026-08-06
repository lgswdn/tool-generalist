#!/usr/bin/env python3
"""Bake generated-gripper tip links into the moving finger links.

This rewrites only generated URDF files. USDs must be regenerated after this
script, because Isaac reads the robot topology from the converted USD assets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping


DEFAULT_GENERATED_DIR = Path("/mnt/project/world_model/tool_generalist/gripper/franka_with_diverse_hands")

LEFT_FINGER_LINK = "panda_leftfinger"
RIGHT_FINGER_LINK = "panda_rightfinger"
LEFT_TIP_LINK = "panda_leftfinger_tip"
RIGHT_TIP_LINK = "panda_rightfinger_tip"
LEFT_TIP_JOINT = "panda_leftfinger_tip_joint"
RIGHT_TIP_JOINT = "panda_rightfinger_tip_joint"
TIP_MESH_NAME = "finger_tip.STL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite URDFs in place. Without this flag the script only validates and reports changes.",
    )
    args = parser.parse_args()

    generated_dir = args.generated_dir.expanduser().resolve()
    if not generated_dir.is_dir():
        raise RuntimeError(f"Generated gripper directory does not exist: {generated_dir}")

    changed = 0
    checked = 0
    for hand_dir in _iter_hand_dirs(generated_dir):
        checked += 1
        urdf_path = hand_dir / "isaac.urdf"
        params_path = hand_dir / "params.json"
        mesh_path = hand_dir / "meshes" / "collision" / TIP_MESH_NAME
        if not urdf_path.is_file():
            raise RuntimeError(f"Missing generated gripper URDF: {urdf_path}")
        params = _read_params(params_path)
        has_tip = _require_nonnegative_param(params, "tip_cuboid_thickness", params_path) > 0.0
        if has_tip and not mesh_path.is_file():
            raise RuntimeError(f"{hand_dir.name}: has_tip=True but missing {mesh_path}")
        if not has_tip and mesh_path.exists():
            raise RuntimeError(f"{hand_dir.name}: has_tip=False but {mesh_path} exists")

        original = urdf_path.read_text(encoding="utf-8")
        updated = bake_tip_links_in_urdf(original, urdf_path=urdf_path, has_tip=has_tip)
        if updated != original:
            changed += 1
            if args.write:
                urdf_path.write_text(updated, encoding="utf-8")

    mode = "rewrote" if args.write else "would rewrite"
    print(f"Checked {checked} generated gripper URDFs; {mode} {changed}.")


def bake_tip_links_in_urdf(content: str, *, urdf_path: Path, has_tip: bool) -> str:
    has_tip_links = _has_link(content, LEFT_TIP_LINK) or _has_link(content, RIGHT_TIP_LINK)
    has_tip_joints = _has_joint(content, LEFT_TIP_JOINT) or _has_joint(content, RIGHT_TIP_JOINT)

    if has_tip:
        if has_tip_links or has_tip_joints:
            _require_identity_tip_joint(content, LEFT_TIP_JOINT, urdf_path)
            _require_identity_tip_joint(content, RIGHT_TIP_JOINT, urdf_path)
            content = _insert_tip_mesh_into_finger_link(content, LEFT_FINGER_LINK, LEFT_TIP_LINK, urdf_path)
            content = _insert_tip_mesh_into_finger_link(content, RIGHT_FINGER_LINK, RIGHT_TIP_LINK, urdf_path)
        _require_mesh_count(content, LEFT_FINGER_LINK, TIP_MESH_NAME, 2, urdf_path)
        _require_mesh_count(content, RIGHT_FINGER_LINK, TIP_MESH_NAME, 2, urdf_path)
    else:
        _require_mesh_count(content, LEFT_FINGER_LINK, TIP_MESH_NAME, 0, urdf_path)
        _require_mesh_count(content, RIGHT_FINGER_LINK, TIP_MESH_NAME, 0, urdf_path)

    if has_tip_links or has_tip_joints:
        content = _remove_link_and_joint(content, LEFT_TIP_LINK, LEFT_TIP_JOINT, urdf_path)
        content = _remove_link_and_joint(content, RIGHT_TIP_LINK, RIGHT_TIP_JOINT, urdf_path)

    for link_name in (LEFT_TIP_LINK, RIGHT_TIP_LINK):
        if _has_link(content, link_name):
            raise RuntimeError(f"{urdf_path}: unexpected remaining tip link {link_name!r}")
    for joint_name in (LEFT_TIP_JOINT, RIGHT_TIP_JOINT):
        if _has_joint(content, joint_name):
            raise RuntimeError(f"{urdf_path}: unexpected remaining tip joint {joint_name!r}")

    return content


def _iter_hand_dirs(generated_dir: Path) -> list[Path]:
    hand_dirs = sorted(path for path in generated_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    non_numeric = [path.name for path in hand_dirs if not path.name.isdigit()]
    if non_numeric:
        preview = ", ".join(non_numeric[:5])
        raise RuntimeError(f"Generated gripper directories must be numbered; found: {preview}")
    if not hand_dirs:
        raise RuntimeError(f"No generated gripper directories found under: {generated_dir}")
    return hand_dirs


def _read_params(path: Path) -> Mapping:
    if not path.is_file():
        raise RuntimeError(f"Missing generated gripper params: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Generated gripper params must be a JSON object: {path}")
    return data


def _require_nonnegative_param(params: Mapping, key: str, params_path: Path) -> float:
    if key not in params:
        raise RuntimeError(f"{params_path}: missing required parameter {key!r}")
    value = params[key]
    if isinstance(value, bool):
        raise RuntimeError(f"{params_path}: parameter {key!r} must be numeric")
    parsed = float(value)
    if parsed < 0.0:
        raise RuntimeError(f"{params_path}: parameter {key!r} must be >= 0")
    return parsed


def _has_link(content: str, link_name: str) -> bool:
    return re.search(rf'<link name="{re.escape(link_name)}">', content) is not None


def _has_joint(content: str, joint_name: str) -> bool:
    return re.search(rf'<joint name="{re.escape(joint_name)}"', content) is not None


def _require_link_block(content: str, link_name: str, urdf_path: Path) -> str:
    matches = re.findall(rf'\s*<link name="{re.escape(link_name)}">[\s\S]*?</link>\s*', content)
    if len(matches) != 1:
        raise RuntimeError(f"{urdf_path}: expected exactly one link {link_name!r}; found {len(matches)}")
    return matches[0]


def _require_single_child_block(parent_block: str, tag_name: str, link_name: str, urdf_path: Path) -> str:
    matches = re.findall(rf'(\s*<{tag_name}>[\s\S]*?</{tag_name}>)', parent_block)
    if len(matches) != 1:
        raise RuntimeError(
            f"{urdf_path}: expected exactly one <{tag_name}> block in link {link_name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _require_identity_tip_joint(content: str, joint_name: str, urdf_path: Path) -> None:
    pattern = rf'<joint name="{re.escape(joint_name)}"[^>]*>[\s\S]*?<origin[^>]*xyz="([^"]*?)"[^>]*rpy="([^"]*?)"[\s\S]*?</joint>'
    match = re.search(pattern, content)
    if match is None:
        raise RuntimeError(f"{urdf_path}: missing tip joint {joint_name!r}")
    xyz = [float(value) for value in match.group(1).split()]
    rpy = [float(value) for value in match.group(2).split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise RuntimeError(f"{urdf_path}: tip joint {joint_name!r} must have 3D xyz/rpy origin")
    if any(abs(value) > 1e-9 for value in xyz + rpy):
        raise RuntimeError(
            f"{urdf_path}: tip joint {joint_name!r} has non-identity origin; "
            "baking tips into finger links requires an identity fixed-joint transform"
        )


def _insert_tip_mesh_into_finger_link(content: str, finger_link_name: str, tip_link_name: str, urdf_path: Path) -> str:
    _require_mesh_count(content, finger_link_name, TIP_MESH_NAME, 0, urdf_path)
    tip_link_block = _require_link_block(content, tip_link_name, urdf_path)
    tip_visual = _require_single_child_block(tip_link_block, "visual", tip_link_name, urdf_path)
    tip_collision = _require_single_child_block(tip_link_block, "collision", tip_link_name, urdf_path)
    insertion = "\n" + tip_visual + "\n" + tip_collision
    pattern = rf'(<link name="{re.escape(finger_link_name)}">[\s\S]*?)(\n\s*</link>)'
    updated, count = re.subn(pattern, rf'\1{insertion}\2', content, count=1)
    if count != 1:
        raise RuntimeError(f"{urdf_path}: expected exactly one link {finger_link_name!r}; found {count}")
    return updated


def _remove_link_and_joint(content: str, link_name: str, joint_name: str, urdf_path: Path) -> str:
    content, link_count = re.subn(rf'\s*<link name="{re.escape(link_name)}">[\s\S]*?</link>\s*', '\n', content)
    if link_count != 1:
        raise RuntimeError(f"{urdf_path}: expected exactly one removable tip link {link_name!r}; found {link_count}")
    content, joint_count = re.subn(rf'\s*<joint name="{re.escape(joint_name)}"[^>]*>[\s\S]*?</joint>\s*', '\n', content)
    if joint_count != 1:
        raise RuntimeError(
            f"{urdf_path}: expected exactly one removable tip joint {joint_name!r}; found {joint_count}"
        )
    return content


def _require_mesh_count(content: str, link_name: str, mesh_name: str, expected_count: int, urdf_path: Path) -> None:
    link_block = _require_link_block(content, link_name, urdf_path)
    count = len(re.findall(rf'filename="[^"]*{re.escape(mesh_name)}"', link_block))
    if count != expected_count:
        raise RuntimeError(
            f"{urdf_path}: link {link_name!r} expected {expected_count} references to "
            f"{mesh_name!r}; found {count}"
        )


if __name__ == "__main__":
    main()

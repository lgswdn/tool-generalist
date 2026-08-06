#!/usr/bin/env python3
"""Build the explicit generated-gripper runtime manifest.

This is an offline asset step.  Runtime environments still consume only the
manifest path from paths.yaml and do not scan generated asset directories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GRIPPER_ROOT = Path("/mnt/project/world_model/tool_generalist/gripper")
DEFAULT_MANIFEST_PATH = DEFAULT_GRIPPER_ROOT / "generated_grippers.json"

ROBOT_GENERATED_SUBDIRS = {
    "franka": "franka_with_diverse_hands",
}

ROBOT_HAND_CONTRACTS = {
    "franka": {
        "palm_body_name": "plank_link",
        "ee_body_name": "eef_link",
        "ee_joint_name": "panda_eef_joint",
        "finger_body_names": ["panda_leftfinger", "panda_rightfinger"],
        "finger_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
        "fingertip_body_names": ["panda_leftfinger_tip", "panda_rightfinger_tip"],
        "finger_tip_joint_names": ["panda_leftfinger_tip_joint", "panda_rightfinger_tip_joint"],
    },
}

IDENTITY_TRANSFORM = {
    "translation": [0.0, 0.0, 0.0],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
}


def build_generated_gripper_manifest(
    *,
    gripper_root: str | Path = DEFAULT_GRIPPER_ROOT,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    robot_name: str = "franka",
    generated_subdir: str | None = None,
) -> dict[str, Any]:
    if robot_name not in ROBOT_HAND_CONTRACTS:
        raise ValueError(f"Unsupported generated gripper robot_name {robot_name!r}")

    gripper_root = Path(gripper_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    generated_subdir = generated_subdir or ROBOT_GENERATED_SUBDIRS[robot_name]
    generated_dir = gripper_root / generated_subdir
    if not generated_dir.is_dir():
        raise RuntimeError(f"Generated gripper directory does not exist: {generated_dir}")

    entries = [
        _build_manifest_entry(hand_dir, robot_name=robot_name)
        for hand_dir in _iter_generated_hand_dirs(generated_dir)
    ]
    manifest = {
        "schema_version": 1,
        "robot_name": robot_name,
        "generated_root": str(generated_dir),
        "grippers": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    from utils.assets.generated_gripper_assets import load_generated_gripper_manifest

    load_generated_gripper_manifest(output_path)
    return manifest


def _iter_generated_hand_dirs(generated_dir: Path) -> list[Path]:
    all_dirs = sorted(
        path for path in generated_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    legacy_dirs = [path.name for path in all_dirs if not path.name.isdigit()]
    if legacy_dirs:
        preview = ", ".join(legacy_dirs[:5])
        suffix = "" if len(legacy_dirs) <= 5 else f", ... ({len(legacy_dirs)} total)"
        raise RuntimeError(
            "Generated gripper directories must be numbered. "
            f"Found non-numbered directories under {generated_dir}: {preview}{suffix}. "
            "Regenerate with procedure_gen_hand.py --clean_output_dir to remove legacy UUID outputs."
        )
    hand_dirs = all_dirs
    if not hand_dirs:
        raise RuntimeError(f"No generated gripper directories found under: {generated_dir}")
    return hand_dirs


def _build_manifest_entry(hand_dir: Path, *, robot_name: str) -> dict[str, Any]:
    contract = ROBOT_HAND_CONTRACTS[robot_name]
    params_path = _require_file(hand_dir / "params.json")
    urdf_path = _require_file(hand_dir / "isaac.urdf")
    usd_path = _require_file(hand_dir / "isaac.usd")
    mesh_dir = _require_dir(hand_dir / "meshes" / "collision")
    plank_mesh = _require_file(mesh_dir / "plank.obj")
    finger_mesh = _require_file(mesh_dir / "finger.obj")

    params = _read_params(params_path)
    tip_cuboid_thickness = _require_nonnegative_param(params, "tip_cuboid_thickness", params_path)
    has_tip = tip_cuboid_thickness > 0.0
    if has_tip:
        _require_positive_param(params, "tip_cuboid_height", params_path)
        finger_tip_mesh = _require_file(mesh_dir / "finger_tip.STL")
    else:
        finger_tip_mesh = mesh_dir / "finger_tip.STL"
        if finger_tip_mesh.exists():
            raise RuntimeError(
                f"{hand_dir.name}: params.json declares no tip cuboid, but finger_tip.STL exists: "
                f"{finger_tip_mesh}"
            )

    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Generated gripper URDF is not valid XML: {urdf_path}") from exc
    _require_link(root, contract["palm_body_name"], urdf_path)
    _require_link(root, contract["ee_body_name"], urdf_path)
    ee_joint = _require_joint(root, contract["ee_joint_name"], urdf_path)
    if ee_joint.get("type") != "fixed":
        raise RuntimeError(f"{urdf_path}: ee joint {contract['ee_joint_name']!r} must be fixed")
    _require_joint_parent_child(
        ee_joint,
        contract["palm_body_name"],
        contract["ee_body_name"],
        urdf_path,
    )
    for link_name in contract["finger_body_names"]:
        _require_link(root, link_name, urdf_path)

    finger_joint_specs = []
    finger_open_limits = []
    for joint_name, child_name in zip(
        contract["finger_joint_names"],
        contract["finger_body_names"],
    ):
        joint = _require_joint(root, joint_name, urdf_path)
        if joint.get("type") != "prismatic":
            raise RuntimeError(f"{urdf_path}: joint {joint_name!r} must be prismatic")
        _require_joint_parent_child(joint, contract["palm_body_name"], child_name, urdf_path)
        finger_joint_specs.append(_parse_joint_spec(joint, urdf_path))
        finger_open_limits.append(_parse_zero_lower_upper_limit(joint, urdf_path))
    if not math.isclose(finger_open_limits[0], finger_open_limits[1], rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"{urdf_path}: generated gripper finger joints must have identical upper limits, "
            f"got {finger_open_limits}"
        )

    left_finger_mesh_frame = _mesh_origin_transform(
        root,
        contract["finger_body_names"][0],
        "finger.obj",
        urdf_path,
    )
    right_finger_mesh_frame = _mesh_origin_transform(
        root,
        contract["finger_body_names"][1],
        "finger.obj",
        urdf_path,
    )
    if left_finger_mesh_frame != right_finger_mesh_frame:
        raise RuntimeError(f"{urdf_path}: left/right finger mesh origins must be identical")

    entry: dict[str, Any] = {
        "id": hand_dir.name,
        "root_dir": str(hand_dir),
        "usd_path": usd_path.name,
        "urdf_path": urdf_path.name,
        "params_path": params_path.name,
        "mesh_dir": "meshes/collision",
        "plank_mesh": plank_mesh.name,
        "finger_mesh": finger_mesh.name,
        "finger_tip_mesh": finger_tip_mesh.name if has_tip else None,
        "has_tip": has_tip,
        "palm_body_name": contract["palm_body_name"],
        "ee_body_name": contract["ee_body_name"],
        "finger_body_names": contract["finger_body_names"],
        "finger_joint_names": contract["finger_joint_names"],
        "open_joint_pos": finger_open_limits[0],
        "mesh_to_body_frame": {
            "plank": _mesh_origin_transform(root, contract["palm_body_name"], "plank.obj", urdf_path),
            "finger": left_finger_mesh_frame,
        },
        "finger_joint_local_poses": finger_joint_specs,
        "fingertip_local_offsets": [
            _fingertip_local_offset(params, has_tip=has_tip, params_path=params_path),
            _fingertip_local_offset(params, has_tip=has_tip, params_path=params_path),
        ],
    }

    if has_tip:
        left_tip_mesh_frame = _mesh_origin_transform(
            root,
            contract["finger_body_names"][0],
            "finger_tip.STL",
            urdf_path,
        )
        right_tip_mesh_frame = _mesh_origin_transform(
            root,
            contract["finger_body_names"][1],
            "finger_tip.STL",
            urdf_path,
        )
        if left_tip_mesh_frame != right_tip_mesh_frame:
            raise RuntimeError(
                f"{urdf_path}: left/right baked finger tip mesh origins must be identical"
            )
        entry["mesh_to_body_frame"]["finger_tip"] = left_tip_mesh_frame
        entry["finger_tip_to_finger_frame"] = IDENTITY_TRANSFORM
    else:
        for link_name in contract["finger_body_names"]:
            if _count_collision_meshes(root, link_name, "finger_tip.STL") != 0:
                raise RuntimeError(
                    f"{urdf_path}: no-tip gripper must not contain baked finger_tip.STL "
                    f"under link {link_name!r}"
                )

    for link_name in contract["fingertip_body_names"]:
        if _find_link(root, link_name) is not None:
            raise RuntimeError(
                f"{urdf_path}: generated gripper tips must be baked into the finger links; "
                f"unexpected separate tip link {link_name!r}"
            )
    for joint_name in contract["finger_tip_joint_names"]:
        if _find_joint(root, joint_name) is not None:
            raise RuntimeError(
                f"{urdf_path}: generated gripper tips must be baked into the finger links; "
                f"unexpected separate tip joint {joint_name!r}"
            )

    return entry


def _read_params(params_path: Path) -> Mapping[str, Any]:
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, Mapping):
        raise RuntimeError(f"Generated gripper params must be a JSON object: {params_path}")
    return params


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Required generated gripper file does not exist: {path}")
    return path


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"Required generated gripper directory does not exist: {path}")
    return path


def _find_link(root: ET.Element, link_name: str) -> ET.Element | None:
    return root.find(f"./link[@name='{link_name}']")


def _require_link(root: ET.Element, link_name: str, urdf_path: Path) -> ET.Element:
    link = _find_link(root, link_name)
    if link is None:
        raise RuntimeError(f"{urdf_path}: missing required link {link_name!r}")
    return link


def _find_joint(root: ET.Element, joint_name: str) -> ET.Element | None:
    return root.find(f"./joint[@name='{joint_name}']")


def _require_joint(root: ET.Element, joint_name: str, urdf_path: Path) -> ET.Element:
    joint = _find_joint(root, joint_name)
    if joint is None:
        raise RuntimeError(f"{urdf_path}: missing required joint {joint_name!r}")
    return joint


def _require_joint_parent_child(
    joint: ET.Element,
    parent_name: str,
    child_name: str,
    urdf_path: Path,
) -> None:
    parent = joint.find("parent")
    child = joint.find("child")
    joint_name = joint.get("name")
    if parent is None or parent.get("link") != parent_name:
        raise RuntimeError(
            f"{urdf_path}: joint {joint_name!r} must have parent link {parent_name!r}"
        )
    if child is None or child.get("link") != child_name:
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} must have child link {child_name!r}")


def _parse_joint_spec(joint: ET.Element, urdf_path: Path) -> dict[str, list[float]]:
    joint_name = joint.get("name") or "<unnamed>"
    origin = joint.find("origin")
    axis = joint.find("axis")
    if origin is None or axis is None:
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} must define origin and axis")
    return {
        "origin_xyz": _parse_vec3_attr(origin, "xyz", urdf_path, f"{joint_name}.origin.xyz"),
        "origin_rpy": _parse_vec3_attr(origin, "rpy", urdf_path, f"{joint_name}.origin.rpy"),
        "axis_xyz": _parse_unit_axis(axis, urdf_path, f"{joint_name}.axis.xyz"),
    }


def _parse_zero_lower_upper_limit(joint: ET.Element, urdf_path: Path) -> float:
    joint_name = joint.get("name") or "<unnamed>"
    limit = joint.find("limit")
    if limit is None:
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} must define a limit")
    lower = _parse_float_attr(limit, "lower", urdf_path, f"{joint_name}.limit.lower")
    upper = _parse_float_attr(limit, "upper", urdf_path, f"{joint_name}.limit.upper")
    if not math.isclose(lower, 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} lower limit must be exactly 0")
    if upper <= 0.0:
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} upper limit must be > 0")
    return upper


def _mesh_origin_transform(
    root: ET.Element,
    link_name: str,
    expected_mesh_name: str,
    urdf_path: Path,
) -> dict[str, list[float]]:
    link = _require_link(root, link_name, urdf_path)
    matches = []
    for collision in link.findall("collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename")
        if filename is not None and Path(filename).name == expected_mesh_name:
            matches.append(collision)
    if len(matches) != 1:
        raise RuntimeError(
            f"{urdf_path}: link {link_name!r} must contain exactly one collision mesh "
            f"named {expected_mesh_name!r}; found {len(matches)}"
        )
    origin = matches[0].find("origin")
    if origin is None:
        raise RuntimeError(
            f"{urdf_path}: link {link_name!r} collision mesh {expected_mesh_name!r} "
            "must define an explicit origin"
        )
    return _origin_transform(origin, urdf_path, f"{link_name}.{expected_mesh_name}.origin")


def _count_collision_meshes(root: ET.Element, link_name: str, expected_mesh_name: str) -> int:
    link = _require_link(root, link_name, Path("<generated-gripper-urdf>"))
    count = 0
    for collision in link.findall("collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename")
        if filename is not None and Path(filename).name == expected_mesh_name:
            count += 1
    return count


def _joint_origin_transform(joint: ET.Element, urdf_path: Path) -> dict[str, list[float]]:
    joint_name = joint.get("name") or "<unnamed>"
    origin = joint.find("origin")
    if origin is None:
        raise RuntimeError(f"{urdf_path}: joint {joint_name!r} must define an explicit origin")
    return _origin_transform(origin, urdf_path, f"{joint_name}.origin")


def _origin_transform(origin: ET.Element, urdf_path: Path, label: str) -> dict[str, list[float]]:
    rpy = _parse_vec3_attr(origin, "rpy", urdf_path, f"{label}.rpy")
    return {
        "translation": _parse_vec3_attr(origin, "xyz", urdf_path, f"{label}.xyz"),
        "quat_wxyz": _rpy_to_quat_wxyz(rpy),
    }


def _fingertip_local_offset(
    params: Mapping[str, Any],
    *,
    has_tip: bool,
    params_path: Path,
) -> list[float]:
    finger_length = _require_positive_param(params, "finger_length", params_path)
    if has_tip:
        tip_cuboid_thickness = _require_positive_param(params, "tip_cuboid_thickness", params_path)
        z_offset = -0.5 * tip_cuboid_thickness
    else:
        tip_thickness = _require_positive_param(params, "tip_thickness", params_path)
        z_offset = -0.5 * tip_thickness
    return [finger_length, 0.0, z_offset]


def _require_positive_param(params: Mapping[str, Any], key: str, params_path: Path) -> float:
    value = _require_numeric_param(params, key, params_path)
    if value <= 0.0:
        raise RuntimeError(f"{params_path}: parameter {key!r} must be > 0")
    return value


def _require_nonnegative_param(params: Mapping[str, Any], key: str, params_path: Path) -> float:
    value = _require_numeric_param(params, key, params_path)
    if value < 0.0:
        raise RuntimeError(f"{params_path}: parameter {key!r} must be >= 0")
    return value


def _require_numeric_param(params: Mapping[str, Any], key: str, params_path: Path) -> float:
    if key not in params:
        raise RuntimeError(f"{params_path}: missing required parameter {key!r}")
    value = params[key]
    if isinstance(value, bool):
        raise RuntimeError(f"{params_path}: parameter {key!r} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{params_path}: parameter {key!r} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{params_path}: parameter {key!r} must be finite")
    return parsed


def _parse_vec3_attr(
    element: ET.Element,
    attr: str,
    urdf_path: Path,
    label: str,
) -> list[float]:
    value = element.get(attr)
    if value is None:
        raise RuntimeError(f"{urdf_path}: missing {label}")
    parts = value.split()
    if len(parts) != 3:
        raise RuntimeError(f"{urdf_path}: {label} must contain exactly three numbers")
    try:
        parsed = [float(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(f"{urdf_path}: {label} must contain exactly three numbers") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise RuntimeError(f"{urdf_path}: {label} must contain finite numbers")
    return parsed


def _parse_unit_axis(element: ET.Element, urdf_path: Path, label: str) -> list[float]:
    axis = _parse_vec3_attr(element, "xyz", urdf_path, label)
    norm = math.sqrt(sum(item * item for item in axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise RuntimeError(f"{urdf_path}: {label} must be unit length")
    return axis


def _parse_float_attr(element: ET.Element, attr: str, urdf_path: Path, label: str) -> float:
    value = element.get(attr)
    if value is None:
        raise RuntimeError(f"{urdf_path}: missing {label}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{urdf_path}: {label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{urdf_path}: {label} must be finite")
    return parsed


def _rpy_to_quat_wxyz(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    quat = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 0.0:
        raise RuntimeError(f"Invalid zero-norm quaternion from rpy={rpy!r}")
    return [item / norm for item in quat]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated_grippers.json.")
    parser.add_argument("--robot-name", default="franka", choices=sorted(ROBOT_HAND_CONTRACTS))
    parser.add_argument("--gripper-root", type=Path, default=DEFAULT_GRIPPER_ROOT)
    parser.add_argument(
        "--generated-subdir",
        type=str,
        default=None,
        help="Generated hand subdirectory. Defaults to the robot's generator convention.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_generated_gripper_manifest(
        gripper_root=args.gripper_root,
        output_path=args.output,
        robot_name=args.robot_name,
        generated_subdir=args.generated_subdir,
    )
    print(f"Wrote {len(manifest['grippers'])} generated gripper entries to {args.output}")


if __name__ == "__main__":
    main()

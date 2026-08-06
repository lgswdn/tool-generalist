"""Manifest contract for heterogeneous grippers with one policy control DoF."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ONE_DOF_GRIPPER_CATEGORIES = frozenset(
    {
        "robotiq_like",
        "rg_like",
        "three_finger",
        "two_finger_revolute",
        "three_finger_high_dof",
    }
)
ONE_DOF_CONTROL_ADAPTERS = frozenset({"primary_joint_with_mimics", "joint_synergy"})


class OneDofGripperAssetError(ValueError):
    """Raised when a one-DoF gripper manifest violates the runtime contract."""


@dataclass(frozen=True)
class TransformSpec:
    translation: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class GripperCloudPart:
    body_name: str
    geometry_type: str
    mesh_path: Path | None
    mesh_scale: tuple[float, float, float]
    box_size: tuple[float, float, float] | None
    cylinder_radius: float | None
    cylinder_length: float | None
    geometry_to_body: TransformSpec


@dataclass(frozen=True)
class GripperActuatorSpec:
    effort_limit: float
    stiffness: float
    damping: float
    armature: float
    velocity_limit: float


@dataclass(frozen=True)
class OneDofGripperAsset:
    gripper_id: str
    category: str
    topology_family: str
    manifest_path: Path
    root_dir: Path
    urdf_path: Path
    usd_path: Path
    palm_body_name: str
    ee_body_name: str
    grasp_frame_body_name: str
    grasp_frame_offset: TransformSpec
    actuated_joint_names: tuple[str, ...]
    measured_joint_name: str
    open_joint_positions: tuple[float, ...]
    closed_joint_positions: tuple[float, ...]
    control_adapter: str
    actuator: GripperActuatorSpec
    cloud_parts: tuple[GripperCloudPart, ...]
    params: Mapping[str, Any]

    @property
    def topology_signature(self) -> tuple[tuple[str | None, ...], tuple[tuple[str | None, ...], ...]]:
        """Return the PhysX-relevant link/joint graph parsed from the source URDF."""

        root = ET.parse(self.urdf_path).getroot()
        links = tuple(link.get("name") for link in root.findall("link"))
        joints = []
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            joints.append(
                (
                    joint.get("name"),
                    joint.get("type"),
                    None if parent is None else parent.get("link"),
                    None if child is None else child.get("link"),
                )
            )
        return links, tuple(joints)


def load_one_dof_gripper_manifest(
    path: str | Path,
    *,
    expected_root: str | Path | None = None,
    require_usd: bool = True,
) -> list[OneDofGripperAsset]:
    """Load assets that all expose one normalized policy-controlled gripper command."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise OneDofGripperAssetError(f"One-DoF gripper manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise OneDofGripperAssetError("One-DoF gripper manifest must be an object with schema_version=1")
    entries = payload.get("grippers")
    if not isinstance(entries, list) or not entries:
        raise OneDofGripperAssetError("One-DoF gripper manifest must contain a non-empty 'grippers' list")

    assets = [
        _parse_asset(entry, manifest_path, index, require_usd=require_usd)
        for index, entry in enumerate(entries)
    ]
    if require_usd:
        for asset in assets:
            validate_one_dof_gripper_usd(asset)
    ids = [asset.gripper_id for asset in assets]
    if len(ids) != len(set(ids)):
        raise OneDofGripperAssetError("One-DoF gripper manifest contains duplicate gripper ids")
    official_categories = [
        asset.category for asset in assets if not bool(asset.params.get("generated", False))
    ]
    if len(official_categories) != len(set(official_categories)):
        raise OneDofGripperAssetError(
            "One-DoF gripper manifest permits only one official model per category"
        )
    if expected_root is not None:
        root = Path(expected_root).expanduser().resolve()
        for asset in assets:
            try:
                asset.root_dir.relative_to(root)
            except ValueError as exc:
                raise OneDofGripperAssetError(
                    f"One-DoF gripper {asset.gripper_id!r} escapes configured root {root}: {asset.root_dir}"
                ) from exc
    return assets


def validate_one_dof_gripper_usd(asset: OneDofGripperAsset) -> None:
    """Validate the runtime PhysX constraint corresponding to each URDF mimic."""

    if asset.control_adapter != "primary_joint_with_mimics":
        return
    try:
        from pxr import Usd
    except ImportError as exc:
        raise OneDofGripperAssetError(
            "Validating a one-DoF gripper USD requires the Isaac Sim pxr runtime"
        ) from exc
    stage = Usd.Stage.Open(str(asset.usd_path))
    if stage is None:
        raise OneDofGripperAssetError(f"Could not open one-DoF gripper USD: {asset.usd_path}")
    urdf_root = ET.parse(asset.urdf_path).getroot()
    mimic_joint_names = {
        joint.get("name")
        for joint in urdf_root.findall("joint")
        if joint.find("mimic") is not None
    }
    usd_joint_prims = {
        prim.GetName(): prim
        for prim in stage.TraverseAll()
        if prim.GetTypeName() in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    }
    missing_joints = sorted(name for name in mimic_joint_names if name not in usd_joint_prims)
    missing_apis = sorted(
        name
        for name in mimic_joint_names
        if name in usd_joint_prims
        and not any(
            prop.startswith("physxMimicJoint:") and prop.endswith(":referenceJoint")
            for prop in usd_joint_prims[name].GetPropertyNames()
        )
    )
    compliant_mimics = []
    for name in sorted(mimic_joint_names):
        prim = usd_joint_prims.get(name)
        if prim is None or name in missing_apis:
            continue
        axes = {
            prop.split(":", 2)[1]
            for prop in prim.GetPropertyNames()
            if prop.startswith("physxMimicJoint:") and prop.endswith(":referenceJoint")
        }
        for axis in axes:
            natural_frequency = prim.GetAttribute(
                f"physxMimicJoint:{axis}:naturalFrequency"
            ).Get()
            damping_ratio = prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Get()
            if float(natural_frequency or 0.0) > 0.0 or float(damping_ratio or 0.0) > 0.0:
                compliant_mimics.append(
                    (name, float(natural_frequency or 0.0), float(damping_ratio or 0.0))
                )
    if missing_joints or missing_apis or compliant_mimics:
        raise OneDofGripperAssetError(
            f"One-DoF gripper USD {asset.usd_path} does not preserve the URDF mimic contract: "
            f"missing_joints={missing_joints}, missing_physx_mimic_api={missing_apis}, "
            f"compliant_mimics={compliant_mimics}. "
            "Reconvert it with scripts/convert_one_dof_gripper.py --force."
        )


def _parse_asset(
    raw: Any,
    manifest_path: Path,
    index: int,
    *,
    require_usd: bool,
) -> OneDofGripperAsset:
    manifest_dir = manifest_path.parent
    if not isinstance(raw, Mapping):
        raise OneDofGripperAssetError(f"One-DoF gripper entry {index} must be an object")
    gripper_id = _string(raw, "id", index)
    category = _string(raw, "category", gripper_id)
    if category not in ONE_DOF_GRIPPER_CATEGORIES:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} category must be one of {sorted(ONE_DOF_GRIPPER_CATEGORIES)}"
        )
    topology_family = _string(raw, "topology_family", gripper_id)
    root_dir = _path(_string(raw, "root_dir", gripper_id), manifest_dir)
    if not root_dir.is_dir():
        raise OneDofGripperAssetError(f"One-DoF gripper root does not exist: {root_dir}")
    urdf_path = _path(_string(raw, "urdf_path", gripper_id), root_dir)
    usd_path = _path(_string(raw, "usd_path", gripper_id), root_dir)
    if not urdf_path.is_file():
        raise OneDofGripperAssetError(f"One-DoF gripper URDF does not exist: {urdf_path}")
    if require_usd and not usd_path.is_file():
        raise OneDofGripperAssetError(
            f"One-DoF gripper USD does not exist: {usd_path}. Run scripts/convert_one_dof_gripper.py."
        )

    root = ET.parse(urdf_path).getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    joint_by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    palm_body_name = _known_name(raw, "palm_body_name", gripper_id, link_names, "body")
    ee_body_name = _known_name(raw, "ee_body_name", gripper_id, link_names, "body")

    grasp = _mapping(raw, "grasp_frame", gripper_id)
    grasp_frame_body_name = _known_name(grasp, "body_name", gripper_id, link_names, "body")
    grasp_frame_offset = _transform(grasp.get("offset", {}), f"{gripper_id}.grasp_frame.offset")

    control = _mapping(raw, "control", gripper_id)
    if control.get("command_dim") != 1:
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} control.command_dim must equal 1")
    control_adapter = _string(control, "adapter", gripper_id)
    if control_adapter not in ONE_DOF_CONTROL_ADAPTERS:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} control.adapter must be one of {sorted(ONE_DOF_CONTROL_ADAPTERS)}"
        )
    actuated_joint_names = _string_tuple(control, "actuated_joint_names", gripper_id)
    if not actuated_joint_names:
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} needs at least one actuated joint")
    for joint_name in actuated_joint_names:
        if joint_name not in joint_by_name:
            raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} has unknown joint {joint_name!r}")
    measured_joint_name = _string(control, "measured_joint_name", gripper_id)
    if measured_joint_name not in actuated_joint_names:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} measured_joint_name must be actuated"
        )
    open_positions = _float_tuple(control, "open_joint_positions", gripper_id)
    closed_positions = _float_tuple(control, "closed_joint_positions", gripper_id)
    if len(open_positions) != len(actuated_joint_names) or len(closed_positions) != len(actuated_joint_names):
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} open/closed positions must match actuated_joint_names"
        )
    if all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(open_positions, closed_positions)):
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} has no open-to-closed motion")
    _validate_joint_targets(
        gripper_id=gripper_id,
        joint_by_name=joint_by_name,
        joint_names=actuated_joint_names,
        open_positions=open_positions,
        closed_positions=closed_positions,
    )
    actuator = _actuator_spec(control.get("actuator", {}), gripper_id)

    raw_parts = raw.get("cloud_parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} needs cloud_parts")
    cloud_parts = tuple(_cloud_part(part, root_dir, gripper_id, link_names) for part in raw_parts)
    if control_adapter == "primary_joint_with_mimics":
        _validate_primary_joint_with_mimics(
            gripper_id=gripper_id,
            joint_by_name=joint_by_name,
            primary_joint_names=actuated_joint_names,
            cloud_body_names={part.body_name for part in cloud_parts},
        )
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} params must be an object")
    return OneDofGripperAsset(
        gripper_id=gripper_id,
        category=category,
        topology_family=topology_family,
        manifest_path=manifest_path,
        root_dir=root_dir,
        urdf_path=urdf_path,
        usd_path=usd_path,
        palm_body_name=palm_body_name,
        ee_body_name=ee_body_name,
        grasp_frame_body_name=grasp_frame_body_name,
        grasp_frame_offset=grasp_frame_offset,
        actuated_joint_names=actuated_joint_names,
        measured_joint_name=measured_joint_name,
        open_joint_positions=open_positions,
        closed_joint_positions=closed_positions,
        control_adapter=control_adapter,
        actuator=actuator,
        cloud_parts=cloud_parts,
        params=params,
    )


def _actuator_spec(raw: Any, gripper_id: str) -> GripperActuatorSpec:
    if not isinstance(raw, Mapping):
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} control.actuator must be an object"
        )
    defaults = {
        "effort_limit": 24.0,
        "stiffness": 275.0,
        "damping": 0.06,
        "armature": 5.0e-3,
        "velocity_limit": 2.0,
    }
    values: dict[str, float] = {}
    for name, default in defaults.items():
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError) as exc:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} actuator {name!r} must be numeric"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} actuator {name!r} must be finite and positive"
            )
        values[name] = value
    return GripperActuatorSpec(**values)


def _validate_joint_targets(
    *,
    gripper_id: str,
    joint_by_name: Mapping[str | None, ET.Element],
    joint_names: tuple[str, ...],
    open_positions: tuple[float, ...],
    closed_positions: tuple[float, ...],
) -> None:
    for joint_name, open_pos, closed_pos in zip(joint_names, open_positions, closed_positions):
        if not math.isfinite(open_pos) or not math.isfinite(closed_pos):
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} joint {joint_name!r} targets must be finite"
            )
        limit = joint_by_name[joint_name].find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} joint {joint_name!r} needs URDF limits"
            )
        try:
            lower = float(limit.get("lower", ""))
            upper = float(limit.get("upper", ""))
        except ValueError as exc:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} joint {joint_name!r} has invalid URDF limits"
            ) from exc
        if lower > upper or any(value < lower or value > upper for value in (open_pos, closed_pos)):
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} joint {joint_name!r} open/closed targets "
                f"must lie in [{lower}, {upper}]"
            )


def _validate_primary_joint_with_mimics(
    *,
    gripper_id: str,
    joint_by_name: Mapping[str | None, ET.Element],
    primary_joint_names: tuple[str, ...],
    cloud_body_names: set[str],
) -> None:
    if len(primary_joint_names) != 1:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} primary_joint_with_mimics requires exactly one primary joint"
        )
    primary_name = primary_joint_names[0]
    mimic_count = 0
    for joint_name, joint in joint_by_name.items():
        child = joint.find("child")
        if child is None or child.get("link") not in cloud_body_names or joint.get("type") == "fixed":
            continue
        if joint_name == primary_name:
            continue
        mimic = joint.find("mimic")
        if mimic is None or mimic.get("joint") != primary_name:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} movable cloud joint {joint_name!r} "
                f"must mimic primary joint {primary_name!r}"
            )
        try:
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
        except ValueError as exc:
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} mimic joint {joint_name!r} has invalid coefficients"
            ) from exc
        if not math.isfinite(multiplier) or not math.isfinite(offset) or math.isclose(multiplier, 0.0):
            raise OneDofGripperAssetError(
                f"One-DoF gripper {gripper_id!r} mimic joint {joint_name!r} needs finite, non-zero motion"
            )
        mimic_count += 1
    if mimic_count == 0:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} primary_joint_with_mimics found no URDF mimic joints"
        )


def _cloud_part(raw: Any, root_dir: Path, gripper_id: str, link_names: set[str | None]) -> GripperCloudPart:
    if not isinstance(raw, Mapping):
        raise OneDofGripperAssetError(f"One-DoF gripper {gripper_id!r} cloud part must be an object")
    body_name = _known_name(raw, "body_name", gripper_id, link_names, "body")
    geometry_type = _string(raw, "geometry_type", gripper_id)
    if geometry_type == "mesh":
        mesh_path = _path(_string(raw, "mesh_path", gripper_id), root_dir)
        if not mesh_path.is_file():
            raise OneDofGripperAssetError(f"One-DoF gripper cloud mesh does not exist: {mesh_path}")
        mesh_scale = _vec(raw.get("mesh_scale", [1.0, 1.0, 1.0]), 3, f"{gripper_id}.mesh_scale")
        box_size = None
        cylinder_radius = None
        cylinder_length = None
    elif geometry_type == "box":
        mesh_path = None
        mesh_scale = (1.0, 1.0, 1.0)
        box_size = _vec(raw.get("box_size"), 3, f"{gripper_id}.box_size", positive=True)
        cylinder_radius = None
        cylinder_length = None
    elif geometry_type == "cylinder":
        mesh_path = None
        mesh_scale = (1.0, 1.0, 1.0)
        box_size = None
        cylinder_radius = _positive_float(
            raw.get("cylinder_radius"), f"{gripper_id}.cylinder_radius"
        )
        cylinder_length = _positive_float(
            raw.get("cylinder_length"), f"{gripper_id}.cylinder_length"
        )
    else:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {gripper_id!r} cloud geometry_type must be mesh, box, or cylinder"
        )
    return GripperCloudPart(
        body_name=body_name,
        geometry_type=geometry_type,
        mesh_path=mesh_path,
        mesh_scale=mesh_scale,
        box_size=box_size,
        cylinder_radius=cylinder_radius,
        cylinder_length=cylinder_length,
        geometry_to_body=_transform(raw.get("geometry_to_body", {}), f"{gripper_id}.geometry_to_body"),
    )


def _positive_float(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise OneDofGripperAssetError(f"{label} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise OneDofGripperAssetError(f"{label} must be finite and positive")
    return value


def _transform(raw: Any, label: str) -> TransformSpec:
    if not isinstance(raw, Mapping):
        raise OneDofGripperAssetError(f"{label} must be an object")
    translation = _vec(raw.get("translation", [0.0, 0.0, 0.0]), 3, f"{label}.translation")
    quat = _vec(raw.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), 4, f"{label}.quat_wxyz")
    norm = math.sqrt(sum(value * value for value in quat))
    if not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise OneDofGripperAssetError(f"{label}.quat_wxyz must be unit length")
    return TransformSpec(translation=translation, quat_wxyz=quat)


def _mapping(raw: Mapping[str, Any], key: str, label: Any) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} field {key!r} must be an object")
    return value


def _string(raw: Mapping[str, Any], key: str, label: Any) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} field {key!r} must be a string")
    return value


def _known_name(
    raw: Mapping[str, Any], key: str, label: Any, known: set[str | None], kind: str
) -> str:
    value = _string(raw, key, label)
    if value not in known:
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} has unknown {kind} {value!r}")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str, label: Any) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} field {key!r} must be a string list")
    return tuple(value)


def _float_tuple(raw: Mapping[str, Any], key: str, label: Any) -> tuple[float, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} field {key!r} must be a number list")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OneDofGripperAssetError(
            f"One-DoF gripper {label!r} field {key!r} must be a number list"
        ) from exc
    if any(not math.isfinite(item) for item in result):
        raise OneDofGripperAssetError(f"One-DoF gripper {label!r} field {key!r} must be finite")
    return result


def _vec(raw: Any, length: int, label: str, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise OneDofGripperAssetError(f"{label} must contain {length} numbers")
    try:
        value = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise OneDofGripperAssetError(f"{label} must contain {length} numbers") from exc
    if any(not math.isfinite(item) or (positive and item <= 0.0) for item in value):
        raise OneDofGripperAssetError(f"{label} contains invalid values")
    return value


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()

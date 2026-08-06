"""Generated gripper manifest and metadata contract helpers."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class GeneratedGripperAssetError(ValueError):
    """Raised when a generated-gripper manifest violates the asset contract."""


@dataclass(frozen=True)
class RigidTransformSpec:
    translation: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class PrismaticJointSpec:
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class GeneratedGripperAsset:
    gripper_id: str
    root_dir: Path
    usd_path: Path
    params_path: Path
    mesh_dir: Path
    plank_mesh: Path
    finger_mesh: Path
    finger_tip_mesh: Path | None
    has_tip: bool
    palm_body_name: str
    ee_body_name: str
    finger_body_names: tuple[str, str]
    finger_joint_names: tuple[str, str]
    open_joint_pos: float
    mesh_to_body_frame: Mapping[str, RigidTransformSpec]
    finger_joint_local_poses: tuple[PrismaticJointSpec, PrismaticJointSpec]
    finger_tip_to_finger_frame: RigidTransformSpec | None
    fingertip_body_names: tuple[str, str] | None
    fingertip_local_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    params: Mapping[str, Any]


def load_generated_gripper_manifest(
    path: str | Path,
    *,
    expected_root: str | Path | None = None,
) -> list[GeneratedGripperAsset]:
    """Load an explicit generated-gripper manifest.

    The manifest must be either a list of entries or an object with a
    ``grippers`` list.  Paths inside each entry are explicit: ``root_dir`` is
    resolved relative to the manifest file; asset paths are resolved relative
    to ``root_dir`` unless they are absolute.
    """

    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise GeneratedGripperAssetError(f"Generated gripper manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, Mapping):
        entries = data.get("grippers")
    else:
        entries = data
    if not isinstance(entries, list):
        raise GeneratedGripperAssetError(
            f"Generated gripper manifest must be a list or contain a 'grippers' list: {manifest_path}"
        )
    if not entries:
        raise GeneratedGripperAssetError(f"Generated gripper manifest is empty: {manifest_path}")

    manifest_dir = manifest_path.parent
    assets = [_parse_entry(entry, manifest_dir, index) for index, entry in enumerate(entries)]
    if expected_root is not None:
        configured_root = Path(expected_root).expanduser().resolve()
        for asset in assets:
            try:
                asset.root_dir.resolve().relative_to(configured_root)
            except ValueError as exc:
                raise GeneratedGripperAssetError(
                    "Generated gripper manifest entry "
                    f"{asset.gripper_id!r} points outside configured root "
                    f"{configured_root}: {asset.root_dir}"
                ) from exc
    ids = [asset.gripper_id for asset in assets]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise GeneratedGripperAssetError(
            f"Generated gripper manifest contains duplicate gripper ids: {duplicates}"
        )
    return assets


def load_generated_gripper_manifest_entry(
    path: str | Path,
    gripper_id: str,
) -> GeneratedGripperAsset:
    """Load and validate one named entry without touching unrelated assets."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise GeneratedGripperAssetError(
            f"Generated gripper manifest does not exist: {manifest_path}"
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("grippers") if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise GeneratedGripperAssetError(
            "Generated gripper manifest must be a list or contain a 'grippers' "
            f"list: {manifest_path}"
        )
    matches = [
        (index, entry)
        for index, entry in enumerate(entries)
        if isinstance(entry, Mapping) and entry.get("id") == gripper_id
    ]
    if len(matches) != 1:
        raise GeneratedGripperAssetError(
            f"Expected exactly one gripper {gripper_id!r} in {manifest_path}, "
            f"found {len(matches)}"
        )
    index, entry = matches[0]
    return _parse_entry(entry, manifest_path.parent, index)


def _parse_entry(entry: Any, manifest_dir: Path, index: int) -> GeneratedGripperAsset:
    if not isinstance(entry, Mapping):
        raise GeneratedGripperAssetError(f"Generated gripper manifest entry {index} must be an object")

    gripper_id = _require_str(entry, "id", index)
    root_dir = _require_dir(_resolve_path(_require_str(entry, "root_dir", index), manifest_dir))
    usd_path = _require_file(_entry_path(entry, "usd_path", index, root_dir))
    params_path = _require_file(_entry_path(entry, "params_path", index, root_dir))
    mesh_dir = _require_dir(_entry_path(entry, "mesh_dir", index, root_dir))
    plank_mesh = _require_file(_mesh_path(entry, "plank_mesh", index, mesh_dir))
    finger_mesh = _require_file(_mesh_path(entry, "finger_mesh", index, mesh_dir))

    has_tip = _require_bool(entry, "has_tip", index)
    raw_tip_mesh = entry.get("finger_tip_mesh")
    if has_tip:
        finger_tip_mesh = _require_file(_mesh_path(entry, "finger_tip_mesh", index, mesh_dir))
    else:
        if raw_tip_mesh not in (None, ""):
            raise GeneratedGripperAssetError(
                f"Generated gripper entry {gripper_id!r} has has_tip=False but finger_tip_mesh is set"
            )
        finger_tip_mesh = None

    params = _load_params(params_path, gripper_id)
    palm_body_name = _require_str(entry, "palm_body_name", index)
    ee_body_name = _require_str(entry, "ee_body_name", index)
    finger_body_names = _require_str_pair(entry, "finger_body_names", index)
    finger_joint_names = _require_str_pair(entry, "finger_joint_names", index)
    open_joint_pos = _require_positive_float(entry, "open_joint_pos", index)

    mesh_to_body_frame = _parse_mesh_to_body_frame(entry, gripper_id, has_tip)
    finger_tip_to_finger_frame = (
        _parse_transform(_require_mapping(entry, "finger_tip_to_finger_frame", index), "finger_tip_to_finger_frame")
        if has_tip
        else None
    )
    if not has_tip and "finger_tip_to_finger_frame" in entry:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} has has_tip=False but finger_tip_to_finger_frame is set"
        )

    finger_joint_local_poses = _parse_joint_specs(entry, root_dir, gripper_id, finger_joint_names)
    fingertip_body_names, fingertip_local_offsets = _parse_fingertip_contract(entry, gripper_id)

    return GeneratedGripperAsset(
        gripper_id=gripper_id,
        root_dir=root_dir,
        usd_path=usd_path,
        params_path=params_path,
        mesh_dir=mesh_dir,
        plank_mesh=plank_mesh,
        finger_mesh=finger_mesh,
        finger_tip_mesh=finger_tip_mesh,
        has_tip=has_tip,
        palm_body_name=palm_body_name,
        ee_body_name=ee_body_name,
        finger_body_names=finger_body_names,
        finger_joint_names=finger_joint_names,
        open_joint_pos=open_joint_pos,
        mesh_to_body_frame=mesh_to_body_frame,
        finger_joint_local_poses=finger_joint_local_poses,
        finger_tip_to_finger_frame=finger_tip_to_finger_frame,
        fingertip_body_names=fingertip_body_names,
        fingertip_local_offsets=fingertip_local_offsets,
        params=params,
    )


def _parse_mesh_to_body_frame(
    entry: Mapping[str, Any],
    gripper_id: str,
    has_tip: bool,
) -> dict[str, RigidTransformSpec]:
    raw = _require_mapping(entry, "mesh_to_body_frame", gripper_id)
    required = ("plank", "finger") + (("finger_tip",) if has_tip else ())
    transforms = {
        name: _parse_transform(_require_mapping(raw, name, gripper_id), f"mesh_to_body_frame.{name}")
        for name in required
    }
    if not has_tip and "finger_tip" in raw:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} has has_tip=False but mesh_to_body_frame.finger_tip is set"
        )
    return transforms


def _parse_joint_specs(
    entry: Mapping[str, Any],
    root_dir: Path,
    gripper_id: str,
    finger_joint_names: tuple[str, str],
) -> tuple[PrismaticJointSpec, PrismaticJointSpec]:
    raw_specs = entry.get("finger_joint_local_poses")
    if raw_specs is not None:
        if not isinstance(raw_specs, list) or len(raw_specs) != 2:
            raise GeneratedGripperAssetError(
                f"Generated gripper entry {gripper_id!r} finger_joint_local_poses must contain two entries"
            )
        return (
            _parse_joint_spec(raw_specs[0], f"{gripper_id}.finger_joint_local_poses[0]"),
            _parse_joint_spec(raw_specs[1], f"{gripper_id}.finger_joint_local_poses[1]"),
        )

    raw_urdf = entry.get("urdf_path")
    if raw_urdf in (None, ""):
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} must define finger_joint_local_poses "
            "or an explicit urdf_path for deterministic joint parsing"
        )
    urdf_path = _require_file(_resolve_path(str(raw_urdf), root_dir))
    return _parse_joint_specs_from_urdf(urdf_path, gripper_id, finger_joint_names)


def _parse_joint_specs_from_urdf(
    urdf_path: Path,
    gripper_id: str,
    finger_joint_names: tuple[str, str],
) -> tuple[PrismaticJointSpec, PrismaticJointSpec]:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} has invalid URDF XML: {urdf_path}"
        ) from exc

    specs: list[PrismaticJointSpec] = []
    for joint_name in finger_joint_names:
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise GeneratedGripperAssetError(
                f"Generated gripper entry {gripper_id!r} URDF is missing finger joint {joint_name!r}: {urdf_path}"
            )
        if joint.get("type") != "prismatic":
            raise GeneratedGripperAssetError(
                f"Generated gripper entry {gripper_id!r} joint {joint_name!r} must be prismatic"
            )
        origin = joint.find("origin")
        axis = joint.find("axis")
        if origin is None or axis is None:
            raise GeneratedGripperAssetError(
                f"Generated gripper entry {gripper_id!r} joint {joint_name!r} must define origin and axis"
            )
        specs.append(
            _validate_joint_spec(
                PrismaticJointSpec(
                    origin_xyz=_parse_vec_attr(origin, "xyz", f"{joint_name}.origin.xyz"),
                    origin_rpy=_parse_vec_attr(origin, "rpy", f"{joint_name}.origin.rpy"),
                    axis_xyz=_parse_vec_attr(axis, "xyz", f"{joint_name}.axis.xyz"),
                ),
                f"{gripper_id}.{joint_name}",
            )
        )
    return specs[0], specs[1]


def _parse_fingertip_contract(
    entry: Mapping[str, Any],
    gripper_id: str,
) -> tuple[
    tuple[str, str] | None,
    tuple[tuple[float, float, float], tuple[float, float, float]] | None,
]:
    has_body_names = entry.get("fingertip_body_names") is not None
    has_offsets = entry.get("fingertip_local_offsets") is not None
    if has_body_names == has_offsets:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} must define exactly one of "
            "fingertip_body_names or fingertip_local_offsets"
        )
    if has_body_names:
        return _require_str_pair(entry, "fingertip_body_names", gripper_id), None

    raw_offsets = entry["fingertip_local_offsets"]
    if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} fingertip_local_offsets must contain two vec3 entries"
        )
    return None, (
        _require_vec3(raw_offsets[0], f"{gripper_id}.fingertip_local_offsets[0]"),
        _require_vec3(raw_offsets[1], f"{gripper_id}.fingertip_local_offsets[1]"),
    )


def _parse_transform(raw: Mapping[str, Any], label: str) -> RigidTransformSpec:
    translation = _require_vec3(raw.get("translation"), f"{label}.translation")
    quat = _require_vec4(raw.get("quat_wxyz"), f"{label}.quat_wxyz")
    norm = math.sqrt(sum(value * value for value in quat))
    if abs(norm - 1.0) > 1e-4:
        raise GeneratedGripperAssetError(f"{label}.quat_wxyz must be unit length")
    return RigidTransformSpec(translation=translation, quat_wxyz=quat)


def _parse_joint_spec(raw: Any, label: str) -> PrismaticJointSpec:
    if not isinstance(raw, Mapping):
        raise GeneratedGripperAssetError(f"{label} must be an object")
    return _validate_joint_spec(
        PrismaticJointSpec(
            origin_xyz=_require_vec3(raw.get("origin_xyz"), f"{label}.origin_xyz"),
            origin_rpy=_require_vec3(raw.get("origin_rpy"), f"{label}.origin_rpy"),
            axis_xyz=_require_vec3(raw.get("axis_xyz"), f"{label}.axis_xyz"),
        ),
        label,
    )


def _validate_joint_spec(spec: PrismaticJointSpec, label: str) -> PrismaticJointSpec:
    norm = math.sqrt(sum(value * value for value in spec.axis_xyz))
    if abs(norm - 1.0) > 1e-4:
        raise GeneratedGripperAssetError(f"{label}.axis_xyz must be unit length")
    return spec


def _load_params(path: Path, gripper_id: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, Mapping):
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {gripper_id!r} params_path must contain a JSON object: {path}"
        )
    return params


def _entry_path(entry: Mapping[str, Any], key: str, index: int | str, root_dir: Path) -> Path:
    return _resolve_path(_require_str(entry, key, index), root_dir)


def _mesh_path(entry: Mapping[str, Any], key: str, index: int | str, mesh_dir: Path) -> Path:
    return _resolve_path(_require_str(entry, key, index), mesh_dir)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise GeneratedGripperAssetError(f"Required generated gripper file does not exist: {path}")
    return path


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise GeneratedGripperAssetError(f"Required generated gripper directory does not exist: {path}")
    return path


def _require_mapping(entry: Mapping[str, Any], key: str, index: int | str) -> Mapping[str, Any]:
    value = entry.get(key)
    if not isinstance(value, Mapping):
        raise GeneratedGripperAssetError(f"Generated gripper entry {index!r} missing object field {key!r}")
    return value


def _require_str(entry: Mapping[str, Any], key: str, index: int | str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GeneratedGripperAssetError(f"Generated gripper entry {index!r} missing string field {key!r}")
    return value


def _require_bool(entry: Mapping[str, Any], key: str, index: int | str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise GeneratedGripperAssetError(f"Generated gripper entry {index!r} field {key!r} must be a bool")
    return value


def _require_int(entry: Mapping[str, Any], key: str, index: int | str) -> int:
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedGripperAssetError(f"Generated gripper entry {index!r} field {key!r} must be an int")
    return value


def _require_positive_float(entry: Mapping[str, Any], key: str, index: int | str) -> float:
    value = entry.get(key)
    if isinstance(value, bool):
        raise GeneratedGripperAssetError(f"Generated gripper entry {index!r} field {key!r} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {index!r} field {key!r} must be a number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {index!r} field {key!r} must be finite and > 0"
        )
    return parsed


def _require_str_pair(entry: Mapping[str, Any], key: str, index: int | str) -> tuple[str, str]:
    value = entry.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {index!r} field {key!r} must contain exactly two strings"
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise GeneratedGripperAssetError(
            f"Generated gripper entry {index!r} field {key!r} must contain exactly two strings"
        )
    return str(value[0]), str(value[1])


def _require_vec3(value: Any, label: str) -> tuple[float, float, float]:
    values = _require_float_sequence(value, label, 3)
    return values[0], values[1], values[2]


def _require_vec4(value: Any, label: str) -> tuple[float, float, float, float]:
    values = _require_float_sequence(value, label, 4)
    return values[0], values[1], values[2], values[3]


def _require_float_sequence(value: Any, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise GeneratedGripperAssetError(f"{label} must be a length-{length} list")
    parsed = []
    for item in value:
        if isinstance(item, bool):
            raise GeneratedGripperAssetError(f"{label} must contain only finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise GeneratedGripperAssetError(f"{label} must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise GeneratedGripperAssetError(f"{label} must contain only finite numbers")
        parsed.append(number)
    return tuple(parsed)


def _parse_vec_attr(element: ET.Element, attr: str, label: str) -> tuple[float, float, float]:
    value = element.get(attr)
    if value is None:
        raise GeneratedGripperAssetError(f"{label} is missing")
    parts = value.split()
    if len(parts) != 3:
        raise GeneratedGripperAssetError(f"{label} must contain three numbers")
    try:
        parsed = [float(part) for part in parts]
    except ValueError as exc:
        raise GeneratedGripperAssetError(f"{label} must contain three numbers") from exc
    return _require_vec3(parsed, label)

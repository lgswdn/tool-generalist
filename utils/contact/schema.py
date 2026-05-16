"""contact_pt_env_v1 schema validation and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.assets import assert_adjusted_decomposed_mesh_path
from utils.geometry.mesh_io import scaled_mesh_bbox


CONTACT_SCHEMA_VERSION = "contact_pt_env_v1"


class ContactSchemaError(ValueError):
    """Raised when a contact_pt_env_v1 artifact is missing required semantics."""


_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "generation_status",
    "config_name",
    "config_hash",
    "num_contacts",
    "object_id",
    "tool_id",
    "object_mesh_path",
    "tool_mesh_path",
    "object_scale",
    "tool_scale_xyz",
    "tool_head_area_aabb_norm",
    "object_bbox_center_M",
    "object_bbox_extent_M",
    "tool_bbox_center_M",
    "tool_bbox_extent_M",
    "object_point_sample_seed",
    "tool_point_sample_seed",
    "object_rotation_E",
    "object_bbox_center_E",
    "tool_translation_E",
    "tool_rotation_E",
    "contact_point_E",
    "object_mass",
    "tool_mass",
    "object_friction",
    "tool_friction",
    "ground_friction",
    "post_tool_delta_pose9d_E",
    "post_tool_achieved_delta_pose9d_E",
    "post_object_delta_pose9d_E",
    "stabilize_steps",
    "postcontact_steps",
)

_PER_CONTACT_SHAPES = {
    "object_rotation_E": (3, 3),
    "object_bbox_center_E": (3,),
    "tool_translation_E": (3,),
    "tool_rotation_E": (3, 3),
    "contact_point_E": (3,),
    "object_mass": (),
    "tool_mass": (),
    "object_friction": (),
    "tool_friction": (),
    "ground_friction": (),
    "post_tool_delta_pose9d_E": (9,),
    "post_tool_achieved_delta_pose9d_E": (9,),
    "post_object_delta_pose9d_E": (9,),
    "stabilize_steps": (),
    "postcontact_steps": (),
}


def _forbidden_object_frame_keys() -> set[str]:
    suffix = "O"
    return {
        f"tool_translation_{suffix}",
        f"tool_rotation_{suffix}",
        f"contact_point_{suffix}",
        f"post_tool_delta_pose9d_{suffix}",
        f"post_object_delta_pose9d_{suffix}",
    }


def load_and_validate_contact_pt(
    path: str | Path,
    *,
    allow_mock: bool = False,
    require_real_physics: bool = False,
    require_complete: bool = True,
    strict_mesh: bool = True,
) -> Mapping[str, Any]:
    """Load a torch-saved contact artifact and validate contact_pt_env_v1 semantics."""

    import torch

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ContactSchemaError(f"contact artifact does not exist: {artifact_path}")
    data = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(data, Mapping):
        raise ContactSchemaError(f"contact artifact must contain a mapping: {artifact_path}")
    validate_contact_dict(
        data,
        allow_mock=allow_mock,
        require_real_physics=require_real_physics,
        require_complete=require_complete,
        strict_mesh=strict_mesh,
    )
    return data


def validate_contact_dict(
    data: Mapping[str, Any],
    *,
    allow_mock: bool = False,
    require_real_physics: bool = False,
    require_complete: bool = True,
    strict_mesh: bool = True,
    rotation_orth_eps: float = 1e-4,
) -> None:
    import torch

    forbidden = sorted(set(data.keys()) & _forbidden_object_frame_keys())
    if forbidden:
        raise ContactSchemaError(f"contact artifact contains forbidden object-frame fields: {forbidden}")

    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in data]
    if missing:
        raise ContactSchemaError(f"contact_pt_env_v1 missing required fields: {missing}")
    if data["schema_version"] != CONTACT_SCHEMA_VERSION:
        raise ContactSchemaError(
            f"Unknown contact schema version: {data['schema_version']!r}"
        )

    status = str(data["generation_status"])
    if require_complete and status != "complete":
        raise ContactSchemaError(
            f"contact_pt_env_v1 must be complete before pretrain, got generation_status={status!r}"
        )
    if require_real_physics and not bool(data.get("is_real_physics", False)):
        raise ContactSchemaError("contact_pt_env_v1 is not a complete real-physics artifact")

    tool_id = str(data["tool_id"])
    try:
        assert_adjusted_decomposed_mesh_path(data["tool_mesh_path"], tool_id)
    except Exception as exc:
        raise ContactSchemaError(str(exc)) from exc

    n = int(data["num_contacts"])
    if n < 0:
        raise ContactSchemaError("num_contacts must be >= 0")
    if require_complete and n == 0:
        raise ContactSchemaError("complete contact_pt_env_v1 artifact must contain at least one contact")
    for key, suffix_shape in _PER_CONTACT_SHAPES.items():
        tensor = torch.as_tensor(data[key])
        expected = (n,) + suffix_shape
        if tuple(tensor.shape) != expected:
            raise ContactSchemaError(f"{key} must have shape {expected}, got {tuple(tensor.shape)}")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise ContactSchemaError(f"{key} contains non-finite values")

    _require_shape(data, "tool_scale_xyz", (3,))
    _require_shape(data, "tool_head_area_aabb_norm", (2, 3))
    _require_shape(data, "object_bbox_center_M", (3,))
    _require_shape(data, "object_bbox_extent_M", (3,))
    _require_shape(data, "tool_bbox_center_M", (3,))
    _require_shape(data, "tool_bbox_extent_M", (3,))
    _require_hex_hash(data, "config_hash")

    _validate_rotations(torch.as_tensor(data["object_rotation_E"], dtype=torch.float32), rotation_orth_eps, "object_rotation_E")
    _validate_rotations(torch.as_tensor(data["tool_rotation_E"], dtype=torch.float32), rotation_orth_eps, "tool_rotation_E")

    if strict_mesh:
        _validate_bbox_from_mesh(
            data["object_mesh_path"],
            float(data["object_scale"]),
            data["object_bbox_center_M"],
            data["object_bbox_extent_M"],
            "object",
        )
        _validate_bbox_from_mesh(
            data["tool_mesh_path"],
            torch.as_tensor(data["tool_scale_xyz"], dtype=torch.float32).tolist(),
            data["tool_bbox_center_M"],
            data["tool_bbox_extent_M"],
            "tool",
        )


def _require_shape(data: Mapping[str, Any], key: str, expected: tuple[int, ...]) -> None:
    import torch

    actual = tuple(torch.as_tensor(data[key]).shape)
    if actual != expected:
        raise ContactSchemaError(f"{key} must have shape {expected}, got {actual}")


def _require_hex_hash(data: Mapping[str, Any], key: str) -> None:
    value = str(data[key])
    if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise ContactSchemaError(f"{key} must be a 64-character hex hash")


def _validate_rotations(rot: Any, eps: float, key: str) -> None:
    import torch

    if rot.numel() == 0:
        return
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
    err = torch.matmul(rot.transpose(-1, -2), rot) - eye
    if float(err.abs().max().item()) > float(eps):
        raise ContactSchemaError(f"{key} rotation matrices must be orthogonal")


def _validate_bbox_from_mesh(
    mesh_path: str | Path,
    scale: float | Sequence[float],
    center: Any,
    extent: Any,
    label: str,
) -> None:
    import torch

    path = Path(mesh_path)
    if not path.exists():
        raise ContactSchemaError(f"{label}_mesh_path does not exist: {path}")
    actual_center, actual_extent = scaled_mesh_bbox(path, scale)
    expected_center = torch.as_tensor(center, dtype=torch.float64)
    expected_extent = torch.as_tensor(extent, dtype=torch.float64)
    if not torch.allclose(torch.as_tensor(actual_center), expected_center, atol=1e-5, rtol=1e-5):
        raise ContactSchemaError(f"{label}_bbox_center_M does not match scaled mesh bbox")
    if not torch.allclose(torch.as_tensor(actual_extent), expected_extent, atol=1e-5, rtol=1e-5):
        raise ContactSchemaError(f"{label}_bbox_extent_M does not match scaled mesh bbox")

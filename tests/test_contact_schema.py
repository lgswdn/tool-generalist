import copy

import pytest
import torch

from utils.contact.schema import ContactSchemaError, validate_contact_dict
from utils.geometry.mesh_io import scaled_mesh_bbox


MESH_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _write_mesh(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MESH_OBJ)


def _valid_payload(tmp_path, n=2):
    object_path = tmp_path / "objects" / "object_a.obj"
    tool_path = tmp_path / "meshdata" / "tool_a" / "coacd" / "decomposed.obj"
    _write_mesh(object_path)
    _write_mesh(tool_path)

    object_center, object_extent = scaled_mesh_bbox(object_path, 2.0)
    tool_center, tool_extent = scaled_mesh_bbox(tool_path, [0.1, 0.2, 0.3])
    eye = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)

    return {
        "schema_version": "contact_pt_env_v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "generator": "tests",
        "config_name": "unit",
        "config_hash": "b" * 64,
        "generation_status": "complete",
        "physics_runner": "isaac",
        "is_real_physics": True,
        "object_id": "object_a",
        "tool_id": "tool_a",
        "object_mesh_path": str(object_path),
        "tool_mesh_path": str(tool_path),
        "object_scale": 2.0,
        "tool_scale_xyz": torch.tensor([0.1, 0.2, 0.3]),
        "tool_head_area_aabb_norm": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "object_bbox_center_M": torch.tensor(object_center, dtype=torch.float32),
        "object_bbox_extent_M": torch.tensor(object_extent, dtype=torch.float32),
        "tool_bbox_center_M": torch.tensor(tool_center, dtype=torch.float32),
        "tool_bbox_extent_M": torch.tensor(tool_extent, dtype=torch.float32),
        "num_contacts": n,
        "object_point_sample_seed": 123,
        "tool_point_sample_seed": 123,
        "object_rotation_E": eye.clone(),
        "object_bbox_center_E": torch.zeros(n, 3),
        "tool_translation_E": torch.zeros(n, 3),
        "tool_rotation_E": eye.clone(),
        "contact_point_E": torch.zeros(n, 3),
        "object_mass": torch.full((n,), 0.1),
        "tool_mass": torch.full((n,), 0.1),
        "object_friction": torch.full((n,), 0.8),
        "tool_friction": torch.full((n,), 0.8),
        "ground_friction": torch.full((n,), 0.8),
        "stabilize_steps": torch.full((n,), 120),
        "post_tool_delta_pose9d_E": torch.zeros(n, 9),
        "post_tool_achieved_delta_pose9d_E": torch.zeros(n, 9),
        "post_object_delta_pose9d_E": torch.zeros(n, 9),
        "postcontact_steps": torch.full((n,), 120),
    }


def test_contact_pt_env_v1_validates_required_fields(tmp_path):
    validate_contact_dict(_valid_payload(tmp_path))


def test_unknown_schema_version_is_rejected(tmp_path):
    data = _valid_payload(tmp_path)
    data["schema_version"] = "contact_pt_v2"

    with pytest.raises(ContactSchemaError, match="Unknown contact schema version"):
        validate_contact_dict(data)


def test_rotation_must_be_orthogonal(tmp_path):
    data = _valid_payload(tmp_path)
    data["tool_rotation_E"] = data["tool_rotation_E"].clone()
    data["tool_rotation_E"][0, 0, 0] = 2.0

    with pytest.raises(ContactSchemaError, match="orthogonal"):
        validate_contact_dict(data)


def test_legacy_object_frame_fields_are_rejected(tmp_path):
    data = _valid_payload(tmp_path)
    data[f"tool_translation_{'O'}"] = torch.zeros(data["num_contacts"], 3)
    with pytest.raises(ContactSchemaError, match="object-frame"):
        validate_contact_dict(data)


def test_bbox_is_recomputed_from_scaled_mesh(tmp_path):
    data = _valid_payload(tmp_path)
    data["tool_bbox_extent_M"] = data["tool_bbox_extent_M"] + 1.0

    with pytest.raises(ContactSchemaError, match="tool_bbox_extent"):
        validate_contact_dict(data)


def test_object_mesh_tool_contract_allows_object_as_tool_mesh(tmp_path):
    data = _valid_payload(tmp_path)
    object_tool_path = tmp_path / "objects" / "tool_as_object.obj"
    _write_mesh(object_tool_path)
    center, extent = scaled_mesh_bbox(object_tool_path, [0.1, 0.2, 0.3])

    data["tool_id"] = "tool_as_object-0.100"
    data["tool_mesh_path"] = str(object_tool_path)
    data["tool_bbox_center_M"] = torch.tensor(center, dtype=torch.float32)
    data["tool_bbox_extent_M"] = torch.tensor(extent, dtype=torch.float32)

    with pytest.raises(ContactSchemaError, match="tool mesh must be"):
        validate_contact_dict(data)
    validate_contact_dict(data, tool_mesh_contract="object_mesh")


@pytest.mark.parametrize("field", ["object_bbox_center_M", "tool_bbox_center_M"])
def test_bbox_center_is_recomputed_from_scaled_mesh(tmp_path, field):
    data = _valid_payload(tmp_path)
    data[field] = data[field] + torch.tensor([1.0, 0.0, 0.0])

    with pytest.raises(ContactSchemaError, match=field):
        validate_contact_dict(data)


def test_optional_cache_cloud_own_bbox_center_does_not_need_to_be_zero(tmp_path):
    data = _valid_payload(tmp_path)
    data["object_points_O"] = torch.tensor([[10.0, 1.0, 0.0], [12.0, 3.0, 2.0]])
    data["tool_points_T"] = torch.tensor([[-5.0, 2.0, 1.0], [-3.0, 4.0, 3.0]])

    validate_contact_dict(data)


def test_strict_final_validation_rejects_mock_payload(tmp_path):
    data = _valid_payload(tmp_path)
    data["generation_status"] = "mock_complete"
    data["physics_runner"] = "mock"
    data["is_real_physics"] = False

    with pytest.raises(ContactSchemaError, match="must be complete"):
        validate_contact_dict(data, require_real_physics=True)

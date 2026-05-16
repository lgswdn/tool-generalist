import numpy as np

import torch

from utils.contact.isaac import IsaacSimAdapter, _MeshData, _metric_failure


class _FakeAttr:
    def __init__(self, prim):
        self.prim = prim

    def Set(self, value):
        self.prim.calls.append(("attr_set", value))


class _FakePhysicsMaterial:
    def __init__(self, prim):
        self.prim = prim

    def CreateStaticFrictionAttr(self):
        self.prim.calls.append(("static_friction_attr", None))
        return _FakeAttr(self.prim)

    def CreateDynamicFrictionAttr(self):
        self.prim.calls.append(("dynamic_friction_attr", None))
        return _FakeAttr(self.prim)


class _FakeMaterialAPI:
    @staticmethod
    def Apply(prim):
        prim.calls.append(("material_api_apply", None))
        return _FakePhysicsMaterial(prim)


class _FakeUsdPhysics:
    MaterialAPI = _FakeMaterialAPI


class _FakeMaterialPrim:
    def __init__(self, path):
        self.path = path
        self.calls = []


class _FakeShadeMaterial:
    def __init__(self, prim):
        self.prim = prim

    def GetPrim(self):
        return self.prim


class _FakeShadeMaterialFactory:
    @staticmethod
    def Define(stage, material_path):
        prim = _FakeMaterialPrim(material_path)
        stage.materials[material_path] = prim
        return _FakeShadeMaterial(prim)


class _FakeBinding:
    def __init__(self, prim):
        self.prim = prim

    def Bind(self, material, materialPurpose=None):
        self.prim.calls.append(("bind", material.GetPrim().path, materialPurpose))


class _FakeMaterialBindingAPI:
    @staticmethod
    def Apply(prim):
        prim.calls.append(("binding_api_apply", None))
        return _FakeBinding(prim)


class _FakeUsdShade:
    Material = _FakeShadeMaterialFactory
    MaterialBindingAPI = _FakeMaterialBindingAPI


class _FakePrim:
    def __init__(self):
        self.calls = []


class _FakeStage:
    def __init__(self):
        self.materials = {}


def test_apply_physics_material_uses_case_friction_for_body():
    adapter = IsaacSimAdapter(headless=True)
    adapter._modules = {"UsdPhysics": _FakeUsdPhysics, "UsdShade": _FakeUsdShade}
    stage = _FakeStage()
    prim = _FakePrim()

    debug = adapter._apply_physics_material(stage, prim, "/World/Object_PhysicsMaterial", 0.73)

    assert debug == {
        "material_path": "/World/Object_PhysicsMaterial",
        "static_friction": 0.73,
        "dynamic_friction": 0.73,
        "bound": True,
    }
    assert "/World/Object_PhysicsMaterial" in stage.materials
    material_prim = stage.materials["/World/Object_PhysicsMaterial"]
    assert material_prim.calls == [
        ("material_api_apply", None),
        ("static_friction_attr", None),
        ("attr_set", 0.73),
        ("dynamic_friction_attr", None),
        ("attr_set", 0.73),
    ]
    assert prim.calls == [
        ("binding_api_apply", None),
        ("bind", "/World/Object_PhysicsMaterial", "physics"),
    ]


def test_penetration_metrics_use_reported_separation_or_penetration():
    adapter = IsaacSimAdapter(headless=True)
    points_a = np.array([[0.0, 0.0, 0.0]])
    points_b = np.array([[0.0, 0.0, 0.001]])

    metrics = adapter._compute_penetration_metrics(
        [{"distance": -0.002, "penetration_depth": 0.001}],
        object_points_E=points_a,
        tool_points_E=points_b,
    )

    assert metrics["penetration_depth_max"] == 0.002
    assert metrics["penetration_metric_source"] == "contact_report_separation"
    assert metrics["nearest_vertex_distance"] == 0.001

    missing = adapter._compute_penetration_metrics(
        [{"position": [0.0, 0.0, 0.0]}],
        object_points_E=points_a,
        tool_points_E=points_b,
    )

    assert missing["penetration_depth_max"] is None
    assert missing["penetration_metric_source"] == "missing"


def test_run_candidate_primary_metrics_are_stabilize_metrics(monkeypatch, tmp_path):
    import utils.contact.isaac as adapter_module

    adapter = IsaacSimAdapter(headless=True)
    mesh = _MeshData(
        points=np.array(
            [
                [0.0, 0.0, 0.10],
                [0.01, 0.0, 0.10],
                [0.0, 0.01, 0.10],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        bbox_center=np.zeros(3, dtype=np.float64),
    )
    monkeypatch.setattr(adapter_module, "_load_centered_mesh", lambda _path, _scale: mesh)
    monkeypatch.setattr(adapter, "_ensure_app", lambda: None)
    monkeypatch.setattr(adapter, "_new_stage", lambda: object())
    monkeypatch.setattr(adapter, "_define_ground", lambda _stage, _friction: (object(), {}))

    class _Prim:
        def __init__(self, name):
            self.name = name

    def define_body(_stage, *, prim_path, **_kwargs):
        return _Prim("object" if "Object" in prim_path else "tool"), {}

    monkeypatch.setattr(adapter, "_define_mesh_body", define_body)
    monkeypatch.setattr(adapter, "_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "_set_matrix_xform", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "_write_debug_artifacts", lambda **_kwargs: (None, None))
    pose_calls = {"object": 0, "tool": 0}

    def get_pose(prim):
        pose_calls[prim.name] += 1
        if prim.name == "object":
            center = np.array([0.0, 0.0, 0.0]) if pose_calls[prim.name] == 1 else np.array([0.001, 0.0, 0.0])
        else:
            center = np.array([0.0, 0.0, 0.0]) if pose_calls[prim.name] == 1 else np.array([0.002, 0.0, 0.0])
        return np.eye(3, dtype=np.float64), center

    monkeypatch.setattr(adapter, "_get_matrix_pose", get_pose)
    records = iter(
        [
            [{"distance": 0.001, "penetration_depth": 0.0}],
            [{"distance": 0.004, "penetration_depth": 0.0}],
        ]
    )
    monkeypatch.setattr(adapter, "_read_contact_records", lambda _paths: next(records))
    monkeypatch.setattr(
        adapter,
        "_read_velocity_norms",
        lambda _prim, _path: {
            "linear_velocity_norm": 0.0,
            "angular_velocity_norm": 0.0,
            "velocity_metric_source": "unit",
        },
    )

    candidate = {
        "object_rotation_E": torch.eye(3),
        "object_bbox_center_E": torch.zeros(3),
        "tool_translation_E": torch.zeros(3),
        "tool_rotation_E": torch.eye(3),
        "contact_point_E": torch.zeros(3),
    }
    props = {
        "object_mass": torch.tensor(0.2),
        "tool_mass": torch.tensor(0.3),
        "object_friction": torch.tensor(0.6),
        "tool_friction": torch.tensor(0.7),
        "ground_friction": torch.tensor(0.8),
    }
    cfg = adapter_module.PhysicsRunConfig(
        penetration_eps=0.0003,
        t_stabilize=2,
        t_postcontact=2,
        object_mesh_path=str(tmp_path / "object.obj"),
        tool_mesh_path=str(tmp_path / "tool.obj"),
    )
    command = torch.tensor([0.002, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    result = adapter.run_candidate(
        candidate=candidate,
        physical_props=props,
        cfg=cfg,
        commanded_tool_delta_pose9d_E=command,
        candidate_index=0,
    )

    assert result.success is True
    assert result.penetration_depth_max == 0.0
    assert result.metrics["postcontact_penetration_depth_max"] == 0.0


def test_metric_failure_rejects_missing_and_velocity_exceeded():
    missing = _metric_failure(
        prefix="stabilize",
        penetration_depth_max=None,
        linear_velocity_norm=0.0,
        angular_velocity_norm=0.0,
        penetration_eps=0.0003,
        linear_velocity_eps=0.001,
        angular_velocity_eps=0.001,
    )
    assert missing == "stabilize_missing_metrics:penetration_depth_max"

    moving = _metric_failure(
        prefix="postcontact",
        penetration_depth_max=0.0,
        linear_velocity_norm=0.02,
        angular_velocity_norm=0.0,
        penetration_eps=0.0003,
        linear_velocity_eps=0.01,
        angular_velocity_eps=0.01,
    )
    assert moving == "postcontact_linear_velocity_exceeded"

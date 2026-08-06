from __future__ import annotations

import ast
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from configs.panda_experiment_common import (
    generated_gripper_diff_post_rl_cfg,
    onrobot_rg2_diff_post_rl_cfg,
    official_panda_diff_post_rl_cfg,
    robotiq_2f140_diff_post_rl_cfg,
    robotiq_3f_diff_post_rl_cfg,
)
from utils.assets import OneDofGripperAssetError, load_one_dof_gripper_manifest
from utils.config.paths import load_project_paths
from utils.experiment.effective_paths import materialize_runtime_paths_yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/grippers/robotiq_2f140.json"
RG2_MANIFEST = ROOT / "configs/grippers/onrobot_rg2.json"
THREE_FINGER_MANIFEST = ROOT / "configs/grippers/robotiq_3f.json"
URDF = (
    ROOT
    / "thirdparty/rpdiff/src/rpdiff/descriptions/franka_panda_table/panda_2f140.urdf"
)
ENV_TOOL = (
    ROOT
    / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
    / "isaaclab_nonprehensile/env_tool.py"
)
CONVERTER = ROOT / "scripts/convert_one_dof_gripper.py"
VISUALIZER = ROOT / "scripts/visualize_one_dof_gripper_random.py"


def _absolute_manifest_payload() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = payload["grippers"][0]
    asset_root = (MANIFEST.parent / entry["root_dir"]).resolve()
    entry["root_dir"] = str(asset_root)
    entry["urdf_path"] = str((asset_root / entry["urdf_path"]).resolve())
    entry["usd_path"] = str((asset_root / entry["usd_path"]).resolve())
    for part in entry["cloud_parts"]:
        if "mesh_path" in part:
            part["mesh_path"] = str((asset_root / part["mesh_path"]).resolve())
    return payload


def test_robotiq_manifest_is_one_policy_dof_and_has_complete_geometry():
    assets = load_one_dof_gripper_manifest(MANIFEST, require_usd=False)
    assert len(assets) == 1
    asset = assets[0]
    assert asset.gripper_id == "robotiq_2f140"
    assert asset.category == "robotiq_like"
    assert asset.control_adapter == "primary_joint_with_mimics"
    assert asset.actuated_joint_names == ("finger_joint",)
    assert len(asset.cloud_parts) == 12
    assert asset.cloud_parts[0].mesh_scale == (0.001, 0.001, 0.001)
    links, joints = asset.topology_signature
    assert len(links) == 22
    assert len(joints) == 21

    for mesh in ET.parse(asset.urdf_path).getroot().findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith("package://")
        assert (asset.urdf_path.parent / filename).is_file(), filename


@pytest.mark.parametrize(
    ("manifest", "gripper_id", "category", "adapter", "physical_joint_count", "cloud_count"),
    [
        (RG2_MANIFEST, "onrobot_rg2", "rg_like", "primary_joint_with_mimics", 1, 7),
        (THREE_FINGER_MANIFEST, "robotiq_3f", "three_finger", "joint_synergy", 9, 13),
    ],
)
def test_additional_official_gripper_manifests_are_complete(
    manifest, gripper_id, category, adapter, physical_joint_count, cloud_count
):
    asset = load_one_dof_gripper_manifest(manifest, require_usd=False)[0]
    assert asset.gripper_id == gripper_id
    assert asset.category == category
    assert asset.control_adapter == adapter
    assert len(asset.actuated_joint_names) == physical_joint_count
    assert len(asset.cloud_parts) == cloud_count
    assert asset.params["policy_control_dof"] == 1
    assert asset.actuator.effort_limit > 0.0
    for mesh in ET.parse(asset.urdf_path).getroot().findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith("package://")
        assert (asset.urdf_path.parent / filename).is_file(), filename


def test_all_official_grippers_reuse_exact_generated_gripper_franka_arm():
    def canonical(element):
        element = ET.fromstring(ET.tostring(element))
        for node in element.iter():
            if node.text is not None and not node.text.strip():
                node.text = None
            node.tail = None
        return ET.tostring(element)

    source = ET.parse(URDF).getroot()
    source_elements = {
        element.get("name"): canonical(element)
        for element in source
        if element.get("name", "").startswith("panda_link")
        or element.get("name", "").startswith("panda_joint")
    }
    for manifest in (RG2_MANIFEST, THREE_FINGER_MANIFEST):
        asset = load_one_dof_gripper_manifest(manifest, require_usd=False)[0]
        root = ET.parse(asset.urdf_path).getroot()
        actual = {
            element.get("name"): canonical(element)
            for element in root
            if element.get("name", "").startswith("panda_link")
            or element.get("name", "").startswith("panda_joint")
        }
        assert actual == source_elements


def test_three_finger_is_one_policy_synergy_with_fixed_scissor_mode():
    asset = load_one_dof_gripper_manifest(THREE_FINGER_MANIFEST, require_usd=False)[0]
    root = ET.parse(asset.urdf_path).getroot()
    assert root.find("joint[@name='palm_finger_1_joint']").attrib["type"] == "fixed"
    assert root.find("joint[@name='palm_finger_2_joint']").attrib["type"] == "fixed"
    assert asset.control_adapter == "joint_synergy"
    assert len(asset.open_joint_positions) == len(asset.closed_joint_positions) == 9
    for index in (0, 3, 6):
        assert asset.closed_joint_positions[index] == pytest.approx(1.155)
        assert asset.closed_joint_positions[index + 1] == pytest.approx(0.0)
        assert asset.closed_joint_positions[index + 2] == pytest.approx(-1.2217304764)


def test_three_finger_mount_maps_upstream_approach_axis_to_franka_forward():
    asset = load_one_dof_gripper_manifest(THREE_FINGER_MANIFEST, require_usd=False)[0]
    mount = ET.parse(asset.urdf_path).getroot().find("joint[@name='panda_hand_joint']/origin")
    roll, pitch, yaw = (float(value) for value in mount.attrib["rpy"].split())

    # URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll). The upstream 3F
    # fingers extend along palm-local +Y, which must become flange-local +Z.
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotated_local_y = (
        cy * sp * sr - sy * cr,
        sy * sp * sr + cy * cr,
        cp * sr,
    )
    assert rotated_local_y == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-9)


def test_robotiq_combined_urdf_matches_generated_gripper_arm_and_has_massive_pads():
    root = ET.parse(URDF).getroot()

    # Keep the arm identical to gripper/franka_template/isaac.urdf, which is
    # also the source template for every generated-gripper robot.
    expected_arm = {
        1: ((0.0, -0.04, -0.05), 2.7),
        2: ((0.0, -0.04, 0.06), 2.73),
        3: ((0.01, 0.01, -0.05), 2.04),
        4: ((-0.03, 0.03, 0.02), 2.08),
        5: ((0.0, 0.04, -0.12), 3.0),
        6: ((0.04, 0.0, 0.0), 1.3),
        7: ((0.0, 0.0, 0.08), 0.2),
    }
    assert root.find("link[@name='panda_link0']/inertial") is None
    assert root.find("link[@name='panda_link8']/inertial") is None
    for index, (expected_origin, expected_mass) in expected_arm.items():
        inertial = root.find(f"link[@name='panda_link{index}']/inertial")
        assert inertial is not None
        origin = tuple(float(value) for value in inertial.find("origin").attrib["xyz"].split())
        assert origin == pytest.approx(expected_origin)
        assert float(inertial.find("mass").attrib["value"]) == pytest.approx(expected_mass)
        inertia = inertial.find("inertia")
        assert all(float(inertia.attrib[key]) == pytest.approx(0.1) for key in ("ixx", "iyy", "izz"))

    joint5_limit = root.find("joint[@name='panda_joint5']/limit")
    assert joint5_limit is not None
    assert float(joint5_limit.attrib["lower"]) == pytest.approx(-2.8975)
    assert float(joint5_limit.attrib["upper"]) == pytest.approx(2.8975)
    joint8_origin = root.find("joint[@name='panda_joint8']/origin")
    assert tuple(float(value) for value in joint8_origin.attrib["rpy"].split()) == pytest.approx(
        (0.0, 0.0, -0.785398163397)
    )

    for index in range(8):
        visual_mesh = root.find(f"link[@name='panda_link{index}']/visual/geometry/mesh")
        collision_mesh = root.find(f"link[@name='panda_link{index}']/collision/geometry/mesh")
        assert visual_mesh.attrib["filename"].endswith(f"gripper/franka_template/meshes/visual/link{index}.dae")
        assert collision_mesh.attrib["filename"].endswith(f"meshes/panda/meshes/collision/link{index}.obj")

    for pad_name in ("left_inner_finger_pad", "right_inner_finger_pad"):
        inertial = root.find(f"link[@name='{pad_name}']/inertial")
        assert inertial is not None
        assert float(inertial.find("mass").attrib["value"]) > 0.0
        inertia = inertial.find("inertia")
        assert all(float(inertia.attrib[key]) > 0.0 for key in ("ixx", "iyy", "izz"))

    grasp_target = root.find("link[@name='panda_grasptarget']/inertial")
    assert float(grasp_target.find("mass").attrib["value"]) > 0.0


def test_primary_joint_adapter_rejects_unconstrained_secondary_joint(tmp_path):
    payload = _absolute_manifest_payload()
    urdf_text = URDF.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace(
        '<mimic joint="finger_joint" multiplier="-1" offset="0"/>', "", 1
    )
    broken_urdf = tmp_path / "broken.urdf"
    broken_urdf.write_text(urdf_text, encoding="utf-8")
    payload["grippers"][0]["urdf_path"] = str(broken_urdf)
    broken_manifest = tmp_path / "broken.json"
    broken_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OneDofGripperAssetError, match="must mimic primary joint"):
        load_one_dof_gripper_manifest(broken_manifest, require_usd=False)


def test_manifest_allows_only_one_official_model_per_category(tmp_path):
    payload = _absolute_manifest_payload()
    duplicate = json.loads(json.dumps(payload["grippers"][0]))
    duplicate["id"] = "second_robotiq_like"
    payload["grippers"].append(duplicate)
    path = tmp_path / "duplicate_category.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OneDofGripperAssetError, match="one official model per category"):
        load_one_dof_gripper_manifest(path, require_usd=False)


def test_robotiq_paths_and_rl_contract():
    paths = load_project_paths(ROOT / "configs/paths/robotiq_2f140.yaml")
    assert paths.get("one_dof_grippers.root") == URDF.parent
    assert paths.get("one_dof_grippers.manifest") == MANIFEST

    cfg = robotiq_2f140_diff_post_rl_cfg("robotiq_contract")
    cfg.validate()
    assert cfg.rl.env.robot_mode == "one_dof_gripper"
    assert cfg.rl.isaac_task_id == "one-dof-gripper-v0"
    assert cfg.rl.action_dim == 8
    assert cfg.rl.observation.robot_state_dim == 18
    assert cfg.rl.observation.tool_cloud_source == "gripper_cloud_cache_v1"

    converter = CONVERTER.read_text(encoding="utf-8")
    assert "fix_base=True" in converter
    assert "merge_fixed_joints=False" in converter
    assert "convert_mimic_joints_to_normal_joints=True" in converter
    assert "validate_one_dof_gripper_usd" in converter
    assert "_author_hard_mimic_constraints" in converter
    assert 'if asset.control_adapter == "primary_joint_with_mimics"' in converter
    assert "asset.usd_path.stat().st_mtime_ns >= asset.urdf_path.stat().st_mtime_ns" in converter
    assert 'naturalFrequency").Set(0.0)' in converter
    assert 'dampingRatio").Set(0.0)' in converter

    env_source = ENV_TOOL.read_text(encoding="utf-8")
    assert "def _build_gripper_robot_cfg(" in env_source
    assert 'mode_name="generated_gripper"' in env_source
    assert 'mode_name="one_dof_gripper"' in env_source
    assert "enabled_self_collisions = False" not in env_source
    assert "max_depenetration_velocity, 1.0" not in env_source
    assert "hand_actuator.effort_limit_sim = actuator_spec.effort_limit" in env_source
    assert "hand_actuator.stiffness = actuator_spec.stiffness" in env_source
    assert "hand_actuator.damping = actuator_spec.damping" in env_source
    assert "hand_actuator.armature = actuator_spec.armature" in env_source


@pytest.mark.parametrize(
    ("builder", "paths_name", "manifest"),
    [
        (onrobot_rg2_diff_post_rl_cfg, "onrobot_rg2.yaml", RG2_MANIFEST),
        (robotiq_3f_diff_post_rl_cfg, "robotiq_3f.yaml", THREE_FINGER_MANIFEST),
    ],
)
def test_additional_gripper_rl_configs(builder, paths_name, manifest):
    cfg = builder("additional_gripper_contract")
    cfg.validate()
    assert cfg.rl.env.robot_mode == "one_dof_gripper"
    assert cfg.rl.action_dim == 8
    assert cfg.rl.observation.robot_state_dim == 18
    paths = load_project_paths(ROOT / "configs/paths" / paths_name)
    assert paths.get("one_dof_grippers.manifest") == manifest
    assert paths.get("one_dof_grippers.root") == URDF.parent


def test_runtime_paths_keep_one_dof_manifest_absolute_when_moved_to_tmp(tmp_path):
    cfg = robotiq_2f140_diff_post_rl_cfg("robotiq_runtime_paths")
    source_paths = load_project_paths(ROOT / "configs/paths/robotiq_2f140.yaml")
    runtime_yaml = materialize_runtime_paths_yaml(
        cfg,
        source_paths,
        tmp_path / "paths.runtime.yaml",
    )
    assert runtime_yaml == tmp_path / "paths.runtime.yaml"
    runtime_paths = load_project_paths(runtime_yaml)
    assert runtime_paths.get("one_dof_grippers.manifest") == MANIFEST
    assert runtime_paths.get("one_dof_grippers.root") == URDF.parent


def test_panda_and_generated_gripper_action_configs_remain_distinct():
    for builder in (generated_gripper_diff_post_rl_cfg, official_panda_diff_post_rl_cfg):
        cfg = builder("regression")
        cfg.validate()
        assert cfg.rl.action_dim == 8
        assert cfg.rl.observation.robot_state_dim == 18

    module = ast.parse(ENV_TOOL.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in module.body if isinstance(node, ast.ClassDef)
    }

    def action_ctor(class_name: str) -> str:
        assignments = [
            node
            for node in classes[class_name].body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "gripper_action" for target in node.targets)
        ]
        assert len(assignments) == 1
        call = assignments[0].value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        return call.func.attr

    assert action_ctor("GeneratedGripperActionsCfg") == "SymmetricGeneratedGripperActionCfg"
    assert action_ctor("OneDofGripperActionsCfg") == "SemanticOneDofGripperActionCfg"


def test_generated_parallel_gripper_uses_real_franka_finger_speed():
    template_root = ET.parse(ROOT / "gripper/franka_template/isaac.urdf").getroot()
    for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
        limit = template_root.find(f"joint[@name='{joint_name}']/limit")
        assert limit is not None
        assert float(limit.attrib["velocity"]) == pytest.approx(0.05)

    env_source = ENV_TOOL.read_text(encoding="utf-8")
    assert "_GENERATED_PARALLEL_FINGER_VELOCITY_LIMIT_M_S = 0.05" in env_source
    assert 'robot_cfg.actuators["panda_hand"].velocity_limit_sim = (' in env_source


def test_one_dof_visualizer_uses_rl_signals_and_semantic_action_polarity():
    source = VISUALIZER.read_text(encoding="utf-8")
    assert 'DEFAULT_TASK = "one-dof-gripper-v0"' in source
    assert 'DEFAULT_CONFIG = "configs/experiments/robotiq_2f140_diff_post.py"' in source
    assert 'actions[:, 7] = _semantic_gripper_command(step)' in source
    assert 'if mode == "open":\n        return -1.0' in source
    assert 'if mode == "closed":\n        return 1.0' in source
    assert "get_head_area_pos_w" in source
    assert 'getattr(base_env, "_obs_tool_cloud_E", None)' in source
    assert "visualize_tool_pointcloud = not args_cli.no_debug_markers" in source
    assert "visualize_head_area_center = not args_cli.no_debug_markers" in source
    assert "max_joint_tracking_error=" in source
    assert "joint_sync_error=" in source

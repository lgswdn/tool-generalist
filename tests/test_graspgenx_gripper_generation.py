from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.generate_graspgenx_grippers import (
    FRANKA_REAL_CLOSURE_TIME_S,
    PARALLEL_MATCHED_OPEN_GAP_M,
    THREE_FINGER_CLOSE_ANGLE_RANGE_RAD,
    REVOLUTE_OPEN_ANGLE_RANGE_RAD,
    REVOLUTE_TRAVEL_ANGLE_RAD,
    _revolute_inward_extent,
    generate_family,
)
from utils.assets import load_one_dof_gripper_manifest
from utils.config.paths import load_project_paths


ROOT = Path(__file__).resolve().parents[1]
ARM_TEMPLATE = ROOT / "gripper/franka_template/isaac.urdf"


@pytest.fixture
def generated_families(tmp_path):
    revolute = generate_family(
        tmp_path,
        family="two_finger_revolute",
        count=16,
        seed=7,
        overwrite=False,
    )
    three_finger = generate_family(
        tmp_path,
        family="three_finger_high_dof",
        count=8,
        seed=7,
        overwrite=False,
    )
    return revolute, three_finger


def _joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"joint[@name='{name}']")
    assert joint is not None
    return joint


def _axis(root: ET.Element, name: str) -> tuple[float, float, float]:
    axis = _joint(root, name).find("axis")
    assert axis is not None
    return tuple(float(value) for value in axis.attrib["xyz"].split())


def _joint_origin_rpy(root: ET.Element, name: str) -> tuple[float, float, float]:
    origin = _joint(root, name).find("origin")
    assert origin is not None
    return tuple(float(value) for value in origin.attrib["rpy"].split())


def _normalized_arm_element(element: ET.Element) -> bytes:
    element = copy.deepcopy(element)
    for node in element.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None
    for mesh in element.findall(".//mesh"):
        filename = mesh.get("filename", "")
        marker = "meshes/"
        if marker in filename:
            mesh.set("filename", marker + filename.split(marker, 1)[1])
    return ET.tostring(element)


def test_generated_families_keep_exact_parallel_template_arm(generated_families):
    expected_root = ET.parse(ARM_TEMPLATE).getroot()
    expected = {
        element.get("name"): _normalized_arm_element(element)
        for element in expected_root
        if (
            element.tag == "link"
            and element.get("name", "").startswith("panda_link")
            and element.get("name", "")[10:].isdigit()
        )
        or (
            element.tag == "joint"
            and element.get("name", "").startswith("panda_joint")
            and element.get("name", "")[11:].isdigit()
        )
    }
    for manifest in generated_families:
        asset = load_one_dof_gripper_manifest(manifest, require_usd=False)[0]
        actual_root = ET.parse(asset.urdf_path).getroot()
        actual = {
            name: _normalized_arm_element(
                actual_root.find(f"{'link' if name.startswith('panda_link') else 'joint'}[@name='{name}']")
            )
            for name in expected
        }
        assert actual == expected


def test_two_finger_revolute_family_has_four_joint_one_dof_synergy(generated_families):
    manifest, _ = generated_families
    assets = load_one_dof_gripper_manifest(manifest, require_usd=False)
    assert len(assets) == 16
    assert {asset.category for asset in assets} == {"two_finger_revolute"}
    assert {asset.topology_family for asset in assets} == {
        "generated_two_finger_revolute_v1"
    }
    assert len({asset.topology_signature for asset in assets}) == 1
    assert {asset.open_joint_positions for asset in assets} == {(0.0,) * 4}
    assert {asset.closed_joint_positions for asset in assets} == {
        (REVOLUTE_TRAVEL_ANGLE_RAD,) * 4
    }
    assert {asset.actuator.velocity_limit for asset in assets} == {
        REVOLUTE_TRAVEL_ANGLE_RAD / FRANKA_REAL_CLOSURE_TIME_S
    }
    open_angles = {asset.params["open_angle_rad"] for asset in assets}
    closed_angles = {asset.params["closed_angle_rad"] for asset in assets}
    assert len(open_angles) > 1
    assert len(closed_angles) > 1

    modes = {asset.params["closure_mode"] for asset in assets}
    assert modes == {"parallel_tip", "pinch"}
    for asset in assets:
        root = ET.parse(asset.urdf_path).getroot()
        links = {link.get("name") for link in root.findall("link")}
        assert {
            "gripper_palm",
            "left_mid_link",
            "left_top_link",
            "right_mid_link",
            "right_top_link",
        }.issubset(links)
        assert _axis(root, "left_mid_joint") == pytest.approx((1.0, 0.0, 0.0))
        assert _axis(root, "right_mid_joint") == pytest.approx((-1.0, 0.0, 0.0))
        open_angle = asset.params["open_angle_rad"]
        closed_angle = asset.params["closed_angle_rad"]
        assert REVOLUTE_OPEN_ANGLE_RANGE_RAD[0] <= open_angle <= REVOLUTE_OPEN_ANGLE_RANGE_RAD[1]
        assert closed_angle == pytest.approx(open_angle + REVOLUTE_TRAVEL_ANGLE_RAD)
        assert _joint_origin_rpy(root, "left_mid_joint") == pytest.approx(
            (open_angle, 0.0, 0.0)
        )
        assert _joint_origin_rpy(root, "right_mid_joint") == pytest.approx(
            (-open_angle, 0.0, 0.0)
        )
        mode = asset.params["closure_mode"]
        if mode == "parallel_tip":
            assert asset.params["top_to_mid_motion_ratio"] == -1.0
            assert _axis(root, "left_top_joint") == pytest.approx((-1.0, 0.0, 0.0))
            assert _axis(root, "right_top_joint") == pytest.approx((1.0, 0.0, 0.0))
            assert _joint_origin_rpy(root, "left_top_joint") == pytest.approx(
                (-open_angle, 0.0, 0.0)
            )
            assert _joint_origin_rpy(root, "right_top_joint") == pytest.approx(
                (open_angle, 0.0, 0.0)
            )
            assert asset.params["closed_top_global_angle_rad"] == pytest.approx(0.0)
        else:
            assert asset.params["top_to_mid_motion_ratio"] == 1.0
            assert _axis(root, "left_top_joint") == pytest.approx((1.0, 0.0, 0.0))
            assert _axis(root, "right_top_joint") == pytest.approx((-1.0, 0.0, 0.0))
            assert _joint_origin_rpy(root, "left_top_joint") == pytest.approx(
                (0.0, 0.0, 0.0)
            )
            assert _joint_origin_rpy(root, "right_top_joint") == pytest.approx(
                (0.0, 0.0, 0.0)
            )
            assert asset.params["closed_top_global_angle_rad"] == pytest.approx(
                open_angle + 2.0 * REVOLUTE_TRAVEL_ANGLE_RAD
            )
        mid_length = asset.params["mid_size"][2]
        top_length = asset.params["top_size"][2]
        outer_size = asset.params["outer_size"]
        max_inward_extent = _revolute_inward_extent(
            length_scale=1.0,
            mid_length=mid_length,
            top_length=top_length,
            tip_length=asset.params["tip_length"],
            mid_y=asset.params["mid_size"][1],
            top_y=asset.params["top_size"][1],
            tip_width=asset.params["tip_width"],
            tip_shape=asset.params["tip_shape"],
            add_outer=asset.params["has_outer_finger"],
            outer_y=outer_size[1],
            outer_length_ratio=outer_size[2] / mid_length,
            mid_angle=closed_angle,
            top_angle=asset.params["closed_top_global_angle_rad"],
        )
        actual_surface_gap = (
            asset.params["finger_separation"]
            - 2.0 * max_inward_extent
        )
        assert actual_surface_gap == pytest.approx(asset.params["closed_surface_gap"])
        assert actual_surface_gap == pytest.approx(0.0, abs=1.0e-9)

        open_top_angle = 0.0 if mode == "parallel_tip" else open_angle
        open_inward_extent = _revolute_inward_extent(
            length_scale=1.0,
            mid_length=mid_length,
            top_length=top_length,
            tip_length=asset.params["tip_length"],
            mid_y=asset.params["mid_size"][1],
            top_y=asset.params["top_size"][1],
            tip_width=asset.params["tip_width"],
            tip_shape=asset.params["tip_shape"],
            add_outer=asset.params["has_outer_finger"],
            outer_y=outer_size[1],
            outer_length_ratio=outer_size[2] / mid_length,
            mid_angle=open_angle,
            top_angle=open_top_angle,
        )
        open_surface_gap = (
            asset.params["finger_separation"] - 2.0 * open_inward_extent
        )
        assert asset.params["open_surface_gap"] == pytest.approx(
            PARALLEL_MATCHED_OPEN_GAP_M
        )
        assert open_surface_gap == pytest.approx(
            PARALLEL_MATCHED_OPEN_GAP_M,
            abs=1.0e-9,
        )


def test_round_revolute_tips_are_present_in_rl_cloud(generated_families):
    manifest, _ = generated_families
    assets = load_one_dof_gripper_manifest(manifest, require_usd=False)
    round_assets = [asset for asset in assets if asset.params["tip_shape"] == "round"]
    assert round_assets
    for asset in round_assets:
        cylinders = [part for part in asset.cloud_parts if part.geometry_type == "cylinder"]
        assert len(cylinders) == 2
        assert all(part.cylinder_radius > 0.0 for part in cylinders)
        assert all(part.cylinder_length > 0.0 for part in cylinders)


def test_three_finger_family_has_nine_joint_one_dof_inward_synergy(generated_families):
    _, manifest = generated_families
    assets = load_one_dof_gripper_manifest(manifest, require_usd=False)
    assert len(assets) == 8
    assert {asset.category for asset in assets} == {"three_finger_high_dof"}
    assert {asset.topology_family for asset in assets} == {
        "generated_three_finger_9dof_v1"
    }
    assert len({asset.topology_signature for asset in assets}) == 1
    assert {asset.open_joint_positions for asset in assets} == {(0.0,) * 9}
    for asset in assets:
        assert asset.params["physical_dof"] == 9
        assert asset.params["graspgenx_joint_count_per_finger"] == 3
        closed_angle = asset.params["closed_angle_rad"]
        assert THREE_FINGER_CLOSE_ANGLE_RANGE_RAD[0] <= closed_angle <= THREE_FINGER_CLOSE_ANGLE_RANGE_RAD[1]
        assert asset.closed_joint_positions == pytest.approx((closed_angle,) * 9)
        assert asset.actuator.velocity_limit == pytest.approx(
            closed_angle / FRANKA_REAL_CLOSURE_TIME_S
        )
        assert asset.params["closed_centerline_gap"] == pytest.approx(0.0)
        assert asset.params["closed_tip_radial_clearance"] == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        assert len(asset.params["link_depths"]) == 3
        root = ET.parse(asset.urdf_path).getroot()
        assert len(
            [
                joint
                for joint in root.findall("joint")
                if joint.get("name", "").startswith("finger_")
            ]
        ) == 9
        for phi, lengths in zip(
            asset.params["finger_azimuths_rad"],
            asset.params["link_lengths"],
        ):
            # Every positive joint angle bends the initially +z chain toward
            # the palm axis, so closed radial reach is less than open reach.
            inward_travel = sum(
                length * math.sin((index + 1) * closed_angle)
                for index, length in enumerate(lengths)
            )
            assert inward_travel > 0.0
            finger_index = asset.params["finger_azimuths_rad"].index(phi) + 1
            assert asset.params["base_radii"][finger_index - 1] == pytest.approx(
                inward_travel
            )
            expected_axis = (math.sin(phi), -math.cos(phi), 0.0)
            assert _axis(root, f"finger_{finger_index}_joint_1") == pytest.approx(
                expected_axis
            )


def test_generated_manifest_records_reproducible_parameters(generated_families):
    for manifest in generated_families:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["seed"] == 7
        for entry in payload["grippers"]:
            params_file = manifest.parent / entry["root_dir"] / "params.json"
            assert json.loads(params_file.read_text(encoding="utf-8")) == entry["params"]


def test_rl_primitive_cloud_exactly_matches_generated_urdf_collisions(
    generated_families,
):
    def rounded(values):
        return tuple(round(float(value), 7) for value in values)

    for manifest in generated_families:
        for asset in load_one_dof_gripper_manifest(manifest, require_usd=False):
            root = ET.parse(asset.urdf_path).getroot()
            urdf_geometry = []
            for link in root.findall("link"):
                body_name = link.get("name")
                if body_name not in {part.body_name for part in asset.cloud_parts}:
                    continue
                for collision in link.findall("collision"):
                    origin = collision.find("origin")
                    translation = rounded(
                        (origin.get("xyz") if origin is not None else "0 0 0").split()
                    )
                    box = collision.find("geometry/box")
                    cylinder = collision.find("geometry/cylinder")
                    if box is not None:
                        urdf_geometry.append(
                            (
                                body_name,
                                "box",
                                rounded(box.get("size", "").split()),
                                translation,
                            )
                        )
                    elif cylinder is not None:
                        urdf_geometry.append(
                            (
                                body_name,
                                "cylinder",
                                rounded(
                                    (
                                        cylinder.get("radius"),
                                        cylinder.get("length"),
                                    )
                                ),
                                translation,
                            )
                        )

            cloud_geometry = []
            for part in asset.cloud_parts:
                dimensions = (
                    part.box_size
                    if part.geometry_type == "box"
                    else (part.cylinder_radius, part.cylinder_length)
                )
                cloud_geometry.append(
                    (
                        part.body_name,
                        part.geometry_type,
                        rounded(dimensions),
                        rounded(part.geometry_to_body.translation),
                    )
                )
            assert sorted(cloud_geometry) == sorted(urdf_geometry)


def test_workspace_generated_family_configs_are_ready_for_usd_conversion():
    from configs.experiments.generated_three_finger_high_dof_diff_post import (
        EXP_CFG as THREE_FINGER_CFG,
    )
    from configs.experiments.generated_two_finger_revolute_diff_post import (
        EXP_CFG as REVOLUTE_CFG,
    )

    expected = (
        (REVOLUTE_CFG, "two_finger_revolute", 4),
        (THREE_FINGER_CFG, "three_finger_high_dof", 9),
    )
    for cfg, family, joint_count in expected:
        cfg.validate()
        assert cfg.rl.env.robot_mode == "one_dof_gripper"
        paths = load_project_paths(cfg.paths_yaml)
        assets = load_one_dof_gripper_manifest(
            paths.get("one_dof_grippers.manifest"),
            expected_root=paths.get("one_dof_grippers.root"),
            require_usd=False,
        )
        assert len(assets) == 100
        assert {asset.category for asset in assets} == {family}
        assert {len(asset.actuated_joint_names) for asset in assets} == {
            joint_count
        }

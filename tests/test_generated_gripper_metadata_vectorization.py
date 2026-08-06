from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = (
    ROOT
    / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
    "isaaclab_nonprehensile/mdp/observations.py"
)


def _function(name: str) -> tuple[ast.FunctionDef, str]:
    source = OBSERVATIONS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return node, ast.get_source_segment(source, node) or ""


def test_generated_gripper_step_paths_use_cached_vectorized_metadata():
    fingertip_node, fingertip_source = _function(
        "get_generated_gripper_fingertip_center_pos_w"
    )
    cloud_node, cloud_source = _function(
        "get_generated_gripper_pointcloud_in_env_frame"
    )

    assert not any(isinstance(node, ast.For) for node in ast.walk(fingertip_node))
    assert not any(isinstance(node, ast.For) for node in ast.walk(cloud_node))
    assert "_generated_gripper_runtime_metadata(env)" in fingertip_source
    assert 'metadata["finger_body_ids"]' in fingertip_source
    assert "_generated_gripper_runtime_metadata(env)" in cloud_source
    assert "_get_generated_gripper_state_clouds_by_asset(env, metadata)" in cloud_source
    assert 'metadata["asset_indices"]' in cloud_source


def test_generated_gripper_static_metadata_and_groups_are_cached_once():
    _, metadata_source = _function("_generated_gripper_runtime_metadata")
    _, groups_source = _function("_generated_gripper_env_groups")
    _, clouds_source = _function("_get_generated_gripper_state_clouds_by_asset")

    assert "_generated_gripper_runtime_metadata_cache" in metadata_source
    assert "by_asset[asset_indices]" in metadata_source
    assert "_generated_gripper_env_groups_cache" in groups_source
    assert "_generated_gripper_state_clouds_by_asset_cache" in clouds_source
    assert "torch.stack(state_clouds, dim=0)" in clouds_source

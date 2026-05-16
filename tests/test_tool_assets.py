import json

import numpy as np
import pytest

from utils.contact.paths import load_contact_paths
from utils.assets import (
    ToolAssetContractError,
    assert_adjusted_decomposed_mesh_path,
    load_selected_tool_ids,
    load_tool_asset,
    resolve_tool_mesh_path,
    validate_tool_adjusted_entry,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _make_new_paths_yaml(tmp_path, tool_id="tool_a"):
    mesh_root = tmp_path / "meshdata_adjusted"
    mesh_path = mesh_root / tool_id / "coacd" / "decomposed.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("# test mesh\n")

    adjusted = tmp_path / "tools_adjusted.json"
    selected = tmp_path / "tools_selected.json"
    objects = tmp_path / "objects.json"
    obj_dir = tmp_path / "objects"
    obj_dir.mkdir()
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()

    _write_json(adjusted, [{"name": tool_id, "head_area": [[0.0, 0.1, 0.2], [1.0, 0.9, 1.0]]}])
    _write_json(selected, [tool_id])
    _write_json(objects, ["object_a"])

    yaml_path = tmp_path / "paths.yaml"
    yaml_path.write_text(
        f"""
objects:
  candidates_json: {objects}
  usd_dir: {usd_dir}
  obj_dir: {obj_dir}
tools:
  meshdata_adjusted_root: {mesh_root}
  tools_adjusted_json: {adjusted}
  tools_selected_json: {selected}
"""
    )
    return yaml_path, mesh_root, mesh_path


def test_tool_asset_contract_resolves_mesh_and_validates_adjusted_entry(tmp_path):
    yaml_path, mesh_root, mesh_path = _make_new_paths_yaml(tmp_path)
    paths = load_contact_paths(yaml_path)

    assert resolve_tool_mesh_path(mesh_root, "tool_a") == mesh_path
    assert load_selected_tool_ids(paths.tools_selected_json) == ["tool_a"]

    asset = load_tool_asset("tool_a", paths, scale_xyz=[0.1, 0.1, 0.1])

    assert asset.tool_id == "tool_a"
    assert asset.mesh_path == mesh_path
    np.testing.assert_allclose(asset.head_area_aabb_norm, [[0.0, 0.1, 0.2], [1.0, 0.9, 1.0]])


def test_tool_mesh_path_must_be_adjusted_decomposed_path(tmp_path):
    good = tmp_path / "meshdata_adjusted" / "tool_a" / "coacd" / "decomposed.obj"
    bad = tmp_path / "normalized_models" / "tool_a.obj"

    assert assert_adjusted_decomposed_mesh_path(good, "tool_a") == "tool_a"
    with pytest.raises(ToolAssetContractError, match="decomposed.obj"):
        assert_adjusted_decomposed_mesh_path(bad, "tool_a")


def test_tool_asset_requires_adjusted_mesh_path(tmp_path):
    yaml_path, _, _ = _make_new_paths_yaml(tmp_path)
    paths = load_contact_paths(yaml_path)
    missing_mesh = paths.meshdata_adjusted_root / "tool_a" / "coacd" / "decomposed.obj"
    missing_mesh.unlink()

    with pytest.raises(ToolAssetContractError, match="Tool mesh does not exist"):
        load_tool_asset("tool_a", paths, scale_xyz=[0.1, 0.1, 0.1])


def test_tool_adjusted_entry_rejects_bad_head_area():
    with pytest.raises(ToolAssetContractError, match="shape"):
        validate_tool_adjusted_entry({"name": "tool_a", "head_area": [0.0, 1.0]}, "tool_a")

    with pytest.raises(ToolAssetContractError, match="min"):
        validate_tool_adjusted_entry({"name": "tool_a", "head_area": [[0.5, 0, 0], [0.4, 1, 1]]}, "tool_a")

    with pytest.raises(ToolAssetContractError, match="tolerance"):
        validate_tool_adjusted_entry({"name": "tool_a", "head_area": [[0, 0, 0], [1.2, 1, 1]]}, "tool_a")


def test_legacy_paths_warn_and_map_to_new_contract(tmp_path):
    adjusted = tmp_path / "tools.json"
    selected = tmp_path / "selected.json"
    objects = tmp_path / "yes.json"
    obj_dir = tmp_path / "obj"
    usd_dir = tmp_path / "usd"
    mesh_root = tmp_path / "legacy_mesh_root"
    for path in (obj_dir, usd_dir, mesh_root):
        path.mkdir()
    _write_json(adjusted, [{"name": "tool_a", "head_area": [[0, 0, 0], [1, 1, 1]]}])
    _write_json(selected, ["tool_a"])
    _write_json(objects, ["object_a"])
    yaml_path = tmp_path / "paths.yaml"
    yaml_path.write_text(
        f"""
dgn:
  candidates_json: {objects}
  usd_dir: {usd_dir}
  obj_dir: {obj_dir}
tools:
  meshdata_adjusted_root: {mesh_root}
  tools_json: {adjusted}
  tools_selected_json: {selected}
"""
    )

    with pytest.warns(RuntimeWarning):
        paths = load_contact_paths(yaml_path)

    assert paths.meshdata_adjusted_root == mesh_root
    assert paths.tools_adjusted_json == adjusted

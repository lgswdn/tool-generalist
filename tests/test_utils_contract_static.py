from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _tree(rel_path: str) -> ast.Module:
    return ast.parse(_source(rel_path))


def _top_level_functions(rel_path: str) -> set[str]:
    return {node.name for node in _tree(rel_path).body if isinstance(node, ast.FunctionDef)}


def test_mesh_and_pose_helpers_live_in_utils_geometry():
    mesh_funcs = _top_level_functions("utils/geometry/mesh_io.py")
    pose_funcs = _top_level_functions("utils/geometry/pose.py")

    assert {"load_mesh_tensors", "sample_surface_points_torch", "sample_surface_points_numpy"} <= mesh_funcs
    assert {"pose9d_from_rt", "rotation_from_pose9d", "apply_pose9d_delta"} <= pose_funcs
    generator = _source("contact_generation/gen_contact.py")
    assert "from utils.geometry import" in generator
    assert "def load_mesh(" not in generator
    assert "def sample_surface_points(" not in generator


def test_tool_asset_contract_lives_in_utils_assets_with_contact_wrapper_only():
    assert (ROOT / "utils/assets/tool_assets.py").exists()
    assert (ROOT / "utils/assets/head_area.py").exists()

    wrapper = _source("contact_generation/tool_assets.py")
    assert "from utils.assets.tool_assets import" in wrapper
    assert "class ToolAsset" not in wrapper
    assert "def load_tool_asset" not in wrapper
    assert "from utils.assets import" in _source("tests/test_tool_assets.py")


def test_file_hash_and_json_helpers_are_centralized():
    assert (ROOT / "utils/io.py").exists()
    io_funcs = _top_level_functions("utils/io.py")
    assert {"read_json", "write_json", "hash_file", "hash_json", "canonical_json"} <= io_funcs
    assert "from utils.io import" in _source("utils/config/serialization.py")
    assert "from utils.io import" in _source("utils/config/hash.py")
    assert "from utils.io import" in _source("utils/experiment/validation.py")


def test_pretrain_top_level_only_keeps_canonical_implementation_modules():
    py_files = sorted(path.name for path in (ROOT / "pretrain").glob("*.py"))
    assert py_files == ["dataset.py", "model.py", "train.py"]

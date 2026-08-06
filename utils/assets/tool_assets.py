"""Tool asset path and head-area contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from utils.io import read_json


_KINEMATIC_TOOL_CLOUDS: dict[tuple[Path, str], tuple[float, Any]] = {}


class ToolAssetContractError(ValueError):
    """Raised when a tool asset entry violates the asset contract."""


class ToolPathConfig(Protocol):
    meshdata_adjusted_root: Path
    tools_adjusted_json: Path
    tools_selected_json: Path


@dataclass(frozen=True)
class ToolAsset:
    tool_id: str
    mesh_path: Path
    head_area_aabb_norm: np.ndarray
    scale_xyz: Sequence[float]
    adjusted_entry: Mapping[str, Any]


def infer_tool_id_from_mesh_path(mesh_path: str | Path) -> str:
    path = Path(mesh_path)
    if path.name == "decomposed.obj" and path.parent.name == "coacd":
        return path.parent.parent.name
    return path.stem


def assert_adjusted_decomposed_mesh_path(mesh_path: str | Path, tool_id: str | None = None) -> str:
    path = Path(mesh_path)
    if path.name != "decomposed.obj" or path.parent.name != "coacd" or not path.parent.parent.name:
        raise ToolAssetContractError(
            "tool mesh must be '<meshdata_adjusted_root>/<tool_id>/coacd/decomposed.obj'"
        )
    inferred = path.parent.parent.name
    if tool_id is not None and inferred != tool_id:
        raise ToolAssetContractError(
            f"tool mesh path id '{inferred}' does not match tool_id '{tool_id}'"
        )
    return inferred


def resolve_tool_mesh_path(meshdata_adjusted_root: str | Path, tool_id: str) -> Path:
    return Path(meshdata_adjusted_root) / tool_id / "coacd" / "decomposed.obj"


def _iter_tool_entries(data: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        if "tools" in data and isinstance(data["tools"], list):
            yield from data["tools"]
            return
        for key, value in data.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("name", key)
                yield entry
        return
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, Mapping):
                yield entry


def _find_tool_entry(path: str | Path, tool_id: str) -> Mapping[str, Any]:
    for entry in _iter_tool_entries(read_json(path)):
        if entry.get("name") == tool_id:
            return entry
    raise ToolAssetContractError(f"Tool '{tool_id}' was not found in {path}")


def load_tool_adjusted_entry(path: str | Path, tool_id: str) -> Mapping[str, Any]:
    """Load the adjusted tool manifest entry for ``tool_id``."""

    return _find_tool_entry(path, tool_id)


def load_selected_tool_ids(path: str | Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, Mapping):
        data = data.get("tools", data.get("selected", data))
    ids: list[str] = []
    if isinstance(data, Mapping):
        ids = [str(k) for k in data.keys()]
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, Mapping):
                value = item.get("name", item.get("tool_id", item.get("id")))
                if value is not None:
                    ids.append(str(value))
    if not ids:
        raise ToolAssetContractError(f"No selected tool ids found in {path}")
    return ids


def validate_tool_adjusted_entry(entry: Mapping[str, Any], tool_id: str) -> np.ndarray:
    if entry.get("name") != tool_id:
        raise ToolAssetContractError(
            f"tools_adjusted entry name '{entry.get('name')}' does not match tool_id '{tool_id}'"
        )
    if "head_area" not in entry:
        raise ToolAssetContractError(f"Tool '{tool_id}' is missing head_area")
    if (
        "source_generated_gripper_id" in entry
        and entry.get("proxy") != "exact_generated_mesh"
    ):
        raise ToolAssetContractError(
            f"Generated gripper '{tool_id}' must use proxy=exact_generated_mesh"
        )

    head_area = np.asarray(entry["head_area"], dtype=np.float64)
    if head_area.shape != (2, 3):
        raise ToolAssetContractError(
            f"Tool '{tool_id}' head_area must have shape (2, 3), got {head_area.shape}"
        )
    if not np.isfinite(head_area).all():
        raise ToolAssetContractError(f"Tool '{tool_id}' head_area contains non-finite values")
    if not (head_area[0] <= head_area[1]).all():
        raise ToolAssetContractError(f"Tool '{tool_id}' head_area min must be <= max")
    if (head_area < -0.02).any() or (head_area > 1.02).any():
        raise ToolAssetContractError(
            f"Tool '{tool_id}' head_area values must be within tolerance [-0.02, 1.02]"
        )
    return head_area


def load_tool_head_area(
    tools_json_path: str | Path,
    tool_mesh_path: str | Path,
    tool_id: Optional[str] = None,
) -> Tuple[list, list]:
    inferred_tool_id = assert_adjusted_decomposed_mesh_path(tool_mesh_path, tool_id)
    tool_stem = tool_id or inferred_tool_id
    path = Path(tools_json_path)
    if not path.exists():
        raise ToolAssetContractError(f"tools_adjusted.json does not exist: {tools_json_path}")
    entry = _find_tool_entry(path, tool_stem)
    validate_tool_adjusted_entry(entry, tool_stem)
    head_area = entry["head_area"]
    return head_area[0], head_area[1]


def load_tool_contact_tip_mesh(
    tools_json_path: str | Path,
    tool_mesh_path: str | Path,
    tool_id: Optional[str] = None,
) -> Path:
    """Return the strict fingertip surface mesh for contact anchor sampling."""

    inferred_tool_id = assert_adjusted_decomposed_mesh_path(
        tool_mesh_path, tool_id
    )
    tool_stem = tool_id or inferred_tool_id
    manifest_path = Path(tools_json_path)
    if not manifest_path.exists():
        raise ToolAssetContractError(
            f"tools_adjusted.json does not exist: {tools_json_path}"
        )
    entry = _find_tool_entry(manifest_path, tool_stem)
    validate_tool_adjusted_entry(entry, tool_stem)
    raw_path = entry.get("contact_tip_mesh")
    if not raw_path:
        raise ToolAssetContractError(
            f"Tool '{tool_stem}' is missing required contact_tip_mesh"
        )
    tip_path = Path(str(raw_path)).expanduser()
    if not tip_path.is_absolute():
        tip_path = manifest_path.parent / tip_path
    if not tip_path.is_file():
        raise ToolAssetContractError(
            f"Tool '{tool_stem}' contact_tip_mesh does not exist: {tip_path}"
        )
    return tip_path.resolve()


def load_tool_kinematic_cloud(
    tools_json_path: str | Path,
    tool_id: str,
) -> tuple[float, "torch.Tensor"]:
    """Load the canonical cached cloud at the tool asset's recorded openness."""

    cache_key = (Path(tools_json_path).expanduser().resolve(), tool_id)
    cached = _KINEMATIC_TOOL_CLOUDS.get(cache_key)
    if cached is not None:
        return cached
    entry = _find_tool_entry(tools_json_path, tool_id)
    raw_cache = entry.get("kinematic_cloud_cache")
    if raw_cache is None or "opening_fraction" not in entry:
        raise ToolAssetContractError(
            f"Tool {tool_id!r} lacks strict kinematic cache metadata"
        )
    cache_path = Path(str(raw_cache)).expanduser()
    if not cache_path.is_absolute():
        cache_path = Path(tools_json_path).parent / cache_path
    from utils.geometry.gripper_cloud_cache import (
        load_gripper_cloud_cache,
    )

    source_ids = [
        entry[key]
        for key in (
            "source_generated_gripper_id",
            "source_one_dof_gripper_id",
        )
        if key in entry
    ]
    if len(source_ids) != 1:
        raise ToolAssetContractError(
            f"Tool {tool_id!r} must identify exactly one cached gripper source"
        )
    source_id = str(source_ids[0])
    cache = load_gripper_cloud_cache(
        cache_path,
        expected_gripper_id=source_id,
        expected_source_manifest=str(entry["source_manifest"]),
    )
    opening = float(entry["opening_fraction"])
    result = (opening, cache.cloud_at_fraction(opening))
    _KINEMATIC_TOOL_CLOUDS[cache_key] = result
    return result


def load_tool_asset(
    tool_id: str,
    paths: ToolPathConfig,
    scale_xyz: Sequence[float] = (0.1, 0.1, 0.1),
    require_mesh: bool = True,
) -> ToolAsset:
    mesh_path = resolve_tool_mesh_path(paths.meshdata_adjusted_root, tool_id)
    assert_adjusted_decomposed_mesh_path(mesh_path, tool_id)
    if require_mesh and not mesh_path.exists():
        raise ToolAssetContractError(f"Tool mesh does not exist: {mesh_path}")

    entry = _find_tool_entry(paths.tools_adjusted_json, tool_id)
    head_area = validate_tool_adjusted_entry(entry, tool_id)
    return ToolAsset(
        tool_id=tool_id,
        mesh_path=mesh_path,
        head_area_aabb_norm=head_area,
        scale_xyz=list(scale_xyz),
        adjusted_entry=entry,
    )

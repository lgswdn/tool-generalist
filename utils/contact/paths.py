"""Path-schema loader for stage-1 contact generation assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ContactPaths:
    objects_candidates_json: Path
    objects_usd_dir: Path
    objects_obj_dir: Path
    meshdata_adjusted_root: Path
    objects_usd_root: Optional[Path]
    robots_usd_root: Optional[Path]
    tools_adjusted_json: Path
    tools_selected_json: Path
    franka_src_root: Optional[Path]
    source_yaml: Path


def _path(value: Any, base: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _optional_path(value: Any, base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    return _path(value, base)


def _get_path(
    cfg: Mapping[str, Any],
    base: Path,
    new_section: str,
    new_key: str,
    required: bool = True,
) -> Optional[Path]:
    section = cfg.get(new_section)
    if not isinstance(section, Mapping):
        section = {}
    if new_key in section:
        return _path(section[new_key], base)

    if required:
        raise KeyError(f"Missing required paths.yaml key '{new_section}.{new_key}'")
    return None


def load_contact_paths(
    yaml_path: str | Path = "configs/paths/default.yaml",
) -> ContactPaths:
    """Load the strict contact-generation path schema."""

    import yaml

    source = Path(yaml_path).resolve()
    with source.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"paths.yaml must contain a mapping: {source}")
    base = source.parent

    tools = cfg.get("tools")
    if not isinstance(tools, Mapping):
        tools = {}

    objects_candidates_json = _get_path(cfg, base, "objects", "candidates_json")
    objects_usd_dir = _get_path(cfg, base, "objects", "usd_dir")
    objects_obj_dir = _get_path(cfg, base, "objects", "obj_dir")
    meshdata_adjusted_root = _get_path(cfg, base, "tools", "meshdata_adjusted_root")
    tools_adjusted_json = _get_path(cfg, base, "tools", "tools_adjusted_json")
    tools_selected_json = _get_path(cfg, base, "tools", "tools_selected_json")

    robots_usd_root = None
    if "robots_usd_root" in tools:
        robots_usd_root = _path(tools["robots_usd_root"], base)

    return ContactPaths(
        objects_candidates_json=objects_candidates_json,
        objects_usd_dir=objects_usd_dir,
        objects_obj_dir=objects_obj_dir,
        meshdata_adjusted_root=meshdata_adjusted_root,
        objects_usd_root=_optional_path(tools.get("objects_usd_root"), base),
        robots_usd_root=robots_usd_root,
        tools_adjusted_json=tools_adjusted_json,
        tools_selected_json=tools_selected_json,
        franka_src_root=_optional_path(tools.get("franka_src_root"), base),
        source_yaml=source,
    )

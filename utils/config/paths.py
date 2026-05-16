"""Machine-local path config loader for the new experiment framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


class PathsConfigError(ValueError):
    """Raised when paths.yaml is malformed or missing required values."""


@dataclass(frozen=True)
class ProjectPaths:
    source_yaml: Path
    raw: Mapping[str, Any]
    values: Mapping[str, Optional[Path]]

    def get(self, key: str) -> Optional[Path]:
        return self.values.get(key)


PATH_ALIASES: dict[str, tuple[str, ...]] = {
    "objects.candidates_json": ("dgn.candidates_json",),
    "objects.usd_dir": ("dgn.usd_dir",),
    "objects.obj_dir": ("dgn.obj_dir",),
    "tools.meshdata_adjusted_root": (),
    "tools.tools_adjusted_json": ("tools.tools_json",),
    "tools.tools_selected_json": (),
    "tools.objects_usd_root": (),
    "tools.robots_usd_root": ("tools.robots_usd_dir",),
    "tools.franka_src_root": (),
}


def load_project_paths(yaml_path: str | Path) -> ProjectPaths:
    source = Path(yaml_path).expanduser().resolve()
    if not source.exists():
        raise PathsConfigError(f"paths.yaml not found: {source}")

    import yaml

    with source.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise PathsConfigError(f"paths.yaml must contain a mapping: {source}")

    base = source.parent
    values: dict[str, Optional[Path]] = {}
    for canonical_key, aliases in PATH_ALIASES.items():
        value = _lookup(raw, canonical_key)
        if value is None:
            for alias in aliases:
                value = _lookup(raw, alias)
                if value is not None:
                    break
        values[canonical_key] = _optional_path(value, base)

    return ProjectPaths(source_yaml=source, raw=raw, values=values)


def require_path(paths: ProjectPaths, key: str, *, must_exist: bool = True) -> Path:
    path = paths.get(key)
    if path is None:
        raise PathsConfigError(f"Missing required paths.yaml key '{key}'")
    if must_exist and not path.exists():
        raise PathsConfigError(f"Path for '{key}' does not exist: {path}")
    return path


def _lookup(raw: Mapping[str, Any], dotted_key: str) -> Any:
    cursor: Any = raw
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _optional_path(value: Any, base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()

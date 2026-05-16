"""Experiment-level path overrides.

``paths.yaml`` stores machine-local defaults.  Some manifest choices are still
experiment semantics, so ``ExpCfg`` may override them before stages run.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths


def apply_experiment_path_overrides(cfg: ExpCfg, paths: ProjectPaths) -> ProjectPaths:
    overrides = experiment_path_overrides(cfg, paths.source_yaml.parent)
    if not overrides:
        return paths
    values = dict(paths.values)
    values.update(overrides)
    return ProjectPaths(source_yaml=paths.source_yaml, raw=paths.raw, values=values)


def materialize_runtime_paths_yaml(
    cfg: ExpCfg,
    paths: ProjectPaths,
    output_path: str | Path,
) -> Path:
    """Write a runtime paths.yaml when ExpCfg overrides path-backed manifests."""

    overrides = experiment_path_overrides(cfg, paths.source_yaml.parent)
    if not overrides:
        return paths.source_yaml

    raw = deepcopy(dict(paths.raw))
    for key, value in overrides.items():
        section, field = key.split(".", 1)
        _set_nested(raw, section, field, str(value))
        if key == "objects.candidates_json":
            _set_nested(raw, "dgn", "candidates_json", str(value))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return path


def experiment_path_overrides(cfg: ExpCfg, base_dir: str | Path) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    tools_selected = cfg.general.tools_selected_json or cfg.general.tools_manifest
    if tools_selected:
        overrides["tools.tools_selected_json"] = _path(tools_selected, base_dir)
    if cfg.general.objects_manifest:
        overrides["objects.candidates_json"] = _path(cfg.general.objects_manifest, base_dir)
    return overrides


def _path(value: str, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(base_dir) / path).resolve()


def _set_nested(raw: dict[str, Any], section: str, field: str, value: str) -> None:
    current = raw.setdefault(section, {})
    if not isinstance(current, dict):
        raw[section] = {}
        current = raw[section]
    current[field] = value

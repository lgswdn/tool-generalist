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


def apply_experiment_path_overrides(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    stage: str = "rl",
) -> ProjectPaths:
    overrides = experiment_path_overrides(cfg, paths.source_yaml.parent, stage=stage)
    if not overrides:
        return paths
    values = dict(paths.values)
    values.update(overrides)
    return ProjectPaths(source_yaml=paths.source_yaml, raw=paths.raw, values=values)


def materialize_runtime_paths_yaml(
    cfg: ExpCfg,
    paths: ProjectPaths,
    output_path: str | Path,
    *,
    extra_overrides: dict[str, str | Path] | None = None,
    stage: str = "rl",
) -> Path:
    """Write a runtime paths.yaml when ExpCfg overrides path-backed manifests."""

    overrides = experiment_path_overrides(cfg, paths.source_yaml.parent, stage=stage)
    if extra_overrides:
        overrides.update({key: Path(value) for key, value in extra_overrides.items()})
    if not overrides:
        return paths.source_yaml

    raw = deepcopy(dict(paths.raw))
    # The generated YAML lives under the artifact/runtime directory, not next
    # to the source paths.yaml. Preserve the already-resolved meaning of every
    # relative source entry before writing it at the new location.
    for key, value in paths.values.items():
        if value is None:
            continue
        section, field = key.split(".", 1)
        _set_nested(raw, section, field, str(value))
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


def experiment_path_overrides(
    cfg: ExpCfg,
    base_dir: str | Path,
    *,
    stage: str = "rl",
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    tools_selected = cfg.general.tools_selected_json or cfg.general.tools_manifest
    if tools_selected:
        overrides["tools.tools_selected_json"] = _path(tools_selected, base_dir)
    object_manifest = object_manifest_for_stage(cfg, stage)
    if object_manifest:
        overrides["objects.candidates_json"] = _path(object_manifest, base_dir)
    return overrides


def object_manifest_for_stage(cfg: ExpCfg, stage: str) -> str | None:
    stage = str(stage)
    if stage in {"contact", "contact_gen", "pretrain"}:
        return (
            cfg.general.contact_objects_manifest
            or cfg.general.objects_manifest
            or cfg.general.rl_objects_manifest
        )
    if stage in {"rl", "eval", "record", "runtime"}:
        return (
            cfg.general.rl_objects_manifest
            or cfg.general.objects_manifest
            or cfg.general.contact_objects_manifest
        )
    raise ValueError(f"Unknown path override stage: {stage!r}")


def _path(value: str, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(base_dir) / path).resolve()


def _set_nested(raw: dict[str, Any], section: str, field: str, value: str) -> None:
    current = raw.setdefault(section, {})
    if not isinstance(current, dict):
        raw[section] = {}
        current = raw[section]
    current[field] = value

"""Load top-level experiment configs from Python modules or files."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.validate import validate_exp_cfg


class ConfigLoadError(ValueError):
    """Raised when a config module cannot provide an ``ExpCfg`` instance."""


def load_exp_cfg(config: str | Path) -> ExpCfg:
    module = load_config_module(config)
    cfg = _find_exp_cfg(module)
    try:
        return validate_exp_cfg(cfg)
    except ValueError as exc:
        raise ConfigLoadError(str(exc)) from exc


def load_config_module(config: str | Path) -> ModuleType:
    spec = str(config)
    path = Path(spec)
    if path.suffix == ".py" or path.exists():
        return _load_module_from_path(path)
    return importlib.import_module(spec)


def _load_module_from_path(path: Path) -> ModuleType:
    source = path.expanduser().resolve()
    if not source.exists():
        raise ConfigLoadError(f"Config file not found: {source}")
    module_name = f"_experiment_cfg_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ConfigLoadError(f"Could not load config file: {source}")

    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(source.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _find_exp_cfg(module: ModuleType) -> Any:
    if hasattr(module, "EXP_CFG"):
        return getattr(module, "EXP_CFG")
    if hasattr(module, "exp_cfg"):
        value = getattr(module, "exp_cfg")
        return value() if callable(value) else value
    if hasattr(module, "make_exp_cfg"):
        return getattr(module, "make_exp_cfg")()
    raise ConfigLoadError(
        f"{module.__name__} must define EXP_CFG, exp_cfg, or make_exp_cfg()"
    )

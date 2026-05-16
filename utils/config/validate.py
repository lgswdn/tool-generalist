"""Lightweight validation entrypoint for experiment configs."""

from __future__ import annotations

from configs.config_exp import ConfigValidationError, ExpCfg


def validate_exp_cfg(cfg: ExpCfg) -> ExpCfg:
    if not isinstance(cfg, ExpCfg):
        raise ConfigValidationError(
            f"Expected configs.config_exp.ExpCfg, got {type(cfg).__name__}"
        )
    cfg.validate()
    return cfg

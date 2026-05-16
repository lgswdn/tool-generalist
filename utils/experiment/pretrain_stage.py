"""Pretrain-stage entrypoint for the experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths


def run_pretrain_stage(
    cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> Any:
    """Run encoder pretraining.

    The torch-backed implementation is imported lazily so dry-run planning stays
    light and does not initialize model dependencies.
    """

    from pretrain.train import run_pretrain

    return run_pretrain(cfg, paths, artifact_dir)

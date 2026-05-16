"""Contact-stage entrypoint for the experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths


def run_contact_stage(
    cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> Any:
    """Run contact generation using the canonical batch API.

    The heavy generator stack is imported inside ``run_contact_generation``.
    """

    from contact_generation.batch_generate import run_contact_generation

    return run_contact_generation(cfg, paths, artifact_dir)

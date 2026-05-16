"""Config-only CLI helpers for legacy RL wrappers.

Business experiment parameters must live in ``ExpCfg``/``RLCfg``.  These helpers
only locate a config for internal static tooling; user runs go through
``run_experiment.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Python module/path exposing ExpCfg")
    return parser


def parse_config_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-only RL wrapper.")
    add_config_args(parser)
    return parser.parse_args(argv)


def resolve_artifact_dir(config_name: str, artifact_dir: str | None, default_root: str) -> Path:
    if artifact_dir:
        return Path(artifact_dir)
    return Path(default_root) / config_name / "rl"

"""Config-only RL playback wrapper."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths
from utils.io import write_json

from scripts.train import build_rl_runtime_spec


def run_rl_playback(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    spec = build_rl_runtime_spec(exp_cfg, paths, artifact_dir, mode="play")
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    write_json(artifact_path / "rl_play_runtime_spec.json", asdict(spec))
    return asdict(spec)


def main(argv: list[str] | None = None) -> int:
    raise SystemExit("Use run_experiment.py --config <experiment.py> [--mode run|plan].")


if __name__ == "__main__":
    raise SystemExit(main())

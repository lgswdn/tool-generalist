"""RL-stage entrypoint for the experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths


def run_rl_stage(
    cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
    *,
    resolved_encoder_checkpoint: str | None = None,
    runtime_objects_manifest: str | Path | None = None,
    runtime_num_gpus: int | None = None,
    runtime_num_envs: int | None = None,
    runtime_rl_resume_checkpoint: str | Path | None = None,
    runtime_print_fine_grained_timing: bool = False,
) -> Any:
    """Prepare or run RL from the canonical training API.

    The Isaac/rsl_rl stack is imported only inside the script runtime path, so
    planning and static tests stay lightweight.
    """

    from scripts.train import run_rl_training

    return run_rl_training(
        cfg,
        paths,
        artifact_dir,
        encoder_checkpoint_override=resolved_encoder_checkpoint,
        runtime_objects_manifest=runtime_objects_manifest,
        runtime_num_gpus=runtime_num_gpus,
        runtime_num_envs=runtime_num_envs,
        runtime_rl_resume_checkpoint=runtime_rl_resume_checkpoint,
        runtime_print_fine_grained_timing=runtime_print_fine_grained_timing,
        launch=True,
    )

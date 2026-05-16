"""Experiment planning and manifest materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from configs.config_exp import ExpCfg
from utils.artifacts import (
    ArtifactManifest,
    artifact_dir,
    contact_artifact_name,
    encoder_artifact_name,
    experiment_artifact_name,
    manifest_path_for,
    rl_artifact_name,
    write_manifest,
)
from utils.config import config_hash, load_project_paths, to_plain_data
from utils.config.paths import ProjectPaths
from utils.experiment.stages import all_stages
from utils.experiment.runtime import git_metadata, runtime_metadata, utc_timestamp
from utils.experiment.validation import validate_for_plan


@dataclass(frozen=True)
class StagePlan:
    stage: str
    enabled: bool
    artifact_type: str
    artifact_name: str
    artifact_dir: Path
    manifest_path: Path
    status: str = "planned"
    entrypoint: Optional[str] = None


@dataclass(frozen=True)
class ExperimentPlan:
    exp_cfg: ExpCfg
    exp_cfg_hash: str
    timestamp: str
    experiment: StagePlan
    stages: tuple[StagePlan, ...]


def build_experiment_plan(
    cfg: ExpCfg,
    *,
    timestamp: Optional[str] = None,
) -> ExperimentPlan:
    timestamp = timestamp or utc_timestamp()
    artifact_root = Path(cfg.general.artifact_root).expanduser()
    exp_hash = config_hash(_exp_hash_payload(cfg))
    experiment_name = experiment_artifact_name(cfg)
    experiment_dir = artifact_dir(artifact_root, experiment_name)
    experiment_plan = StagePlan(
        stage="experiment",
        enabled=True,
        artifact_type="experiment",
        artifact_name=experiment_name,
        artifact_dir=experiment_dir,
        manifest_path=manifest_path_for(experiment_dir),
    )

    stages: list[StagePlan] = []
    for stage in all_stages(cfg):
        status = "planned" if stage.enabled else "skipped"
        if stage.name == "contact_gen":
            name = contact_artifact_name(cfg)
        elif stage.name == "pretrain":
            name = encoder_artifact_name(cfg)
        elif stage.name == "rl":
            name = rl_artifact_name(cfg, timestamp)
        else:
            name = stage.name
        directory = artifact_dir(artifact_root, name)
        stages.append(
            StagePlan(
                stage=stage.name,
                enabled=stage.enabled,
                artifact_type=stage.artifact_type,
                artifact_name=name,
                artifact_dir=directory,
                manifest_path=manifest_path_for(directory),
                status=status,
                entrypoint=stage.entrypoint,
            )
        )

    return ExperimentPlan(
        exp_cfg=cfg,
        exp_cfg_hash=exp_hash,
        timestamp=timestamp,
        experiment=experiment_plan,
        stages=tuple(stages),
    )


def plan_from_config(
    cfg: ExpCfg,
    *,
    paths_yaml: str | Path | None = None,
    strict_paths: bool = True,
    timestamp: Optional[str] = None,
    cuda_visible_devices: str | None = None,
) -> tuple[ExperimentPlan, ProjectPaths]:
    selected_paths_yaml = paths_yaml or _effective_paths_yaml(cfg)
    paths = load_project_paths(selected_paths_yaml)
    validate_for_plan(
        cfg,
        paths,
        cuda_visible_devices=cuda_visible_devices,
    )
    return build_experiment_plan(cfg, timestamp=timestamp), paths


def materialize_plan(
    plan: ExperimentPlan,
    *,
    paths: ProjectPaths,
    cwd: str | Path,
    argv: list[str],
    write_manifests: bool = True,
) -> list[Path]:
    metadata = runtime_metadata(cwd=cwd, argv=argv)
    metadata.update(git_metadata(cwd))
    config_dump = to_plain_data(plan.exp_cfg)
    source_paths = _source_paths(paths)
    written: list[Path] = []

    all_plans = (plan.experiment, *plan.stages)
    for stage in all_plans:
        stage.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not write_manifests:
            continue
        manifest = ArtifactManifest(
            artifact_type=stage.artifact_type,
            artifact_name=stage.artifact_name,
            exp_cfg_name=plan.exp_cfg.name,
            config_hash=plan.exp_cfg_hash,
            status=stage.status,
            git_commit=str(metadata.get("git_commit", "unknown")),
            git_dirty=bool(metadata.get("git_dirty", False)),
            created_at=plan.timestamp,
            source_paths=source_paths,
            metrics={
                "stage": stage.stage,
                "enabled": stage.enabled,
                "executed": False,
                "entrypoint": stage.entrypoint,
            },
            config_dump=config_dump,
            runtime=metadata,
        )
        written.append(write_manifest(stage.artifact_dir, manifest))
    return written


def iter_stage_lines(plan: ExperimentPlan) -> Iterable[str]:
    yield _format_stage(plan.experiment)
    for stage in plan.stages:
        yield _format_stage(stage)


def _format_stage(stage: StagePlan) -> str:
    return (
        f"{stage.stage}: status={stage.status} "
        f"artifact={stage.artifact_dir} manifest={stage.manifest_path}"
    )


def _exp_hash_payload(cfg: ExpCfg) -> dict[str, Any]:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    return payload


def _source_paths(paths: ProjectPaths) -> dict[str, str]:
    values: dict[str, str] = {"paths_yaml": str(paths.source_yaml)}
    for key, path in paths.values.items():
        if path is not None:
            values[key] = str(path)
    return values


def _effective_paths_yaml(cfg: ExpCfg) -> str:
    return cfg.paths_yaml

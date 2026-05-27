"""Resolve experiment configs into local artifact locations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from configs.config_exp import ExpCfg
from utils.artifacts.naming import (
    artifact_dir,
    contact_artifact_name,
    encoder_artifact_name,
    experiment_artifact_name,
    rl_artifact_name,
)
from utils.artifacts.paths import artifact_root, manifest_path
from utils.config.hash import config_hash
from utils.config.hash_payloads import (
    normalize_artifact_root as _normalize_artifact_root,
    semantic_contact_general_payload as _semantic_contact_general_payload,
    semantic_contact_gen_payload as _semantic_contact_gen_payload,
    semantic_exp_payload as _semantic_exp_payload,
    semantic_pretrain_model_payload as _semantic_pretrain_model_payload,
    semantic_pretrain_payload as _semantic_pretrain_payload,
    strip_contact_runtime_fields as _strip_contact_runtime_fields,
    strip_default_condition_normalization as _strip_default_condition_normalization,
    strip_default_sdf_relative_loss as _strip_default_sdf_relative_loss,
    strip_pretrain_runtime_fields as _strip_pretrain_runtime_fields,
)
from utils.experiment.stages import StageSpec, all_stages


@dataclass(frozen=True)
class ArtifactRef:
    stage: str
    artifact_type: str
    artifact_name: str
    directory: Path
    manifest_path: Path
    config_hash: str
    enabled: bool = True
    requested: bool = True
    required: bool = True
    dependency_reason: str | None = None
    action: str = "planned"
    status: str = "planned"
    entrypoint: str | None = None


@dataclass(frozen=True)
class ResolvedArtifacts:
    experiment: ArtifactRef
    stages: tuple[ArtifactRef, ...]


def resolve_artifacts(cfg: ExpCfg, *, timestamp: str | None = None) -> ResolvedArtifacts:
    timestamp = timestamp or _utc_timestamp()
    root = artifact_root(cfg)
    experiment_name = experiment_artifact_name(cfg)
    exp_dir = artifact_dir(root, experiment_name)
    experiment = ArtifactRef(
        stage="experiment",
        artifact_type="experiment",
        artifact_name=experiment_name,
        directory=exp_dir,
        manifest_path=manifest_path(exp_dir),
        config_hash=config_hash(_semantic_exp_payload(cfg)),
    )
    stages = tuple(_stage_ref(cfg, stage, timestamp=timestamp) for stage in all_stages(cfg))
    return ResolvedArtifacts(experiment=experiment, stages=stages)


def _stage_ref(cfg: ExpCfg, stage: StageSpec, *, timestamp: str) -> ArtifactRef:
    artifact_name, payload = _stage_artifact_name_and_payload(cfg, stage.name, timestamp)
    directory = artifact_dir(artifact_root(cfg), artifact_name)
    return ArtifactRef(
        stage=stage.name,
        artifact_type=stage.artifact_type,
        artifact_name=artifact_name,
        directory=directory,
        manifest_path=manifest_path(directory),
        config_hash=config_hash(payload),
        enabled=stage.enabled,
        requested=stage.requested,
        required=stage.required,
        dependency_reason=stage.dependency_reason,
        action="run-if-needed" if stage.required else "skipped",
        status="planned" if stage.enabled else "skipped",
        entrypoint=stage.entrypoint,
    )


def _stage_artifact_name_and_payload(
    cfg: ExpCfg, stage_name: str, timestamp: str
) -> tuple[str, object]:
    if stage_name == "contact_gen":
        return contact_artifact_name(cfg), {
            "general": _semantic_contact_general_payload(cfg),
            "contact_gen": _semantic_contact_gen_payload(cfg.contact_gen),
        }
    if stage_name == "pretrain":
        return encoder_artifact_name(cfg), {
            "general": cfg.general,
            "contact_gen": _semantic_contact_gen_payload(cfg.contact_gen),
            "pretrain": _semantic_pretrain_payload(cfg.pretrain),
            "model": _semantic_pretrain_model_payload(cfg),
        }
    if stage_name == "rl":
        contact_payload = _semantic_contact_gen_payload(cfg.contact_gen) if cfg.contact_gen.enabled else None
        return rl_artifact_name(cfg, timestamp), {
            "general": cfg.general,
            "contact_gen": contact_payload,
            "model": cfg.model,
            "rl": cfg.rl,
            "timestamp": timestamp,
        }
    return stage_name, {"stage": stage_name, "config": cfg}



def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

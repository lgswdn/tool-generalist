"""Resolve experiment configs into local artifact locations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from configs.config_exp import ExpCfg
from utils.artifacts.naming import (
    CONTACT_ARTIFACT_GENERAL_NAME,
    artifact_dir,
    contact_artifact_name,
    encoder_artifact_name,
    experiment_artifact_name,
    rl_artifact_name,
)
from utils.artifacts.paths import artifact_root, manifest_path
from utils.config.hash import config_hash
from utils.io import to_plain_data
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


def _semantic_exp_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    payload.pop("num_gpus", None)
    _normalize_artifact_root(payload)
    if "contact_gen" in payload:
        payload["contact_gen"] = _strip_contact_runtime_fields(payload["contact_gen"])
    if "pretrain" in payload:
        payload["pretrain"] = _strip_pretrain_runtime_fields(payload["pretrain"])
    return payload


def _normalize_artifact_root(payload: dict) -> None:
    general = payload.get("general")
    if isinstance(general, dict):
        general["artifact_root"] = "artifacts"


def _semantic_contact_general_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg.general)
    payload["artifact_root"] = "artifacts"
    payload["name"] = CONTACT_ARTIFACT_GENERAL_NAME
    return payload


def _semantic_contact_gen_payload(contact_cfg) -> dict:
    return _strip_contact_runtime_fields(to_plain_data(contact_cfg))


def _semantic_pretrain_payload(pretrain_cfg) -> dict:
    return _strip_pretrain_runtime_fields(to_plain_data(pretrain_cfg))


def _semantic_pretrain_model_payload(cfg: ExpCfg) -> dict:
    model = to_plain_data(cfg.model)
    policy_fusion = dict(model.get("policy_fusion") or {})
    if policy_fusion:
        policy_fusion["query_dim"] = 128
        model["policy_fusion"] = policy_fusion
    return model


def _strip_contact_runtime_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    cleaned["enabled"] = True
    physics = dict(cleaned.get("physics") or {})
    physics.pop("num_workers", None)
    cleaned["physics"] = physics
    return cleaned


def _strip_pretrain_runtime_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    batch = dict(cleaned.get("batch") or {})
    batch.pop("num_workers", None)
    cleaned["batch"] = batch
    if isinstance(cleaned.get("loss"), dict):
        cleaned["loss"] = dict(cleaned["loss"])
    for key in ("logger", "wandb_project", "wandb_run_name", "wandb_entity", "wandb_mode"):
        cleaned.pop(key, None)
    _strip_default_sdf_relative_loss(cleaned)
    _strip_default_condition_normalization(cleaned)
    return cleaned


def _strip_default_sdf_relative_loss(pretrain: dict) -> None:
    loss = pretrain.get("loss")
    if not isinstance(loss, dict):
        return
    if bool(loss.get("sdf_relative_loss", False)):
        return
    if float(loss.get("sdf_relative_eps", 0.005)) != 0.005:
        return
    loss.pop("sdf_relative_loss", None)
    loss.pop("sdf_relative_eps", None)


def _strip_default_condition_normalization(pretrain: dict) -> None:
    if pretrain.get("condition_normalization") is not None:
        return
    pretrain.pop("condition_normalization", None)
    pretrain.pop("condition_norm_sample_files", None)
    pretrain.pop("condition_norm_eps", None)



def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

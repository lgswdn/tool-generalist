"""Artifact naming rules for experiment planning."""

from __future__ import annotations

import re
from pathlib import Path

from configs.config_exp import ExpCfg
from utils.config.serialization import config_hash
from utils.io import to_plain_data

CONTACT_ARTIFACT_GENERAL_NAME = "fork_sdf"


def sanitize_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def contact_artifact_name(cfg: ExpCfg) -> str:
    return "/".join(
        (
            "contact",
            sanitize_name(CONTACT_ARTIFACT_GENERAL_NAME),
            sanitize_name(cfg.contact_gen.name),
            config_hash(
                {
                    "general": _contact_stable_general_payload(cfg),
                    "contact_gen": _contact_hash_payload(cfg.contact_gen),
                }
            ),
        )
    )


def encoder_artifact_name(cfg: ExpCfg) -> str:
    stage = f"{sanitize_name(cfg.pretrain.name)}_{sanitize_name(cfg.model.name)}"
    return "/".join(
        (
            "encoder",
            sanitize_name(cfg.general.name),
            sanitize_name(cfg.contact_gen.name),
            stage,
            config_hash(
                {
                    "general": _location_stable_general_payload(cfg),
                    "contact_gen": _contact_hash_payload(cfg.contact_gen),
                    "pretrain": _pretrain_stable_payload(cfg),
                    "model": _pretrain_model_hash_payload(cfg),
                }
            ),
        )
    )


def rl_artifact_name(cfg: ExpCfg, timestamp: str) -> str:
    encoder_name = sanitize_name(cfg.model.tce.name or cfg.model.tce.encoder_type)
    contact_name = sanitize_name(cfg.contact_gen.name) if cfg.contact_gen.enabled else "no-contact"
    return "/".join(
        (
            "RL",
            sanitize_name(cfg.general.name),
            contact_name,
            encoder_name,
            sanitize_name(cfg.rl.name),
            sanitize_name(timestamp),
        )
    )


def experiment_artifact_name(cfg: ExpCfg) -> str:
    return "/".join(
        (
            "experiment",
            sanitize_name(cfg.name),
            config_hash(_location_stable_exp_payload(cfg)),
        )
    )


def artifact_dir(root: str | Path, artifact_name: str) -> Path:
    return Path(root).expanduser() / Path(*artifact_name.split("/"))


def _location_stable_general_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg.general)
    payload["artifact_root"] = "artifacts"
    return payload


def _contact_stable_general_payload(cfg: ExpCfg) -> dict:
    payload = _location_stable_general_payload(cfg)
    payload["name"] = CONTACT_ARTIFACT_GENERAL_NAME
    return payload


def _location_stable_exp_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    if isinstance(payload.get("general"), dict):
        payload["general"]["artifact_root"] = "artifacts"
    if isinstance(payload.get("pretrain"), dict):
        payload["pretrain"] = _strip_pretrain_logging_fields(payload["pretrain"])
    if isinstance(payload.get("contact_gen"), dict):
        payload["contact_gen"]["enabled"] = True
    return payload


def _pretrain_stable_payload(cfg: ExpCfg) -> dict:
    return _strip_pretrain_logging_fields(to_plain_data(cfg.pretrain))


def _contact_hash_payload(contact_cfg) -> dict:
    payload = to_plain_data(contact_cfg)
    payload["enabled"] = True
    return payload


def _pretrain_model_hash_payload(cfg: ExpCfg) -> dict:
    model = to_plain_data(cfg.model)
    policy_fusion = dict(model.get("policy_fusion") or {})
    if policy_fusion:
        policy_fusion["query_dim"] = 128
        model["policy_fusion"] = policy_fusion
    return model


def _strip_pretrain_logging_fields(payload: dict) -> dict:
    cleaned = dict(payload)
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

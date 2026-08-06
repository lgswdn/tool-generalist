"""Artifact naming rules for experiment planning."""

from __future__ import annotations

import re
from pathlib import Path

from configs.config_exp import ExpCfg
from utils.config.hash_payloads import (
    CONTACT_ARTIFACT_GENERAL_NAME,
    contact_general_payload as _contact_general_payload,
    contact_payload as _contact_payload,
    experiment_payload as _experiment_payload,
    pretrain_artifact_payload as _pretrain_artifact_payload,
    pretrain_namespace as _pretrain_namespace,
)
from utils.config.serialization import config_hash


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
                    "general": _contact_general_payload(cfg),
                    "contact_gen": _contact_payload(cfg.contact_gen),
                }
            ),
        )
    )


def encoder_artifact_name(cfg: ExpCfg) -> str:
    stage = f"{sanitize_name(cfg.pretrain.name)}_{sanitize_name(cfg.model.name)}"
    return "/".join(
        (
            "encoder",
            sanitize_name(_pretrain_namespace(cfg)),
            sanitize_name(cfg.contact_gen.name),
            stage,
            config_hash(_pretrain_artifact_payload(cfg)),
        )
    )


def rl_artifact_name(cfg: ExpCfg, timestamp: str) -> str:
    encoder = cfg.model.encoder
    encoder_name = sanitize_name(encoder.name or encoder.encoder_type)
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
            config_hash(_experiment_payload(cfg)),
        )
    )


def artifact_dir(root: str | Path, artifact_name: str) -> Path:
    return Path(root).expanduser() / Path(*artifact_name.split("/"))

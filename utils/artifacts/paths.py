"""Artifact path helpers for dry-run experiment runs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from configs.config_exp import ExpCfg
from utils.artifacts.naming import (
    artifact_dir as named_artifact_dir,
    contact_artifact_name,
    encoder_artifact_name,
    experiment_artifact_name,
    rl_artifact_name,
)
from utils.config.hash import short_hash


MANIFEST_FILENAME = "manifest.json"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def artifact_root(cfg: ExpCfg) -> Path:
    return Path(cfg.general.artifact_root).expanduser()


def experiment_dir(cfg: ExpCfg) -> Path:
    return named_artifact_dir(artifact_root(cfg), experiment_artifact_name(cfg))


def stage_dir(cfg: ExpCfg, stage_name: str, *, timestamp: str | None = None) -> Path:
    if stage_name == "contact_gen":
        artifact_name = contact_artifact_name(cfg)
    elif stage_name == "pretrain":
        artifact_name = encoder_artifact_name(cfg)
    elif stage_name == "rl":
        artifact_name = rl_artifact_name(cfg, timestamp or _utc_timestamp())
    else:
        payload = {"stage": stage_name, "config": cfg}
        artifact_name = f"{safe_name(stage_name)}/{_hashed_name(cfg.name, payload)}"
    return named_artifact_dir(artifact_root(cfg), artifact_name)


def manifest_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / MANIFEST_FILENAME


def _hashed_name(name: str, payload: object) -> str:
    return f"{safe_name(name)}-{short_hash(payload)}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

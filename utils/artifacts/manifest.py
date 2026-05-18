"""Local artifact manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.io import read_json as _read_json, write_json


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "artifact_manifest_v1"


@dataclass
class ArtifactManifest:
    artifact_type: str
    artifact_name: str
    exp_cfg_name: str
    config_hash: str
    status: str = "planned"
    git_commit: str = "unknown"
    git_dirty: bool = False
    created_at: str = ""
    source_paths: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    config_dump: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": self.artifact_type,
            "artifact_name": self.artifact_name,
            "exp_cfg_name": self.exp_cfg_name,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "created_at": self.created_at,
            "status": self.status,
            "source_paths": dict(self.source_paths),
            "metrics": dict(self.metrics),
            "config_dump": dict(self.config_dump),
            "runtime": dict(self.runtime),
        }


def manifest_path_for(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / MANIFEST_FILENAME


def write_manifest(artifact_dir: str | Path, manifest: ArtifactManifest | Mapping[str, Any]) -> Path:
    path = manifest_path_for(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_payload() if isinstance(manifest, ArtifactManifest) else dict(manifest)

    return write_json(path, payload)


def read_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = manifest_path_for(source)

    payload = _read_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a mapping: {source}")
    return payload


def manifest_is_complete(path: str | Path) -> bool:
    source = Path(path)
    if not source.exists():
        return False
    try:
        payload = read_manifest(source)
    except Exception:
        return False
    return payload.get("schema_version") == MANIFEST_SCHEMA_VERSION and payload.get("status") == "complete"


def maybe_existing_manifest(artifact_dir: str | Path) -> Optional[dict[str, Any]]:
    path = manifest_path_for(artifact_dir)
    if not path.exists():
        return None
    return read_manifest(path)

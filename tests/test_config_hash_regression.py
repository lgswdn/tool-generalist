from __future__ import annotations

import json
from pathlib import Path

from contact_generation.batch_generate import contact_config_hash
from utils.artifacts.naming import (
    contact_artifact_name,
    encoder_artifact_name,
    experiment_artifact_name,
)
from utils.artifacts.resolver import resolve_artifacts
from utils.config.hash import config_hash as hash_config_hash
from utils.config.loader import load_exp_cfg
from utils.config.serialization import config_hash as serialization_config_hash


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/config_hash_regression.json"


def _experiment_config_paths() -> list[str]:
    return [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "configs/experiments").glob("*.py"))
        if path.name != "__init__.py"
    ]


def _current_hash_snapshot(path: str, timestamp: str) -> dict:
    cfg = load_exp_cfg(ROOT / path)
    resolved = resolve_artifacts(cfg, timestamp=timestamp)
    return {
        "config_hash": {
            "utils.config.hash": hash_config_hash(cfg),
            "utils.config.serialization": serialization_config_hash(cfg),
        },
        "artifact_names": {
            "experiment": experiment_artifact_name(cfg),
            "contact": contact_artifact_name(cfg),
            "encoder": encoder_artifact_name(cfg),
        },
        "resolve_artifacts": {
            "experiment": resolved.experiment.config_hash,
            "stages": {ref.stage: ref.config_hash for ref in resolved.stages},
        },
        "contact_config_hash": contact_config_hash(cfg),
    }


def test_all_experiment_config_hashes_and_artifact_names_match_fixture():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert fixture["load_errors"] == {}
    assert _experiment_config_paths() == sorted(fixture["configs"])

    timestamp = fixture["timestamp"]
    for path, expected in fixture["configs"].items():
        assert _current_hash_snapshot(path, timestamp) == expected

"""Deterministic hashing helpers for experiment configs."""

from __future__ import annotations

from typing import Any

from utils.io import canonical_json, hash_json, to_plain_data


def config_hash(value: Any) -> str:
    return hash_json(_hash_payload(value))


def short_hash(value: Any, length: int = 12) -> str:
    return config_hash(value)[:length]


def _hash_payload(value: Any) -> Any:
    payload = to_plain_data(value)
    if isinstance(payload, dict):
        payload.pop("pretrain_reuse", None)
    return payload

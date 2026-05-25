"""Config serialization and hashing helpers."""

from __future__ import annotations

from typing import Any

from configs.config_contact_gen import (
    CONTACT_GEN_HASH_COMPAT_DEFAULTS,
    CONTACT_GEN_HASH_RUNTIME_FIELDS,
    strip_contact_gen_hash_defaults,
)
from utils.io import canonical_json, hash_json, to_plain_data


def config_hash(value: Any) -> str:
    return hash_json(_hash_payload(value))


def hash_short(value: Any, length: int = 12) -> str:
    return config_hash(value)[:length]


def _hash_payload(value: Any) -> Any:
    payload = to_plain_data(value)
    if isinstance(payload, dict):
        payload.pop("pretrain_reuse", None)
    return _strip_hash_compat_defaults(payload)


def _strip_hash_compat_defaults(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned = {key: _strip_hash_compat_defaults(value) for key, value in payload.items()}
        if isinstance(cleaned.get("contact_gen"), dict):
            cleaned["contact_gen"] = strip_contact_gen_hash_defaults(cleaned["contact_gen"])
        if _looks_like_contact_gen_payload(cleaned):
            cleaned = strip_contact_gen_hash_defaults(cleaned)
        return cleaned
    if isinstance(payload, list):
        return [_strip_hash_compat_defaults(item) for item in payload]
    return payload


def _looks_like_contact_gen_payload(payload: dict) -> bool:
    if not any(
        key in payload
        for key in CONTACT_GEN_HASH_COMPAT_DEFAULTS.keys() | CONTACT_GEN_HASH_RUNTIME_FIELDS
    ):
        return False
    return any(key in payload for key in ("schema_version", "contact_mode_prob", "physics"))

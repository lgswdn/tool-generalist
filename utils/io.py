"""Small JSON and hashing helpers shared across stages."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def to_plain_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, MappingProxyType):
        return to_plain_data(dict(value))
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_plain_data(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def hash_json(value: Any, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any, *, indent: int = 2, sort_keys: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_plain_data(payload), f, indent=indent, sort_keys=sort_keys)
        f.write("\n")
    return target

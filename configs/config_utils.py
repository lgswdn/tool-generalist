"""Small helpers for composing experiment config presets."""

from __future__ import annotations

from copy import deepcopy
from typing import TypeVar


T = TypeVar("T")


def clone_cfg(cfg: T) -> T:
    """Return an isolated copy of a module-level config preset."""

    return deepcopy(cfg)

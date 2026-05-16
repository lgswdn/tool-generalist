"""Runtime metadata helpers that avoid heavy framework imports."""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def runtime_metadata(*, cwd: str | Path, argv: list[str]) -> dict[str, Any]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "python_version": sys.version.split()[0],
        "torch_version": "not_imported",
        "cuda_version": "unknown",
        "isaac_sim_version": "not_imported",
        "isaac_lab_version": "not_imported",
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "cwd": str(Path(cwd).resolve()),
        "command": list(argv),
        "cuda_visible_devices": cuda_visible_devices,
        "num_gpus_visible": visible_gpu_count(cuda_visible_devices),
    }


def visible_gpu_count(cuda_visible_devices: str | None) -> int | None:
    if cuda_visible_devices is None:
        return None
    stripped = cuda_visible_devices.strip()
    if stripped in {"", "-1"}:
        return 0
    return len([item for item in stripped.split(",") if item.strip()])


def git_metadata(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd).resolve()
    commit = _git(root, "rev-parse", "HEAD") or "unknown"
    status = _git(root, "status", "--short")
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
    }


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()

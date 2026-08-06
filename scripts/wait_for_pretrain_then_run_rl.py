#!/usr/bin/env python3
"""Wait for a shared pretrain artifact, then launch an RL run.

This is useful when pretrain is already running on another machine.  The script
polls the shared artifact directory instead of waiting on a local PID.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config_exp import ExpCfg
from utils.artifacts.manifest import manifest_is_complete, read_manifest
from utils.artifacts.resolver import ArtifactRef, resolve_artifacts
from utils.config.loader import load_exp_cfg
from utils.experiment.runner import (
    _resolve_reuse_config_ref,
    _resolve_stage_encoder_checkpoint_from_manifest,
)


def _resolve_config(raw: str) -> Path:
    path = Path(raw)
    stem = raw[:-3] if raw.endswith(".py") else raw
    normalized = stem
    if "_shard_" in normalized:
        prefix, suffix = normalized.rsplit("_shard_", 1)
        if suffix.isdigit():
            normalized = f"{prefix}_shard{suffix}"

    candidates = [
        path,
        Path(f"{stem}.py"),
        Path(f"{normalized}.py"),
        REPO_ROOT / "configs" / "experiments" / f"{stem}.py",
        REPO_ROOT / "configs" / "experiments" / f"{normalized}.py",
    ]
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find config for {raw!r}. Tried:\n  {tried}")


def _pretrain_ref(cfg: ExpCfg) -> ArtifactRef | None:
    for ref in resolve_artifacts(cfg).stages:
        if ref.stage == "pretrain":
            return ref
    return None


def _watched_pretrain_ref(rl_config: Path, explicit_pretrain_config: str | None) -> tuple[Path, ArtifactRef]:
    if explicit_pretrain_config:
        pretrain_config = _resolve_config(explicit_pretrain_config)
        pretrain_cfg = load_exp_cfg(pretrain_config)
    else:
        rl_cfg = load_exp_cfg(rl_config)
        if rl_cfg.pretrain_reuse:
            reuse_config = _resolve_reuse_config_ref(rl_cfg.pretrain_reuse, str(rl_config))
            pretrain_config = _resolve_config(str(reuse_config))
            pretrain_cfg = load_exp_cfg(pretrain_config)
        else:
            pretrain_config = rl_config
            pretrain_cfg = rl_cfg

    ref = _pretrain_ref(pretrain_cfg)
    if ref is None:
        raise RuntimeError(
            f"No pretrain stage is defined by watched config: {pretrain_config}"
        )
    return pretrain_config, ref


def _checkpoint_from_manifest(ref: ArtifactRef) -> Path:
    resolved = _resolve_stage_encoder_checkpoint_from_manifest(None, ref)
    if resolved:
        return Path(resolved)
    return ref.directory / "best.pt"


def _manifest_status(ref: ArtifactRef) -> str:
    if not ref.manifest_path.exists():
        return "missing"
    try:
        payload = read_manifest(ref.manifest_path)
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}"
    return str(payload.get("status", "unknown"))


def _wait_for_pretrain(ref: ArtifactRef, *, interval_s: float, timeout_s: float) -> Path:
    start = time.monotonic()
    last_line = ""
    while True:
        complete = manifest_is_complete(ref.manifest_path)
        checkpoint = _checkpoint_from_manifest(ref) if complete else ref.directory / "best.pt"
        checkpoint_exists = checkpoint.is_file()
        if complete and checkpoint_exists:
            print(f"[wait-pretrain] complete: {ref.manifest_path}", flush=True)
            print(f"[wait-pretrain] checkpoint: {checkpoint}", flush=True)
            return checkpoint

        elapsed = time.monotonic() - start
        if timeout_s > 0 and elapsed >= timeout_s:
            raise TimeoutError(
                "Timed out waiting for pretrain artifact. "
                f"manifest={ref.manifest_path} status={_manifest_status(ref)} "
                f"checkpoint={checkpoint} checkpoint_exists={checkpoint_exists}"
            )

        line = (
            f"[wait-pretrain] waiting elapsed={elapsed / 60.0:.1f}m "
            f"status={_manifest_status(ref)} checkpoint_exists={checkpoint_exists}"
        )
        if line != last_line:
            print(line, flush=True)
            last_line = line
        time.sleep(interval_s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a pretrain manifest/checkpoint on shared storage, then run RL."
    )
    parser.add_argument("rl_config", help="RL config name or path to pass to run.bash.")
    parser.add_argument(
        "--pretrain-config",
        help=(
            "Optional explicit pretrain config to watch. Defaults to the RL config's "
            "pretrain_reuse target, or the RL config's own pretrain stage."
        ),
    )
    parser.add_argument("--interval", type=float, default=60.0, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=float, default=0.0, help="Timeout in seconds; 0 means forever.")
    parser.add_argument("--run-bash", default=str(REPO_ROOT / "run.bash"), help="Path to run.bash.")
    parser.add_argument("--curr", action="store_true", help="Pass 'curr' to run.bash after the RL config.")
    parser.add_argument(
        "--print-target",
        action="store_true",
        help="Print the watched artifact and launch command, then exit without waiting or running.",
    )
    args, run_args = parser.parse_known_args()

    rl_config = _resolve_config(args.rl_config)
    pretrain_config, ref = _watched_pretrain_ref(rl_config, args.pretrain_config)
    command = [args.run_bash, str(rl_config)]
    if args.curr:
        command.append("curr")
    command.extend(arg for arg in run_args if arg != "--")

    print(f"[wait-pretrain] rl_config={rl_config}", flush=True)
    print(f"[wait-pretrain] watched_pretrain_config={pretrain_config}", flush=True)
    print(f"[wait-pretrain] watched_manifest={ref.manifest_path}", flush=True)
    print(f"[wait-pretrain] watched_default_checkpoint={ref.directory / 'best.pt'}", flush=True)
    print(f"[wait-pretrain] launch_command={' '.join(command)}", flush=True)
    if args.print_target:
        return 0

    _wait_for_pretrain(ref, interval_s=args.interval, timeout_s=args.timeout)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

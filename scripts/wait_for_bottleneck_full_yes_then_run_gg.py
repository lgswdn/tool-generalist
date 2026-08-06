#!/usr/bin/env python3
"""Wait for one bottleneck full-YES parent, then launch its GG transfer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/mnt/project/world_model/tool_generalist/artifacts")
ORIGINAL_GRIPPER_MANIFEST = Path(
    "/mnt/project/world_model/tool_generalist/gripper/generated_grippers.json"
)
ORIGINAL_GRIPPER_GENERATED_ROOT = Path(
    "/mnt/project/world_model/tool_generalist/gripper/franka_with_diverse_hands"
)
ORIGINAL_GRIPPER_COUNT = 400
ORIGINAL_GRIPPER_MANIFEST_RESTORED_AT_UTC = "2026-07-18T13:05:08+00:00"
FULL_YES_OBJECTS = "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _valid_original_gripper_manifest() -> tuple[bool, str]:
    payload = _read_json(ORIGINAL_GRIPPER_MANIFEST)
    if payload is None:
        return False, f"unreadable manifest: {ORIGINAL_GRIPPER_MANIFEST}"
    generated_root = payload.get("generated_root")
    grippers = payload.get("grippers")
    if Path(str(generated_root)).resolve() != ORIGINAL_GRIPPER_GENERATED_ROOT.resolve():
        return False, f"wrong generated_root: {generated_root!r}"
    if not isinstance(grippers, list) or len(grippers) != ORIGINAL_GRIPPER_COUNT:
        count = len(grippers) if isinstance(grippers, list) else None
        return False, f"wrong gripper count: {count!r}"
    return True, "ok"


def _parent_contract_matches(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    parent_experiment: str,
    rank: int,
) -> bool:
    config = manifest.get("config_dump")
    if not isinstance(config, Mapping):
        return False
    general = config.get("general", {})
    model = config.get("model", {})
    rl = config.get("rl", {})
    if not all(isinstance(value, Mapping) for value in (general, model, rl)):
        return False
    tce = model.get("tce", {})
    ppo = rl.get("ppo", {})
    action = rl.get("action", {})
    observation = rl.get("observation", {})
    scale = rl.get("domain_randomization", {}).get("object", {}).get("scale", {})
    return (
        manifest.get("status") == "complete"
        and run_id >= "20260718T130508Z"
        and config.get("name") == parent_experiment
        and isinstance(tce, Mapping)
        and tce.get("encoder_token_bottleneck_rank") == rank
        and general.get("rl_objects_manifest") == FULL_YES_OBJECTS
        and isinstance(ppo, Mapping)
        and ppo.get("max_iterations") == 5000
        and isinstance(action, Mapping)
        and action.get("scale") == 0.06
        and isinstance(observation, Mapping)
        and observation.get("object_cloud_source") == "mesh_sampled"
        and isinstance(scale, Mapping)
        and scale.get("enabled") is True
        and scale.get("range") in ([0.1, 0.2], (0.1, 0.2))
    )


def _completed_parent(
    *,
    parent_experiment: str,
    rank: int,
) -> Path | None:
    runs_root = (
        ARTIFACT_ROOT
        / "RL"
        / parent_experiment
        / "no-contact"
        / "TCE"
        / parent_experiment
    )
    if not runs_root.is_dir():
        return None
    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
        manifest = _read_json(run_dir / "manifest.json")
        checkpoint = run_dir / "model_best.pt"
        if (
            manifest is not None
            and checkpoint.is_file()
            and _parent_contract_matches(
                manifest,
                run_id=run_dir.name,
                parent_experiment=parent_experiment,
                rank=rank,
            )
        ):
            return checkpoint
    return None


def _child_already_launched(child_experiment: str) -> Path | None:
    root = ARTIFACT_ROOT / "RL" / child_experiment
    if not root.is_dir():
        return None
    manifests = sorted(root.rglob("manifest.json"), reverse=True)
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if manifest is None:
            continue
        config = manifest.get("config_dump", {})
        if (
            isinstance(config, Mapping)
            and config.get("name") == child_experiment
            and manifest.get("status") in {"running", "complete"}
        ):
            return manifest_path
    return None


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=(10, 16))
    parser.add_argument("--parent-experiment", required=True)
    parser.add_argument("--child-experiment", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    ok, reason = _valid_original_gripper_manifest()
    if not ok:
        print(f"[wait-bottleneck] refusing to monitor: {reason}", file=sys.stderr, flush=True)
        return 2

    already = _child_already_launched(args.child_experiment)
    if already is not None:
        print(f"[wait-bottleneck] child already running/complete: {already}", flush=True)
        return 0

    print(
        f"[wait-bottleneck] rank={args.rank} parent={args.parent_experiment} "
        f"child={args.child_experiment} poll_seconds={args.poll_seconds:g}",
        flush=True,
    )
    while True:
        checkpoint = _completed_parent(
            parent_experiment=args.parent_experiment,
            rank=args.rank,
        )
        if checkpoint is not None:
            print(f"[wait-bottleneck] completed parent: {checkpoint}", flush=True)
            print(f"[wait-bottleneck] launching: ./run.bash {args.child_experiment}", flush=True)
            run_bash = REPO_ROOT / "run.bash"
            os.chdir(REPO_ROOT)
            os.execv(str(run_bash), [str(run_bash), args.child_experiment])

        print(f"[wait-bottleneck] {_stamp()} parent not complete; sleeping", flush=True)
        if args.check_once:
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

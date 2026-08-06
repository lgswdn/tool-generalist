#!/usr/bin/env python3
"""Wait for the newest configured DGN parent, then launch its GG transfer."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.artifacts.resolver import resolve_artifacts
from utils.config.loader import load_exp_cfg


ARTIFACT_ROOT = Path("/mnt/project/world_model/tool_generalist/artifacts")
FULL_YES_OBJECTS = "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"
DEFAULT_GRIPPER_MANIFEST = Path(
    "/mnt/project/world_model/tool_generalist/gripper/generated_grippers.json"
)
DEFAULT_GRIPPER_GENERATED_ROOT = Path(
    "/mnt/project/world_model/tool_generalist/gripper/franka_with_diverse_hands"
)
DEFAULT_GRIPPER_COUNT = 400


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _valid_generated_gripper_manifest(
    manifest_path: Path,
    generated_root: Path,
    expected_count: int,
) -> tuple[bool, str]:
    payload = _read_json(manifest_path)
    if payload is None:
        return False, f"unreadable manifest: {manifest_path}"
    recorded_root = payload.get("generated_root")
    grippers = payload.get("grippers")
    if Path(str(recorded_root)).resolve() != generated_root.resolve():
        return False, f"wrong generated_root: {recorded_root!r}"
    if not isinstance(grippers, list) or len(grippers) != expected_count:
        count = len(grippers) if isinstance(grippers, list) else None
        return False, f"wrong gripper count: {count!r}"
    return True, "ok"


def _valid_parent_manifest(
    manifest: Mapping[str, Any],
    *,
    parent_experiment: str,
    encoder_backend: str,
    expected_rl_config_hash: str,
    expected_max_iterations: int,
) -> bool:
    config = manifest.get("config_dump")
    if not isinstance(config, Mapping):
        return False
    general = config.get("general")
    model = config.get("model")
    rl = config.get("rl")
    if not all(isinstance(value, Mapping) for value in (general, model, rl)):
        return False
    ppo = rl.get("ppo")
    action = rl.get("action")
    observation = rl.get("observation")
    randomization = rl.get("domain_randomization")
    if not all(
        isinstance(value, Mapping)
        for value in (ppo, action, observation, randomization)
    ):
        return False
    obj = randomization.get("object")
    scale = obj.get("scale") if isinstance(obj, Mapping) else None
    return (
        manifest.get("status") == "complete"
        and manifest.get("config_hash") == expected_rl_config_hash
        and config.get("name") == parent_experiment
        and model.get("encoder_backend") == encoder_backend
        and general.get("rl_objects_manifest") == FULL_YES_OBJECTS
        and ppo.get("max_iterations") == expected_max_iterations
        and action.get("scale") == 0.06
        and observation.get("object_cloud_source") == "mesh_sampled"
        and isinstance(scale, Mapping)
        and scale.get("enabled") is True
        and scale.get("range") in ([0.1, 0.2], (0.1, 0.2))
    )


def _completed_parent(
    *,
    parent_experiment: str,
    parent_contact_name: str,
    encoder_family: str,
    encoder_backend: str,
    expected_rl_config_hash: str,
    expected_max_iterations: int,
    checkpoint_filename: str,
) -> Path | None:
    runs_root = (
        ARTIFACT_ROOT
        / "RL"
        / parent_experiment
        / parent_contact_name
        / encoder_family
        / parent_experiment
    )
    if not runs_root.is_dir():
        return None
    run_dirs = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        return None

    # Do not fall back to an older completed run while a newer parent is still
    # running. The child must always transfer from the newest parent run.
    run_dir = run_dirs[0]
    manifest = _read_json(run_dir / "manifest.json")
    checkpoint = run_dir / checkpoint_filename
    if (
        manifest is not None
        and checkpoint.is_file()
        and _valid_parent_manifest(
            manifest,
            parent_experiment=parent_experiment,
            encoder_backend=encoder_backend,
            expected_rl_config_hash=expected_rl_config_hash,
            expected_max_iterations=expected_max_iterations,
        )
    ):
        return checkpoint
    return None


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment", required=True)
    parser.add_argument("--child-experiment", required=True)
    parser.add_argument("--encoder-family", required=True)
    parser.add_argument("--encoder-backend", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-once", action="store_true")
    parser.add_argument(
        "--parent-checkpoint-filename",
        default="model_best.pt",
    )
    parser.add_argument(
        "--generated-gripper-manifest",
        type=Path,
        default=DEFAULT_GRIPPER_MANIFEST,
    )
    parser.add_argument(
        "--generated-gripper-root",
        type=Path,
        default=DEFAULT_GRIPPER_GENERATED_ROOT,
    )
    parser.add_argument(
        "--generated-gripper-count",
        type=int,
        default=DEFAULT_GRIPPER_COUNT,
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.generated_gripper_count <= 0:
        parser.error("--generated-gripper-count must be positive")

    child_config = REPO_ROOT / "configs/experiments" / f"{args.child_experiment}.py"
    if not child_config.is_file():
        parser.error(f"child experiment config does not exist: {child_config}")
    parent_config = REPO_ROOT / "configs/experiments" / f"{args.parent_experiment}.py"
    if not parent_config.is_file():
        parser.error(f"parent experiment config does not exist: {parent_config}")
    parent_cfg = load_exp_cfg(parent_config)
    parent_rl_refs = [
        ref for ref in resolve_artifacts(parent_cfg).stages if ref.stage == "rl"
    ]
    if len(parent_rl_refs) != 1:
        parser.error(
            f"expected exactly one RL stage for parent config: {parent_config}"
        )
    expected_parent_rl_hash = parent_rl_refs[0].config_hash
    rl_artifact_parts = Path(parent_rl_refs[0].artifact_name).parts
    if len(rl_artifact_parts) < 3 or rl_artifact_parts[0] != "RL":
        parser.error(
            "cannot derive parent contact lineage from RL artifact name: "
            f"{parent_rl_refs[0].artifact_name}"
        )
    parent_contact_name = rl_artifact_parts[2]
    expected_max_iterations = int(parent_cfg.rl.ppo.max_iterations)

    ok, reason = _valid_generated_gripper_manifest(
        args.generated_gripper_manifest,
        args.generated_gripper_root,
        args.generated_gripper_count,
    )
    if not ok:
        print(f"[wait-full-yes] refusing to monitor: {reason}", file=sys.stderr, flush=True)
        return 2

    print(
        f"[wait-full-yes] parent={args.parent_experiment} "
        f"contact={parent_contact_name} "
        f"backend={args.encoder_backend} child={args.child_experiment} "
        f"parent_rl_hash={expected_parent_rl_hash} "
        f"max_iterations={expected_max_iterations} "
        f"poll_seconds={args.poll_seconds:g}",
        flush=True,
    )
    while True:
        checkpoint = _completed_parent(
            parent_experiment=args.parent_experiment,
            parent_contact_name=parent_contact_name,
            encoder_family=args.encoder_family,
            encoder_backend=args.encoder_backend,
            expected_rl_config_hash=expected_parent_rl_hash,
            expected_max_iterations=expected_max_iterations,
            checkpoint_filename=args.parent_checkpoint_filename,
        )
        if checkpoint is not None:
            print(f"[wait-full-yes] completed parent: {checkpoint}", flush=True)
            print(
                f"[wait-full-yes] launching fresh child: "
                f"./run.bash {args.child_experiment}",
                flush=True,
            )
            run_bash = REPO_ROOT / "run.bash"
            os.chdir(REPO_ROOT)
            os.environ["TOOL_GENERALIST_BYPASS_GG_PARENT_WAIT"] = "1"
            os.execv(str(run_bash), [str(run_bash), args.child_experiment])

        print(
            f"[wait-full-yes] {_stamp()} newest parent not complete/valid; sleeping",
            flush=True,
        )
        if args.check_once:
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

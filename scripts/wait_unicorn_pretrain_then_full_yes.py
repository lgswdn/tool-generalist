#!/usr/bin/env python3
"""Wait for a selected UniCORN pretrain, then launch its full-YES 5k RL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "cross_only_depth1": (
        "unicorn_pretrain_ours_cross_only_depth1",
        "panda_general_unicorn_ours_cross_only_depth1_full_yes_5k",
    ),
    "intersecting": (
        "unicorn_pretrain_ours_intersecting_geometry",
        "panda_general_unicorn_ours_intersecting_geometry_full_yes_5k",
    ),
}
INTERSECTING_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_intersecting_geometry/contact_gen_intersecting_geometry/"
    "unicorn_ours_intersecting_geometry_unicorn_ours_intersecting_geometry/"
    "f6117a7cd0bf6725e3eb43d5636c9731cb5d481682c4854c326aadbb090c85b2/best.pt"
)


def _explicit_complete(checkpoint: Path) -> bool:
    manifest = checkpoint.parent / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return payload.get("status") == "complete" and checkpoint.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=0.0)
    args = parser.parse_args()
    pretrain, full_yes = VARIANTS[args.variant]
    if args.variant == "intersecting":
        start = time.monotonic()
        while not _explicit_complete(INTERSECTING_CHECKPOINT):
            elapsed = time.monotonic() - start
            if args.timeout > 0 and elapsed >= args.timeout:
                raise TimeoutError(
                    f"Timed out waiting for {INTERSECTING_CHECKPOINT}"
                )
            print(
                f"[wait-pretrain] waiting for pinned intersecting checkpoint "
                f"elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )
            time.sleep(args.poll_seconds)
        run_bash = REPO_ROOT / "run.bash"
        print(f"[wait-pretrain] complete: {INTERSECTING_CHECKPOINT}", flush=True)
        print(f"[wait-pretrain] launching: {run_bash} {full_yes}", flush=True)
        os.chdir(REPO_ROOT)
        os.execv(str(run_bash), [str(run_bash), full_yes])

    target = REPO_ROOT / "scripts" / "wait_for_pretrain_then_run_rl.py"
    command = [
        sys.executable,
        str(target),
        full_yes,
        "--pretrain-config",
        pretrain,
        "--interval",
        str(args.poll_seconds),
        "--timeout",
        str(args.timeout),
    ]
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    raise SystemExit(main())

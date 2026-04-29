#!/usr/bin/env python3
"""Batch-run generate_pipeline.py over response JSON files."""

import argparse
import subprocess
import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OBJ trials from a directory of tool response JSON files.")
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=project_path / "responses",
        help="Directory containing response JSON recipe files.",
    )
    parser.add_argument("--task_name", type=str, default="eef", help="Task name passed to generate_pipeline.py.")
    parser.add_argument(
        "--num_variations",
        type=int,
        default=10,
        help="Number of randomized variants passed to generate_pipeline.py.",
    )
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for recipe files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of recipes to process.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed. Per-file seed is base seed + index.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining recipes if one command fails.",
    )
    args = parser.parse_args()

    responses_dir = args.responses_dir.resolve()
    if not responses_dir.is_dir():
        raise FileNotFoundError(f"responses directory not found: {responses_dir}")

    response_files = sorted(p for p in responses_dir.glob(args.pattern) if p.is_file())
    if args.limit is not None:
        response_files = response_files[: args.limit]
    if not response_files:
        raise RuntimeError(f"No response files found: {responses_dir}/{args.pattern}")

    generate_pipeline = project_path / "generate_pipeline.py"
    if not generate_pipeline.is_file():
        raise FileNotFoundError(f"generate_pipeline.py not found: {generate_pipeline}")

    print(f"Found {len(response_files)} response file(s).")
    print(f"Responses dir: {responses_dir}")
    print(f"Task name: {args.task_name}")
    print(f"Num variations: {args.num_variations}")

    failures = []
    for idx, response_file in enumerate(response_files):
        cmd = [
            sys.executable,
            str(generate_pipeline),
            "--task_name",
            args.task_name,
            "--response_file",
            str(response_file),
            "--num_variations",
            str(args.num_variations),
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed + idx)])

        print(f"\n[{idx + 1}/{len(response_files)}] {response_file.name}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=project_path)
        if result.returncode != 0:
            failures.append((response_file, result.returncode))
            print(f"[ERROR] failed with return code {result.returncode}: {response_file}")
            if not args.continue_on_error:
                return result.returncode

    print("\nBatch generation finished.")
    print(f"Processed: {len(response_files)}")
    print(f"Failed: {len(failures)}")
    for response_file, returncode in failures:
        print(f"  - {response_file} returncode={returncode}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

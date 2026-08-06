#!/usr/bin/env python3
"""Summarize failed GraspSim eval results by (object, pose)."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

tf = None
tfds = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GraspSim eval outputs and report ungraspable (object, pose) pairs."
    )
    parser.add_argument(
        "--eval-root",
        required=True,
        type=Path,
        help="Eval output root, e.g. /mnt/project/world_model/tool_generalist/grasp_result_2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/graspsim_ungraspable_pairs"),
        help="Directory for CSV/JSON/txt reports.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.0,
        help="Pairs with success_rate <= this value are reported. Default: 0.0.",
    )
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=1,
        help="Minimum number of evaluated episodes for a pair before reporting it.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Decimal rounding for pose grouping. Default: 4.",
    )
    parser.add_argument(
        "--pose-mode",
        choices=("stable", "full", "orientation"),
        default="stable",
        help=(
            "Pose grouping key. stable=scale+z+quat, full=scale+xyz+quat, "
            "orientation=scale+quat. Default: stable."
        ),
    )
    return parser.parse_args()


def tensor_to_python(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def safe_eval_payload(value: Any) -> Any:
    value = tensor_to_python(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return eval(value, {"array": np.array, "np": np})  # noqa: S307 - trusted local eval artifact.
    return value


def find_dataset_dirs(eval_root: Path) -> list[Path]:
    if (eval_root / "dataset_info.json").exists():
        return [eval_root]
    dataset_dirs = sorted(
        path.parent for path in eval_root.rglob("dataset_info.json") if path.is_file()
    )
    return dataset_dirs


def as_float_list(value: Any, digits: int) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(x), digits) for x in arr.tolist()]


def format_vec(value: list[float]) -> str:
    return "[" + ", ".join(f"{x:.6g}" for x in value) + "]"


def get_target_object(scene_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    object_infos = scene_info.get("object_infos") or scene_info.get("task_config") or {}
    if not object_infos:
        target_info = scene_info.get("target_info")
        if isinstance(target_info, dict):
            return "target_info", target_info
        raise KeyError("No object_infos/task_config/target_info found in environment_config")

    target_keys = [key for key in object_infos if str(key).startswith("target")]
    key = target_keys[0] if target_keys else next(iter(object_infos))
    return str(key), object_infos[key]


def object_name(object_info: dict[str, Any]) -> str:
    for key in ("model_id", "id", "name"):
        value = object_info.get(key)
        if value is not None:
            return str(value)
    return "unknown_object"


def pose_parts(
    object_info: dict[str, Any],
    pose_mode: str,
    round_digits: int,
) -> tuple[str, dict[str, Any]]:
    scale = object_info.get("scale", None)
    scale_key = None if scale is None else round(float(scale), round_digits)
    position = as_float_list(object_info.get("position", []), round_digits)
    orientation = as_float_list(object_info.get("orientation", []), round_digits)

    if pose_mode == "full":
        pose_key = f"scale={scale_key}|pos={format_vec(position)}|quat={format_vec(orientation)}"
    elif pose_mode == "orientation":
        pose_key = f"scale={scale_key}|quat={format_vec(orientation)}"
    else:
        z_value = position[2] if len(position) >= 3 else None
        pose_key = f"scale={scale_key}|z={z_value}|quat={format_vec(orientation)}"

    return pose_key, {
        "scale": scale_key,
        "position": position,
        "orientation": orientation,
        "pose_mode": pose_mode,
    }


def score_success(traj: dict[str, Any]) -> bool:
    if "score" in traj:
        score = safe_eval_payload(traj["score"])
        if isinstance(score, dict):
            return bool(score.get("success", False) or score.get("succeed", False))
    if "success" in traj:
        return bool(tensor_to_python(traj["success"]))
    if "valid" in traj:
        return bool(tensor_to_python(traj["valid"]))
    return False


def load_records(dataset_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    builder = tfds.builder_from_directory(str(dataset_dir))
    dataset = builder.as_dataset(split="all", shuffle_files=False)
    records = []
    for traj in dataset:
        scene_info = safe_eval_payload(traj["environment_config"])
        target_key, target = get_target_object(scene_info)
        obj = object_name(target)
        pose_key, pose = pose_parts(target, args.pose_mode, args.round_digits)
        success = score_success(traj)
        task_id = str(scene_info.get("task_id", ""))
        captions = scene_info.get("captions") or []
        caption = captions[0] if captions else ""
        records.append(
            {
                "dataset_dir": str(dataset_dir),
                "task_id": task_id,
                "caption": caption,
                "target_key": target_key,
                "object": obj,
                "category": target.get("category", ""),
                "success": success,
                "pose_key": pose_key,
                "scale": pose["scale"],
                "position": json.dumps(pose["position"]),
                "orientation": json.dumps(pose["orientation"]),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["object"], record["pose_key"])].append(record)

    rows = []
    for (obj, pose_key), items in sorted(groups.items()):
        attempts = len(items)
        successes = sum(1 for item in items if item["success"])
        failures = attempts - successes
        first = items[0]
        rows.append(
            {
                "object": obj,
                "category": first["category"],
                "pose_key": pose_key,
                "scale": first["scale"],
                "position": first["position"],
                "orientation": first["orientation"],
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "success_rate": successes / attempts if attempts else 0.0,
                "task_ids": json.dumps([item["task_id"] for item in items]),
                "captions": json.dumps(sorted({item["caption"] for item in items})),
            }
        )
    rows.sort(key=lambda row: (row["success_rate"], -row["attempts"], row["object"], row["pose_key"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global tf, tfds
    args = parse_args()
    if args.min_attempts < 1:
        raise SystemExit("--min-attempts must be >= 1")
    if args.failure_threshold < 0.0 or args.failure_threshold > 1.0:
        raise SystemExit("--failure-threshold must be in [0, 1]")

    try:
        import tensorflow as tensorflow_module
        import tensorflow_datasets as tfds_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script needs tensorflow and tensorflow_datasets. Run it with the "
            "same Python used by GraspSim eval, for example:\n"
            "  /isaac-sim/python.sh scripts/analyze_graspsim_ungraspable_pairs.py "
            "--eval-root /mnt/project/world_model/tool_generalist/grasp_result_2"
        ) from exc
    tf = tensorflow_module
    tfds = tfds_module
    tf.config.experimental.set_visible_devices([], "GPU")

    dataset_dirs = find_dataset_dirs(args.eval_root)
    if not dataset_dirs:
        raise SystemExit(f"No TFDS dataset_info.json files found under {args.eval_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        print(f"[INFO] reading {dataset_dir}", flush=True)
        all_records.extend(load_records(dataset_dir, args))

    summary_rows = summarize(all_records)
    ungraspable_rows = [
        row
        for row in summary_rows
        if row["attempts"] >= args.min_attempts and row["success_rate"] <= args.failure_threshold
    ]
    failed_episode_rows = [record for record in all_records if not record["success"]]

    write_csv(
        args.output_dir / "episodes.csv",
        all_records,
        [
            "dataset_dir",
            "task_id",
            "caption",
            "target_key",
            "object",
            "category",
            "success",
            "pose_key",
            "scale",
            "position",
            "orientation",
        ],
    )
    write_csv(
        args.output_dir / "object_pose_summary.csv",
        summary_rows,
        [
            "object",
            "category",
            "pose_key",
            "scale",
            "position",
            "orientation",
            "attempts",
            "successes",
            "failures",
            "success_rate",
            "task_ids",
            "captions",
        ],
    )
    write_csv(
        args.output_dir / "failed_episodes.csv",
        failed_episode_rows,
        [
            "dataset_dir",
            "task_id",
            "caption",
            "target_key",
            "object",
            "category",
            "success",
            "pose_key",
            "scale",
            "position",
            "orientation",
        ],
    )

    with (args.output_dir / "ungraspable_pairs.json").open("w") as f:
        json.dump(ungraspable_rows, f, indent=2)

    with (args.output_dir / "ungraspable_pairs.txt").open("w") as f:
        for row in ungraspable_rows:
            f.write(
                f"{row['object']}\t{row['pose_key']}\t"
                f"success_rate={row['success_rate']:.3f}\t"
                f"attempts={row['attempts']}\t"
                f"failures={row['failures']}\n"
            )

    print(f"[INFO] episodes={len(all_records)}")
    print(f"[INFO] object_pose_pairs={len(summary_rows)}")
    print(
        f"[INFO] ungraspable_pairs={len(ungraspable_rows)} "
        f"(success_rate <= {args.failure_threshold}, min_attempts={args.min_attempts})"
    )
    print(f"[INFO] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

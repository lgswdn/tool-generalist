#!/usr/bin/env python3
"""Record fixed first episodes for a checkpoint sweep and build comparison grids.

Example:
    python scripts/compare_checkpoint_episodes.py --dry-run
    python scripts/compare_checkpoint_episodes.py --skip-complete

The default sweep targets the July 1 six-second no-contact run and compares
model_500.pt through model_2500.pt in a 5x1 per-episode MP4 grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_diff_post_oc_6s/no-contact/TCE/"
    "panda_general_diff_post_oc_6s/20260701T070044Z"
)
DEFAULT_START = 500
DEFAULT_STOP = 2500
DEFAULT_STEP = 500
DEFAULT_NUM_EPISODES = 15
DEFAULT_SEED = 0
DEFAULT_OBJECT_RANDOM_SEED = 0
DEFAULT_GRID_COLUMNS = 5
DEFAULT_GRID_ROWS = 1
DEFAULT_TILE_WIDTH = 512
DEFAULT_TILE_HEIGHT = 512
DEFAULT_VIDEO_FPS = 10
DEFAULT_IDENTITY_ABS_TOL = 1.0e-6
FFMPEG_PATH = Path("/usr/bin/ffmpeg")
FFPROBE_PATH = Path("/usr/bin/ffprobe")
RECORDER_SCRIPT = Path(__file__).resolve().with_name("record_fixed_episodes.py")
RECORDER_MANIFEST_NAME = "record_fixed_episodes_manifest.json"
RECORDER_LOG_NAME = "record_fixed_episodes.log"
TOP_MANIFEST_NAME = "fixed_episode_checkpoint_compare_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckpointPlan:
    step: int
    checkpoint: Path
    label: str
    video_dir: Path


def checkpoint_steps(start: int, stop: int, step: int) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    if start <= 0 or stop <= 0:
        raise ValueError("checkpoint start/stop must be positive")
    if stop < start:
        raise ValueError("checkpoint stop must be >= start")
    if (stop - start) % step != 0:
        raise ValueError("checkpoint range must be exactly divisible by step")
    return tuple(range(start, stop + 1, step))


def default_output_dir(
    run_dir: Path,
    *,
    num_episodes: int,
    seed: int,
    object_random_seed: int,
    start: int,
    stop: int,
    step: int,
) -> Path:
    return run_dir / (
        "fixed_episode_compare"
        f"_ep{num_episodes}"
        f"_ckpt{start}-{stop}-{step}"
        f"_seed{seed}"
        f"_objseed{object_random_seed}"
    )


def resolve_inputs(
    run_dir: Path,
    output_dir: Path,
    *,
    start: int,
    stop: int,
    step: int,
    grid_columns: int,
    grid_rows: int,
) -> tuple[Path, list[CheckpointPlan]]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    runtime_spec = run_dir / "rl_runtime_spec.json"
    if not runtime_spec.is_file():
        raise FileNotFoundError(f"Missing runtime spec: {runtime_spec}")

    steps = checkpoint_steps(start, stop, step)
    expected = grid_columns * grid_rows
    if len(steps) != expected:
        raise ValueError(
            f"Expected {expected} checkpoints for a {grid_columns}x{grid_rows} grid, got {len(steps)}"
        )

    plans: list[CheckpointPlan] = []
    for value in steps:
        checkpoint = run_dir / f"model_{value}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        label = checkpoint.stem
        plans.append(
            CheckpointPlan(
                step=value,
                checkpoint=checkpoint.resolve(),
                label=label,
                video_dir=output_dir / "checkpoints" / label,
            )
        )
    return runtime_spec.resolve(), plans


def build_recorder_command(
    *,
    python_executable: str,
    runtime_spec: Path,
    plan: CheckpointPlan,
    num_episodes: int,
    seed: int,
    object_random_seed: int,
    video_width: int,
    video_height: int,
    video_fps: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        python_executable,
        str(RECORDER_SCRIPT),
        "--runtime_spec",
        str(runtime_spec),
        "--checkpoint",
        str(plan.checkpoint),
        "--num_episodes",
        str(num_episodes),
        "--seed",
        str(seed),
        "--object_random_seed",
        str(object_random_seed),
        "--video_dir",
        str(plan.video_dir),
        "--checkpoint_label",
        plan.label,
        "--video_width",
        str(video_width),
        "--video_height",
        str(video_height),
        "--video_fps",
        str(video_fps),
        "--headless",
        *list(extra_args),
    ]


def build_recorder_environment(
    *,
    repo_root: Path,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    python_paths = [
        str(repo_root.resolve()),
        str((repo_root / "source" / "IsaacLab_nonPrehensile").resolve()),
    ]
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def validate_recorder_manifest(
    manifest_path: Path,
    *,
    plan: CheckpointPlan,
    runtime_spec: Path,
    num_episodes: int,
    seed: int,
    object_random_seed: int,
) -> list[dict[str, Any]]:
    manifest = _read_json_object(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Recorder manifest is not complete: {manifest_path}")
    _require_equal(Path(str(manifest.get("checkpoint"))).resolve(), plan.checkpoint, "checkpoint")
    _require_equal(Path(str(manifest.get("runtime_spec"))).resolve(), runtime_spec, "runtime_spec")
    _require_equal(int(manifest.get("num_episodes", -1)), int(num_episodes), "num_episodes")
    _require_equal(int(manifest.get("seed", -1)), int(seed), "seed")
    _require_equal(
        int(manifest.get("object_random_seed", -1)),
        int(object_random_seed),
        "object_random_seed",
    )

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != num_episodes:
        raise RuntimeError(
            f"Recorder manifest must contain {num_episodes} episodes: {manifest_path}"
        )

    records: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Episode entry {expected_index} is not an object: {manifest_path}")
        _require_equal(int(entry.get("episode_index", -1)), expected_index, "episode_index")
        metadata_path = Path(str(entry.get("metadata_path", ""))).expanduser()
        video_path = Path(str(entry.get("video_path", ""))).expanduser()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing episode metadata: {metadata_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing episode video: {video_path}")
        record = _read_json_object(metadata_path)
        if not isinstance(record.get("identity"), dict):
            raise RuntimeError(f"Episode metadata missing identity object: {metadata_path}")
        if not isinstance(record.get("outcome"), dict):
            raise RuntimeError(f"Episode metadata missing outcome object: {metadata_path}")
        _require_equal(int(record.get("episode_index", -1)), expected_index, "episode_index")
        records.append(record)
    return records


def verify_episode_identities(
    records_by_checkpoint: Sequence[tuple[CheckpointPlan, list[dict[str, Any]]]],
    *,
    abs_tol: float,
) -> list[dict[str, Any]]:
    if not records_by_checkpoint:
        raise ValueError("records_by_checkpoint must be non-empty")
    first_plan, first_records = records_by_checkpoint[0]
    identities: list[dict[str, Any]] = []
    for episode_index, base_record in enumerate(first_records):
        base_identity = base_record.get("identity")
        if not isinstance(base_identity, dict):
            raise RuntimeError(
                f"Base checkpoint {first_plan.label} episode {episode_index} is missing identity"
            )
        for plan, records in records_by_checkpoint[1:]:
            if episode_index >= len(records):
                raise RuntimeError(f"Checkpoint {plan.label} is missing episode {episode_index}")
            candidate = records[episode_index].get("identity")
            if not isinstance(candidate, dict):
                raise RuntimeError(
                    f"Checkpoint {plan.label} episode {episode_index} is missing identity"
                )
            compare_identity(base_identity, candidate, path=f"episode[{episode_index}]", abs_tol=abs_tol)
        identities.append(base_identity)
    return identities


def compare_identity(left: Any, right: Any, *, path: str = "identity", abs_tol: float) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise RuntimeError(f"Identity mismatch at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            raise RuntimeError(f"Non-finite identity value at {path}: {left!r}, {right!r}")
        if abs(float(left) - float(right)) > abs_tol:
            raise RuntimeError(f"Identity mismatch at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            missing_left = sorted(set(right).difference(left))
            missing_right = sorted(set(left).difference(right))
            raise RuntimeError(
                f"Identity keys differ at {path}: missing_left={missing_left} missing_right={missing_right}"
            )
        for key in sorted(left):
            compare_identity(left[key], right[key], path=f"{path}.{key}", abs_tol=abs_tol)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"Identity list length differs at {path}: {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            compare_identity(left_item, right_item, path=f"{path}[{index}]", abs_tol=abs_tol)
        return
    if left != right:
        raise RuntimeError(f"Identity mismatch at {path}: {left!r} != {right!r}")


def probe_video_duration(ffprobe_path: Path, video_path: Path) -> float:
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video_path} with code {result.returncode}: {result.stderr.strip()}"
        )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse ffprobe duration for {video_path}: {result.stdout!r}") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError(f"Invalid video duration for {video_path}: {duration!r}")
    return duration


def build_composite_ffmpeg_command(
    *,
    ffmpeg_path: Path,
    videos: Sequence[Path],
    labels: Sequence[str],
    durations: Sequence[float],
    output_path: Path,
    columns: int,
    tile_width: int,
    tile_height: int,
    fps: int,
) -> list[str]:
    if not videos:
        raise ValueError("videos must be non-empty")
    if len(videos) != len(labels) or len(videos) != len(durations):
        raise ValueError("videos, labels, and durations must have the same length")
    max_duration = max(float(value) for value in durations)
    filters: list[str] = []
    for index, (label, duration) in enumerate(zip(labels, durations, strict=True)):
        pad_seconds = max(max_duration - float(duration), 0.0) + 0.25
        safe_label = _escape_drawtext(label)
        filters.append(
            f"[{index}:v]"
            "setpts=PTS-STARTPTS,"
            f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
            f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={pad_seconds:.6f},"
            f"drawtext=text='{safe_label}':x=10:y=10:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.65"
            f"[v{index}]"
        )
    stacked_inputs = "".join(f"[v{index}]" for index in range(len(videos)))
    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * tile_height}"
        for index in range(len(videos))
    )
    filters.append(
        f"{stacked_inputs}xstack=inputs={len(videos)}:layout={layout}:shortest=1,"
        f"trim=duration={max_duration:.6f},setpts=PTS-STARTPTS[outv]"
    )

    cmd = [str(ffmpeg_path), "-y", "-loglevel", "error"]
    for video in videos:
        cmd.extend(["-i", str(video)])
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-r",
            str(fps),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    return cmd


def run_checked(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    print("+ " + shlex.join(str(part) for part in cmd), flush=True)
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        env=None if env is None else dict(env),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with code {result.returncode}: "
            f"{shlex.join(str(part) for part in cmd)}"
        )


def recorder_failure_message(
    *,
    plan: CheckpointPlan,
    returncode: int,
    manifest_path: Path,
    log_path: Path,
    max_log_chars: int = 12000,
) -> str:
    details = [
        f"Recorder failed for {plan.label} with code {returncode}.",
        f"Recorder log: {log_path}",
    ]
    if manifest_path.is_file():
        try:
            manifest = _read_json_object(manifest_path)
        except Exception as exc:
            details.append(f"Recorder manifest could not be read: {exc}")
        else:
            details.append(f"Recorder manifest status: {manifest.get('status')!r}")
            error = manifest.get("error")
            if error is not None:
                details.append(f"Recorder error: {error!r}")
    else:
        details.append(f"Recorder manifest was not created: {manifest_path}")
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if log_text:
            details.append("Recorder log tail:\n" + log_text[-max_log_chars:])
    return "\n".join(details)


def run_recorder_checked(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    plan: CheckpointPlan,
) -> None:
    plan.video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plan.video_dir / RECORDER_MANIFEST_NAME
    log_path = plan.video_dir / RECORDER_LOG_NAME
    manifest_path.unlink(missing_ok=True)
    command_text = shlex.join(str(part) for part in cmd)
    print("+ " + command_text, flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("+ " + command_text + "\n")
        log_stream.flush()
        try:
            process = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            log_stream.write(f"Recorder launch failed: {type(exc).__name__}: {exc}\n")
            log_stream.flush()
            raise RuntimeError(
                recorder_failure_message(
                    plan=plan,
                    returncode=-1,
                    manifest_path=manifest_path,
                    log_path=log_path,
                )
            ) from exc
        if process.stdout is None:
            process.kill()
            process.wait()
            raise RuntimeError("Recorder subprocess did not expose its combined output stream")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        returncode = process.wait()

    manifest_complete = False
    if manifest_path.is_file():
        try:
            manifest_complete = _read_json_object(manifest_path).get("status") == "complete"
        except Exception:
            manifest_complete = False
    if returncode != 0 or not manifest_complete:
        raise RuntimeError(
            recorder_failure_message(
                plan=plan,
                returncode=returncode,
                manifest_path=manifest_path,
                log_path=log_path,
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record fixed first episodes for model_500.pt..model_2500.pt and "
            "compose one 5x1 comparison MP4 per episode."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR, help="RL run directory.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Comparison output directory.")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="First checkpoint step.")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="Last checkpoint step.")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="Checkpoint step interval.")
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed Isaac environment seed.")
    parser.add_argument(
        "--object_random_seed",
        type=int,
        default=DEFAULT_OBJECT_RANDOM_SEED,
        help="Fixed generated-gripper/object assignment seed.",
    )
    parser.add_argument("--video_width", type=int, default=DEFAULT_TILE_WIDTH)
    parser.add_argument("--video_height", type=int, default=DEFAULT_TILE_HEIGHT)
    parser.add_argument("--video_fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--grid_columns", type=int, default=DEFAULT_GRID_COLUMNS)
    parser.add_argument("--grid_rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--identity_abs_tol", type=float, default=DEFAULT_IDENTITY_ABS_TOL)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG_PATH)
    parser.add_argument("--ffprobe", type=Path, default=FFPROBE_PATH)
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        default=False,
        help="Skip a checkpoint recorder only when its manifest validates complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Resolve inputs and print recorder commands without launching Isaac.",
    )
    parser.add_argument(
        "--recorder_arg",
        action="append",
        default=[],
        help="Extra single token appended to each recorder command. Repeat for multiple tokens.",
    )
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num_episodes must be positive")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.object_random_seed < 0:
        parser.error("--object_random_seed must be >= 0")
    if args.video_width <= 0 or args.video_height <= 0:
        parser.error("--video_width and --video_height must be positive")
    if args.video_fps <= 0:
        parser.error("--video_fps must be positive")
    if args.grid_columns <= 0 or args.grid_rows <= 0:
        parser.error("--grid_columns and --grid_rows must be positive")
    if args.identity_abs_tol < 0.0:
        parser.error("--identity_abs_tol must be >= 0")
    return args


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(
            run_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            object_random_seed=args.object_random_seed,
            start=args.start,
            stop=args.stop,
            step=args.step,
        ).resolve()
    )

    _require_file(args.ffmpeg, "ffmpeg")
    _require_file(args.ffprobe, "ffprobe")
    _require_file(RECORDER_SCRIPT, "fixed episode recorder")
    runtime_spec, plans = resolve_inputs(
        run_dir,
        output_dir,
        start=args.start,
        stop=args.stop,
        step=args.step,
        grid_columns=args.grid_columns,
        grid_rows=args.grid_rows,
    )

    commands = [
        build_recorder_command(
            python_executable=sys.executable,
            runtime_spec=runtime_spec,
            plan=plan,
            num_episodes=args.num_episodes,
            seed=args.seed,
            object_random_seed=args.object_random_seed,
            video_width=args.video_width,
            video_height=args.video_height,
            video_fps=args.video_fps,
            extra_args=args.recorder_arg,
        )
        for plan in plans
    ]
    recorder_environment = build_recorder_environment(repo_root=REPO_ROOT)

    print(f"Run dir: {run_dir}")
    print(f"Runtime spec: {runtime_spec}")
    print(f"Output dir: {output_dir}")
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        print("Dry run complete; Isaac recorder was not launched.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_checkpoint: list[tuple[CheckpointPlan, list[dict[str, Any]]]] = []
    for plan, command in zip(plans, commands, strict=True):
        manifest_path = plan.video_dir / RECORDER_MANIFEST_NAME
        if args.skip_complete and manifest_path.is_file():
            try:
                records = validate_recorder_manifest(
                    manifest_path,
                    plan=plan,
                    runtime_spec=runtime_spec,
                    num_episodes=args.num_episodes,
                    seed=args.seed,
                    object_random_seed=args.object_random_seed,
                )
            except Exception as exc:
                print(f"[INFO] Existing manifest is not skippable for {plan.label}: {exc}", flush=True)
            else:
                print(f"[INFO] Skipping complete recorder output for {plan.label}", flush=True)
                records_by_checkpoint.append((plan, records))
                continue

        run_recorder_checked(
            command,
            cwd=REPO_ROOT,
            env=recorder_environment,
            plan=plan,
        )
        records = validate_recorder_manifest(
            manifest_path,
            plan=plan,
            runtime_spec=runtime_spec,
            num_episodes=args.num_episodes,
            seed=args.seed,
            object_random_seed=args.object_random_seed,
        )
        records_by_checkpoint.append((plan, records))

    identities = verify_episode_identities(records_by_checkpoint, abs_tol=args.identity_abs_tol)
    composite_dir = output_dir / "episode_comparisons"
    composite_dir.mkdir(parents=True, exist_ok=True)
    episode_entries = []
    labels = [plan.label for plan, _records in records_by_checkpoint]
    for episode_index in range(args.num_episodes):
        videos = [
            Path(str(records[episode_index]["video"]["path"])).expanduser()
            for _plan, records in records_by_checkpoint
        ]
        for video in videos:
            if not video.is_file():
                raise FileNotFoundError(f"Missing episode video for composition: {video}")
        durations = [probe_video_duration(args.ffprobe, video) for video in videos]
        composite_path = composite_dir / f"episode_{episode_index:03d}_checkpoint_grid.mp4"
        ffmpeg_cmd = build_composite_ffmpeg_command(
            ffmpeg_path=args.ffmpeg,
            videos=videos,
            labels=labels,
            durations=durations,
            output_path=composite_path,
            columns=args.grid_columns,
            tile_width=args.video_width,
            tile_height=args.video_height,
            fps=args.video_fps,
        )
        run_checked(ffmpeg_cmd, cwd=REPO_ROOT)
        if not composite_path.is_file():
            raise FileNotFoundError(f"ffmpeg did not create composite: {composite_path}")
        episode_entries.append(
            {
                "episode_index": episode_index,
                "identity": identities[episode_index],
                "composite_path": str(composite_path),
                "tiles": [
                    {
                        "checkpoint": plan.label,
                        "checkpoint_path": str(plan.checkpoint),
                        "video_path": str(video),
                        "duration_seconds": duration,
                    }
                    for (plan, _records), video, duration in zip(
                        records_by_checkpoint, videos, durations, strict=True
                    )
                ],
            }
        )

    manifest = {
        "schema_version": "fixed_episode_checkpoint_compare_v1",
        "status": "complete",
        "run_dir": str(run_dir),
        "runtime_spec": str(runtime_spec),
        "output_dir": str(output_dir),
        "recorder_script": str(RECORDER_SCRIPT),
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "object_random_seed": args.object_random_seed,
        "identity_abs_tol": args.identity_abs_tol,
        "grid": {
            "columns": args.grid_columns,
            "rows": args.grid_rows,
            "tile_width": args.video_width,
            "tile_height": args.video_height,
        },
        "checkpoints": [
            {
                "step": plan.step,
                "label": plan.label,
                "checkpoint_path": str(plan.checkpoint),
                "recorder_manifest": str(plan.video_dir / RECORDER_MANIFEST_NAME),
            }
            for plan, _records in records_by_checkpoint
        ],
        "episodes": episode_entries,
    }
    _write_json(output_dir / TOP_MANIFEST_NAME, manifest)
    print(f"[INFO] Wrote comparison manifest: {output_dir / TOP_MANIFEST_NAME}", flush=True)


def _require_file(path: Path, label: str) -> None:
    if not path.expanduser().is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")


def _require_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise RuntimeError(f"Recorder manifest {label} mismatch: {left!r} != {right!r}")


def _escape_drawtext(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


if __name__ == "__main__":
    main()

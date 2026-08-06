from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.compare_checkpoint_episodes import (
    CheckpointPlan,
    build_composite_ffmpeg_command,
    build_recorder_command,
    build_recorder_environment,
    checkpoint_steps,
    compare_identity,
    recorder_failure_message,
    resolve_inputs,
    run_recorder_checked,
    validate_recorder_manifest,
)
from scripts.fixed_episode_runtime import backfill_legacy_fixed_episode_fields


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_checkpoint_steps_are_inclusive_and_strict():
    assert checkpoint_steps(500, 1500, 500) == (500, 1000, 1500)
    with pytest.raises(ValueError, match="divisible"):
        checkpoint_steps(500, 1600, 500)


def test_resolve_inputs_requires_every_checkpoint_for_grid(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rl_runtime_spec.json").write_text("{}", encoding="utf-8")
    for step in (500, 1000, 1500):
        (run_dir / f"model_{step}.pt").write_text("checkpoint", encoding="utf-8")

    runtime_spec, plans = resolve_inputs(
        run_dir,
        tmp_path / "out",
        start=500,
        stop=1500,
        step=500,
        grid_columns=3,
        grid_rows=1,
    )

    assert runtime_spec == (run_dir / "rl_runtime_spec.json").resolve()
    assert [plan.label for plan in plans] == ["model_500", "model_1000", "model_1500"]

    with pytest.raises(FileNotFoundError, match="model_2000.pt"):
        resolve_inputs(
            run_dir,
            tmp_path / "out",
            start=500,
            stop=2000,
            step=500,
            grid_columns=4,
            grid_rows=1,
        )


def test_compare_identity_ignores_outcome_by_only_comparing_identity_payload():
    left = {
        "env_id": 0,
        "object": {"index": 2, "scale_xyz": [0.15, 0.15, 0.15]},
        "target_pose": [0.1, 0.2, 0.3000001],
    }
    right = {
        "env_id": 0,
        "object": {"index": 2, "scale_xyz": [0.15, 0.15, 0.15]},
        "target_pose": [0.1, 0.2, 0.3000002],
    }
    compare_identity(left, right, abs_tol=1.0e-5)
    with pytest.raises(RuntimeError, match="object.index"):
        compare_identity(left, {**right, "object": {"index": 3, "scale_xyz": [0.15, 0.15, 0.15]}}, abs_tol=1.0e-5)


def test_build_recorder_command_is_headless_by_default(tmp_path):
    plan = CheckpointPlan(500, tmp_path / "model_500.pt", "model_500", tmp_path / "videos")
    cmd = build_recorder_command(
        python_executable="python",
        runtime_spec=tmp_path / "rl_runtime_spec.json",
        plan=plan,
        num_episodes=15,
        seed=0,
        object_random_seed=0,
        video_width=512,
        video_height=512,
        video_fps=10,
    )

    assert "--headless" in cmd


def test_build_recorder_environment_prepends_repo_paths_and_preserves_existing(tmp_path):
    repo_root = tmp_path / "repo"
    base_environment = {"PYTHONPATH": "/existing/one:/existing/two", "CUDA_VISIBLE_DEVICES": "0"}

    child_environment = build_recorder_environment(
        repo_root=repo_root,
        base_environment=base_environment,
    )

    assert child_environment["PYTHONPATH"] == (
        f"{repo_root.resolve()}:"
        f"{(repo_root / 'source' / 'IsaacLab_nonPrehensile').resolve()}:"
        "/existing/one:/existing/two"
    )
    assert child_environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert base_environment["PYTHONPATH"] == "/existing/one:/existing/two"


def test_backfill_legacy_july_spec_fields_is_exact_and_dimension_neutral():
    spec = {
        "observation_dim": 3129,
        "observation_params": {"model_input_centering": "object_center"},
        "policy_params": {},
        "object_pose_sampling_params": {
            "initial_position_range": 0.15,
            "xy_offset_range": 0.15,
        },
    }

    backfill_legacy_fixed_episode_fields(spec)

    assert spec["observation_params"] == {
        "model_input_centering": "object_center",
        "task_embedding_dim": 0,
    }
    assert spec["policy_params"] == {"task_embedding_dim": 0}
    assert spec["object_pose_sampling_params"] == {
        "initial_position_range": 0.15,
        "xy_offset_range": 0.15,
        "secondary_task": "random_pose",
        "grasp_lift_height": 0.05,
    }
    assert spec["observation_dim"] == 3129

    conflicting = {
        "observation_params": {"task_embedding_dim": 4},
        "policy_params": {"task_embedding_dim": 0},
        "object_pose_sampling_params": {
            "secondary_task": "random_pose",
            "grasp_lift_height": 0.05,
        },
    }
    with pytest.raises(RuntimeError, match="observation_params.task_embedding_dim"):
        backfill_legacy_fixed_episode_fields(conflicting)


def test_backfill_legacy_fields_preserves_valid_values_and_rejects_invalid_values():
    explicit = {
        "observation_params": {"task_embedding_dim": 0},
        "policy_params": {"task_embedding_dim": 0},
        "object_pose_sampling_params": {
            "secondary_task": "grasp_lift",
            "grasp_lift_height": 0.1,
        },
    }
    backfill_legacy_fixed_episode_fields(explicit)
    assert explicit["object_pose_sampling_params"] == {
        "secondary_task": "grasp_lift",
        "grasp_lift_height": 0.1,
    }

    invalid_task = {
        **explicit,
        "object_pose_sampling_params": {
            "secondary_task": "unknown",
            "grasp_lift_height": 0.1,
        },
    }
    with pytest.raises(RuntimeError, match="secondary_task"):
        backfill_legacy_fixed_episode_fields(invalid_task)

    invalid_height = {
        **explicit,
        "object_pose_sampling_params": {
            "secondary_task": "random_pose",
            "grasp_lift_height": 0.0,
        },
    }
    with pytest.raises(RuntimeError, match="grasp_lift_height"):
        backfill_legacy_fixed_episode_fields(invalid_height)


def test_recorder_failure_message_reports_manifest_error_and_log_tail(tmp_path):
    plan = CheckpointPlan(500, tmp_path / "model_500.pt", "model_500", tmp_path / "videos")
    plan.video_dir.mkdir()
    manifest_path = plan.video_dir / "record_fixed_episodes_manifest.json"
    log_path = plan.video_dir / "record_fixed_episodes.log"
    _write_json(
        manifest_path,
        '{"status":"failed","error":{"type":"AttributeError","message":"missing task_embedding_dim"}}',
    )
    log_path.write_text("live traceback: missing task_embedding_dim\n", encoding="utf-8")

    message = recorder_failure_message(
        plan=plan,
        returncode=1,
        manifest_path=manifest_path,
        log_path=log_path,
    )

    assert "Recorder failed for model_500 with code 1" in message
    assert str(log_path) in message
    assert "AttributeError" in message
    assert "live traceback: missing task_embedding_dim" in message


def test_recorder_zero_exit_without_manifest_is_a_logged_failure(tmp_path):
    plan = CheckpointPlan(500, tmp_path / "model_500.pt", "model_500", tmp_path / "videos")

    with pytest.raises(RuntimeError, match="Recorder manifest was not created") as exc_info:
        run_recorder_checked(
            [sys.executable, "-c", "print('root cause from recorder')"],
            cwd=tmp_path,
            env={},
            plan=plan,
        )

    log_path = plan.video_dir / "record_fixed_episodes.log"
    assert str(log_path) in str(exc_info.value)
    assert "root cause from recorder" in log_path.read_text(encoding="utf-8")
    assert "root cause from recorder" in str(exc_info.value)


def test_validate_recorder_manifest_requires_sidecar_and_video(tmp_path):
    runtime_spec = tmp_path / "run" / "rl_runtime_spec.json"
    checkpoint = tmp_path / "run" / "model_500.pt"
    runtime_spec.parent.mkdir()
    runtime_spec.write_text("{}", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    video = tmp_path / "out" / "episode_000_env_000.mp4"
    sidecar = tmp_path / "out" / "episode_000_env_000.json"
    video.parent.mkdir()
    video.write_text("video", encoding="utf-8")
    _write_json(
        sidecar,
        (
            '{"schema_version":"fixed_episode_record_v1","episode_index":0,'
            '"identity":{"env_id":0},"outcome":{"success":true},'
            f'"video":{{"path":"{video}"}}}}'
        ),
    )
    manifest = tmp_path / "out" / "record_fixed_episodes_manifest.json"
    _write_json(
        manifest,
        (
            '{"status":"complete","num_episodes":1,"seed":7,"object_random_seed":9,'
            f'"runtime_spec":"{runtime_spec.resolve()}","checkpoint":"{checkpoint.resolve()}",'
            f'"episodes":[{{"episode_index":0,"metadata_path":"{sidecar}","video_path":"{video}"}}]}}'
        ),
    )
    plan = CheckpointPlan(500, checkpoint.resolve(), "model_500", tmp_path / "out")

    records = validate_recorder_manifest(
        manifest,
        plan=plan,
        runtime_spec=runtime_spec.resolve(),
        num_episodes=1,
        seed=7,
        object_random_seed=9,
    )

    assert records[0]["identity"] == {"env_id": 0}
    video.unlink()
    with pytest.raises(FileNotFoundError, match="Missing episode video"):
        validate_recorder_manifest(
            manifest,
            plan=plan,
            runtime_spec=runtime_spec.resolve(),
            num_episodes=1,
            seed=7,
            object_random_seed=9,
        )


def test_build_composite_ffmpeg_command_pads_and_labels_tiles(tmp_path):
    videos = [tmp_path / f"episode_{idx}.mp4" for idx in range(3)]
    cmd = build_composite_ffmpeg_command(
        ffmpeg_path=Path("/usr/bin/ffmpeg"),
        videos=videos,
        labels=["model_500", "model_1000", "model_1500"],
        durations=[1.0, 2.0, 1.5],
        output_path=tmp_path / "grid.mp4",
        columns=3,
        tile_width=512,
        tile_height=512,
        fps=10,
    )

    joined = " ".join(cmd)
    assert joined.count(" -i ") == 3
    assert "tpad=stop_mode=clone:stop_duration=1.250000" in joined
    assert "drawtext=text='model_1000'" in joined
    assert "xstack=inputs=3:layout=0_0|512_0|1024_0:shortest=1" in joined

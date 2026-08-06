"""Experiment planner and stage runner.

This module deliberately does not import torch, Isaac, contact generation,
pretrain, or RL implementation modules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable

from configs.config_exp import ExpCfg
from utils.artifacts.manifest import ArtifactManifest, manifest_is_complete, read_manifest, write_manifest
from utils.artifacts.resolver import ArtifactRef, ResolvedArtifacts, resolve_artifacts
from utils.config.hash import to_plain_data
from utils.config.loader import load_exp_cfg
from utils.config.paths import ProjectPaths, load_project_paths
from utils.experiment.effective_paths import apply_experiment_path_overrides
from utils.experiment.eval_curriculum import (
    latest_eval_checkpoint,
    latest_eval_objects_summary,
    materialize_curriculum_object_manifest,
    normalize_success_rate_threshold,
)
from utils.experiment.validation import validate_for_plan, validate_for_run


@dataclass(frozen=True)
class ExperimentRun:
    cfg: ExpCfg
    paths: ProjectPaths
    artifacts: ResolvedArtifacts
    manifests: tuple[Path, ...]
    mode: str = "plan"
    stage_results: dict[str, Any] | None = None
    resolved_encoder_checkpoint: str | None = None


def run_from_config(
    config: str | Path,
    *,
    curriculum_from_eval: bool = False,
    curriculum_success_rate_threshold: float = 0.5,
    curriculum_resume_from_eval: bool = True,
    runtime_num_gpus: int | None = None,
    runtime_total_envs: int = 8192,
    runtime_print_fine_grained_timing: bool = False,
) -> ExperimentRun:
    cfg = load_exp_cfg(config)
    return run_experiment(
        cfg,
        config_source=str(config),
        curriculum_from_eval=curriculum_from_eval,
        curriculum_success_rate_threshold=curriculum_success_rate_threshold,
        curriculum_resume_from_eval=curriculum_resume_from_eval,
        runtime_num_gpus=runtime_num_gpus,
        runtime_total_envs=runtime_total_envs,
        runtime_print_fine_grained_timing=runtime_print_fine_grained_timing,
    )


def plan_from_config(config: str | Path) -> ExperimentRun:
    cfg = load_exp_cfg(config)
    return plan_experiment(cfg, config_source=str(config))


def plan_experiment(cfg: ExpCfg, *, config_source: str | None = None) -> ExperimentRun:
    run = _prepare_experiment(cfg, config_source=config_source, mode="plan")
    return ExperimentRun(
        cfg=run.cfg,
        paths=run.paths,
        artifacts=run.artifacts,
        manifests=(),
        mode="plan",
        stage_results={},
        resolved_encoder_checkpoint=run.resolved_encoder_checkpoint,
    )


def run_experiment(
    cfg: ExpCfg,
    *,
    config_source: str | None = None,
    curriculum_from_eval: bool = False,
    curriculum_success_rate_threshold: float = 0.5,
    curriculum_resume_from_eval: bool = True,
    runtime_num_gpus: int | None = None,
    runtime_total_envs: int = 8192,
    runtime_print_fine_grained_timing: bool = False,
) -> ExperimentRun:
    _log_event("RUN", f"experiment={cfg.name} config={config_source or '<in-memory>'} mode=run")
    if curriculum_from_eval:
        threshold = normalize_success_rate_threshold(curriculum_success_rate_threshold)
        _log_event("CURR", f"source=latest_eval_objects threshold={threshold:.4f}")
    runtime_rl = _runtime_rl_override(runtime_num_gpus, runtime_total_envs)
    if runtime_rl is not None:
        _log_event(
            "RUN",
            f"runtime_num_gpus={runtime_rl['num_gpus']} "
            f"runtime_total_envs={runtime_rl['total_envs']} "
            f"runtime_envs_per_gpu={runtime_rl['num_envs_per_gpu']}",
        )
    run = _prepare_experiment(
        cfg,
        config_source=config_source,
        mode="run",
        runtime_rl=runtime_rl,
    )
    _log_event("RUN", f"paths_yaml={run.paths.source_yaml}")
    stage_results: dict[str, Any] = {}
    manifests: list[Path] = []
    resolved_encoder_checkpoint = run.resolved_encoder_checkpoint
    manifests.append(
        _write_stage_manifest(
            cfg,
            run.artifacts.experiment,
            config_source,
            run.paths,
            mode="run",
            status="running",
            action="run",
        )
    )

    for ref in run.artifacts.stages:
        stage_paths = apply_experiment_path_overrides(cfg, run.paths, stage=ref.stage)
        action = _stage_action(cfg, ref)
        if action == "skipped":
            _log_stage("SKIP", ref, action=action)
            stage_results[ref.stage] = _stage_record(ref, status="skipped", action=action)
            manifests.append(
                _write_stage_manifest(
                    cfg,
                    ref,
                    config_source,
                    stage_paths,
                    mode="run",
                    status="skipped",
                    action=action,
                )
            )
            continue
        if action == "reused":
            _log_stage("REUSE", ref, action=action)
            stage_results[ref.stage] = _stage_record(ref, status="complete", action=action, reused=True)
            resolved_encoder_checkpoint = _resolve_stage_encoder_checkpoint_from_manifest(
                resolved_encoder_checkpoint,
                ref,
            )
            if manifest_is_complete(ref.manifest_path):
                manifests.append(ref.manifest_path)
            else:
                manifests.append(
                    _write_stage_manifest(
                        cfg,
                        ref,
                        config_source,
                        stage_paths,
                        mode="run",
                        status="complete",
                        action=action,
                        reused=True,
                        result={"reused_from_existing_outputs": True},
                        resolved_encoder_checkpoint=resolved_encoder_checkpoint,
                    )
                )
            continue
        if not ref.entrypoint:
            raise RuntimeError(f"Stage {ref.stage!r} has no entrypoint")
        _log_stage("RUN", ref, action=action)
        manifests.append(
            _write_stage_manifest(
                cfg,
                ref,
                config_source,
                stage_paths,
                mode="run",
                status="running",
                action="run",
                executed=False,
                resolved_encoder_checkpoint=resolved_encoder_checkpoint,
            )
        )
        kwargs: dict[str, Any] = {}
        if ref.stage == "rl":
            kwargs["resolved_encoder_checkpoint"] = resolved_encoder_checkpoint
            kwargs["runtime_print_fine_grained_timing"] = bool(
                runtime_print_fine_grained_timing
            )
            if runtime_rl is not None:
                kwargs["runtime_num_gpus"] = runtime_rl["num_gpus"]
                kwargs["runtime_num_envs"] = runtime_rl["num_envs_per_gpu"]
            if curriculum_from_eval:
                threshold = normalize_success_rate_threshold(curriculum_success_rate_threshold)
                runtime_objects_manifest = materialize_curriculum_object_manifest(
                    stage_paths,
                    ref.directory,
                    threshold=threshold,
                )
                eval_summary = latest_eval_objects_summary(ref.directory)
                resume_checkpoint = latest_eval_checkpoint(ref.directory) if curriculum_resume_from_eval else None
                _log_event(
                    "CURR",
                    f"eval_summary={eval_summary} selected_manifest={runtime_objects_manifest} "
                    f"resume_checkpoint={resume_checkpoint or '<disabled>'} threshold={threshold:.4f}",
                )
                kwargs["runtime_objects_manifest"] = runtime_objects_manifest
                if resume_checkpoint is not None:
                    kwargs["runtime_rl_resume_checkpoint"] = resume_checkpoint
        try:
            _log_event("START", f"stage={ref.stage} entrypoint={ref.entrypoint}")
            entrypoint = _load_entrypoint(ref.entrypoint)
            result = _call_stage(entrypoint, cfg, stage_paths, ref.directory, kwargs=kwargs)
            _validate_stage_result(ref, result)
            if _contact_result_is_partial_shard(ref, result):
                _log_event("PARTIAL", f"stage={ref.stage} {_stage_result_summary(ref, result)}")
                stage_results[ref.stage] = _stage_record(
                    ref,
                    status="partial",
                    action="run",
                    executed=True,
                    result=result,
                )
                manifests.append(
                    _write_stage_manifest(
                        cfg,
                        ref,
                        config_source,
                        stage_paths,
                        mode="run",
                        status="partial",
                        action="run",
                        executed=True,
                        result=result,
                        resolved_encoder_checkpoint=resolved_encoder_checkpoint,
                    )
                )
                manifests.append(
                    _write_stage_manifest(
                        cfg,
                        run.artifacts.experiment,
                        config_source,
                        run.paths,
                        mode="run",
                        status="partial",
                        action="run",
                        executed=True,
                        result={"stages": stage_results},
                        resolved_encoder_checkpoint=resolved_encoder_checkpoint,
                    )
                )
                return ExperimentRun(
                    cfg=run.cfg,
                    paths=run.paths,
                    artifacts=run.artifacts,
                    manifests=tuple(manifests),
                    mode="run",
                    stage_results=stage_results,
                    resolved_encoder_checkpoint=resolved_encoder_checkpoint,
                )
        except Exception as exc:
            _log_event("FAIL", f"stage={ref.stage} error={repr(exc)}")
            stage_results[ref.stage] = _stage_record(
                ref,
                status="failed",
                action="run",
                executed=True,
                error=repr(exc),
            )
            manifests.append(
                _write_stage_manifest(
                    cfg,
                    ref,
                    config_source,
                    stage_paths,
                    mode="run",
                    status="failed",
                    action="run",
                    executed=True,
                    error=repr(exc),
                    resolved_encoder_checkpoint=resolved_encoder_checkpoint,
                )
            )
            _write_stage_manifest(
                cfg,
                run.artifacts.experiment,
                config_source,
                run.paths,
                mode="run",
                status="failed",
                action="run",
                executed=True,
                result={"failed_stage": ref.stage, "error": repr(exc)},
                resolved_encoder_checkpoint=resolved_encoder_checkpoint,
            )
            raise
        resolved_encoder_checkpoint = _resolve_stage_encoder_checkpoint(
            resolved_encoder_checkpoint,
            stage=result,
            stage_name=ref.stage,
        )
        _log_event("DONE", f"stage={ref.stage} {_stage_result_summary(ref, result)}")
        stage_results[ref.stage] = _stage_record(
            ref,
            status="complete",
            action="run",
            executed=True,
            result=result,
        )
        manifests.append(
            _write_stage_manifest(
                cfg,
                ref,
                config_source,
                stage_paths,
                mode="run",
                status="complete",
                action="run",
                executed=True,
                result=result,
                resolved_encoder_checkpoint=resolved_encoder_checkpoint,
            )
        )

    manifests.append(
        _write_stage_manifest(
            cfg,
            run.artifacts.experiment,
            config_source,
            run.paths,
            mode="run",
            status="complete",
            action="run",
            executed=True,
            result={"stages": stage_results},
            resolved_encoder_checkpoint=resolved_encoder_checkpoint,
        )
    )
    _log_event("DONE", f"experiment={cfg.name} status=complete")

    return ExperimentRun(
        cfg=run.cfg,
        paths=run.paths,
        artifacts=run.artifacts,
        manifests=tuple(manifests),
        mode="run",
        stage_results=stage_results,
        resolved_encoder_checkpoint=resolved_encoder_checkpoint,
    )


def _prepare_experiment(
    cfg: ExpCfg,
    *,
    config_source: str | None,
    mode: str,
    runtime_rl: dict[str, int] | None = None,
) -> ExperimentRun:
    paths = apply_experiment_path_overrides(
        cfg,
        load_project_paths(cfg.paths_yaml),
    )
    validation_cfg = _cfg_for_runtime_validation(cfg, runtime_rl)
    if mode == "plan":
        validate_for_plan(
            validation_cfg,
            paths,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    elif mode == "run":
        validate_for_run(
            validation_cfg,
            paths,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    else:
        raise ValueError(f"Unknown experiment mode: {mode!r}")
    artifacts = resolve_artifacts(cfg)
    return ExperimentRun(
        cfg=cfg,
        paths=paths,
        artifacts=artifacts,
        manifests=(),
        mode=mode,
        stage_results={},
        resolved_encoder_checkpoint=_resolve_initial_encoder_checkpoint(
            cfg,
            config_source=config_source,
        ),
    )


def format_summary(run: ExperimentRun) -> str:
    lines = [
        f"experiment: {run.cfg.name}",
        f"config_hash: {run.artifacts.experiment.config_hash}",
        f"mode: {run.mode}",
        f"paths_yaml: {run.paths.source_yaml}",
    ]
    if run.resolved_encoder_checkpoint:
        lines.append(f"resolved_encoder_checkpoint: {run.resolved_encoder_checkpoint}")
    for ref in (run.artifacts.experiment, *run.artifacts.stages):
        record = (run.stage_results or {}).get(ref.stage, {})
        executed = bool(record.get("executed", False)) if isinstance(record, dict) else False
        status = record.get("status", ref.status) if isinstance(record, dict) else ref.status
        action = record.get("action", ref.action) if isinstance(record, dict) else ref.action
        reused = bool(record.get("reused", False)) if isinstance(record, dict) else False
        lines.append(
            f"{ref.stage}: status={status} action={action} "
            f"executed={executed} reused={reused} artifact={ref.directory}"
        )
    return "\n".join(lines)


def _write_stage_manifest(
    cfg: ExpCfg,
    ref: ArtifactRef,
    config_source: str | None,
    paths: ProjectPaths,
    *,
    mode: str,
    status: str | None = None,
    action: str | None = None,
    executed: bool = False,
    reused: bool = False,
    result: Any = None,
    error: str | None = None,
    resolved_encoder_checkpoint: str | None = None,
) -> Path:
    status = status or ref.status
    action = action or ref.action
    manifest = ArtifactManifest(
        artifact_type=ref.artifact_type,
        artifact_name=ref.artifact_name,
        exp_cfg_name=cfg.name,
        config_hash=ref.config_hash,
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_paths=_source_paths(paths, config_source),
        config_dump=to_plain_data(cfg),
        metrics={
            "stage": ref.stage,
            "enabled": ref.enabled,
            "requested": ref.requested,
            "required": ref.required,
            "dependency_reason": ref.dependency_reason,
            "action": action,
            "executed": executed,
            "reused": reused,
            "entrypoint": ref.entrypoint,
            "result": _json_safe(result),
            "error": error,
        },
        runtime={
            "mode": mode,
            "stage": ref.stage,
            "resolved_encoder_checkpoint": resolved_encoder_checkpoint,
        },
    )
    return write_manifest(ref.directory, manifest)


def _stage_action(cfg: ExpCfg, ref: ArtifactRef) -> str:
    if not ref.required:
        return "skipped"
    if cfg.artifact_policy == "fail-if-exists" and (
        ref.directory.exists() or ref.manifest_path.exists()
    ):
        raise RuntimeError(
            f"Artifact for stage {ref.stage!r} already exists under fail-if-exists: "
            f"{ref.directory}"
        )
    if _stage_forces_run(cfg, ref):
        return "run"
    if cfg.artifact_policy == "reuse" and _stage_is_reusable(ref):
        return "reused"
    return "run"


def _stage_forces_run(cfg: ExpCfg, ref: ArtifactRef) -> bool:
    if ref.stage == "contact_gen":
        return bool(cfg.contact_gen.regenerate or cfg.contact_gen.shard_count > 1)
    if ref.stage == "pretrain":
        return bool(cfg.pretrain.retrain)
    return False


def _stage_is_reusable(ref: ArtifactRef) -> bool:
    if manifest_is_complete(ref.manifest_path):
        if ref.stage == "contact_gen":
            return _contact_manifest_result_is_complete(ref.manifest_path)
        return True
    if ref.manifest_path.exists():
        return False
    if ref.stage == "contact_gen":
        return False
    return False


def _contact_manifest_result_is_complete(manifest_path: Path) -> bool:
    try:
        payload = read_manifest(manifest_path)
    except Exception:
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    result = metrics.get("result")
    if not isinstance(result, dict):
        return False
    ok = _result_count(result, "ok") or 0
    skipped = _result_count(result, "skipped") or 0
    return ok + skipped > 0


def _validate_stage_result(ref: ArtifactRef, result: Any) -> None:
    if _contact_result_is_partial_shard(ref, result):
        return
    if ref.stage == "contact_gen" and _empty_failed_contact_result(result):
        ok = _result_count(result, "ok")
        skipped = _result_count(result, "skipped")
        fail = _result_count(result, "fail")
        raise RuntimeError(
            "Contact generation produced no usable outputs "
            f"(ok={ok}, skipped={skipped}, fail={fail}); refusing to mark artifact complete"
        )


def _empty_failed_contact_result(result: Any) -> bool:
    ok = _result_count(result, "ok")
    skipped = _result_count(result, "skipped")
    fail = _result_count(result, "fail")
    return ok == 0 and skipped == 0


def _contact_result_is_partial_shard(ref: ArtifactRef, result: Any) -> bool:
    if ref.stage != "contact_gen":
        return False
    shard_count = _result_count(result, "shard_count")
    return shard_count is not None and shard_count > 1


def _result_count(result: Any, key: str) -> int | None:
    value: Any = None
    if isinstance(result, dict):
        value = result.get(key)
        if value is None and isinstance(result.get("summary"), dict):
            value = result["summary"].get(key)
    else:
        value = getattr(result, key, None)
        summary = getattr(result, "summary", None)
        if value is None and isinstance(summary, dict):
            value = summary.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stage_record(
    ref: ArtifactRef,
    *,
    status: str,
    action: str,
    executed: bool = False,
    reused: bool = False,
    result: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": ref.stage,
        "status": status,
        "action": action,
        "executed": executed,
        "reused": reused,
        "artifact": str(ref.directory),
        "requested": ref.requested,
        "required": ref.required,
        "dependency_reason": ref.dependency_reason,
        "result": _json_safe(result),
        "error": error,
    }


def _log_stage(event: str, ref: ArtifactRef, *, action: str) -> None:
    _log_event(event, f"stage={ref.stage} action={action} artifact={ref.directory}")


def _log_event(event: str, message: str) -> None:
    print(f"[{_log_timestamp()}] [{event}] {message}", flush=True)


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stage_result_summary(ref: ArtifactRef, result: Any) -> str:
    if ref.stage == "contact_gen":
        keys = (
            "num_pairs",
            "global_num_pairs",
            "num_poses",
            "ok",
            "fail",
            "skipped",
            "shard_index",
            "shard_count",
            "artifact_dir",
        )
        return _format_result_fields(result, keys)
    if isinstance(result, dict):
        keys = (
            "best_checkpoint_path",
            "checkpoint_path",
            "checkpoint_dir",
            "runtime_spec_path",
            "launched",
            "returncode",
        )
        summary = _format_result_fields(result, keys)
        return summary if summary else f"result={_json_safe(result)}"
    return _format_result_fields(
        result,
        (
            "best_checkpoint_path",
            "checkpoint_path",
            "checkpoint_dir",
            "runtime_spec_path",
            "launched",
            "returncode",
        ),
    ) or f"result={_json_safe(result)}"


def _format_result_fields(result: Any, keys: tuple[str, ...]) -> str:
    fields: list[str] = []
    for key in keys:
        value = _result_value(result, key)
        if value is not None:
            fields.append(f"{key}={value}")
    return " ".join(fields)


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        value = result.get(key)
        if value is None and isinstance(result.get("summary"), dict):
            value = result["summary"].get(key)
        return value
    return getattr(result, key, None)


def _load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    module_name, sep, attr = entrypoint.partition(":")
    if not sep or not module_name or not attr:
        raise RuntimeError(f"Invalid stage entrypoint string: {entrypoint!r}")
    module = importlib.import_module(module_name)
    func = getattr(module, attr)
    if not callable(func):
        raise RuntimeError(f"Stage entrypoint is not callable: {entrypoint!r}")
    return func


def _call_stage(
    entrypoint: Callable[..., Any],
    cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: Path,
    *,
    kwargs: dict[str, Any],
) -> Any:
    accepted = set(inspect.signature(entrypoint).parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return entrypoint(cfg, paths, artifact_dir, **filtered_kwargs)


def _resolve_initial_encoder_checkpoint(
    cfg: ExpCfg,
    *,
    config_source: str | None = None,
    seen_reuse: set[str] | None = None,
) -> str | None:
    model = cfg.model
    return (
        model.pretrained_encoder.checkpoint_path
        or model.encoder.checkpoint_path
        or cfg.rl.encoder_checkpoint
        or _resolve_pretrain_reuse_checkpoint(
            cfg,
            config_source=config_source,
            seen_reuse=seen_reuse,
        )
    )


def _resolve_pretrain_reuse_checkpoint(
    cfg: ExpCfg,
    *,
    config_source: str | None = None,
    seen_reuse: set[str] | None = None,
) -> str | None:
    reuse = cfg.pretrain_reuse
    if not reuse:
        return None

    reuse_config = _resolve_reuse_config_ref(reuse, config_source)
    reuse_key = str(Path(reuse_config).resolve()) if _looks_like_path(reuse_config) else str(reuse_config)
    seen = set() if seen_reuse is None else set(seen_reuse)
    if reuse_key in seen:
        raise RuntimeError(f"Cyclic ExpCfg.pretrain_reuse reference: {reuse_key}")
    seen.add(reuse_key)

    reuse_cfg = load_exp_cfg(reuse_config)
    pretrain_ref = _pretrain_artifact_ref(reuse_cfg)
    if pretrain_ref is not None:
        if manifest_is_complete(pretrain_ref.manifest_path):
            resolved = _resolve_stage_encoder_checkpoint_from_manifest(None, pretrain_ref)
            if resolved:
                return resolved
        best_checkpoint = pretrain_ref.directory / "best.pt"
        if best_checkpoint.exists():
            return str(best_checkpoint)

    resolved = _resolve_initial_encoder_checkpoint(
        reuse_cfg,
        config_source=str(reuse_config),
        seen_reuse=seen,
    )
    if resolved:
        return resolved

    details = []
    if pretrain_ref is not None:
        details.append(f"manifest={pretrain_ref.manifest_path}")
        details.append(f"best={pretrain_ref.directory / 'best.pt'}")
    raise RuntimeError(
        f"Could not resolve a pretrained encoder checkpoint from "
        f"ExpCfg.pretrain_reuse={reuse!r} resolved_config={reuse_config!r} "
        + " ".join(details)
    )


def _resolve_reuse_config_ref(reuse: str, config_source: str | None) -> str | Path:
    path = Path(reuse).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.suffix == ".py":
        candidates: list[Path] = []
        if config_source:
            source = Path(config_source).expanduser()
            if source.suffix == ".py" or source.exists():
                candidates.append(source.resolve().parent / path)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                Path.cwd() / path,
                repo_root / "configs" / "experiments" / path,
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return path
    return reuse


def _looks_like_path(value: str | Path) -> bool:
    if isinstance(value, Path):
        return True
    return Path(value).suffix == ".py" or os.sep in value


def _pretrain_artifact_ref(cfg: ExpCfg) -> ArtifactRef | None:
    for ref in resolve_artifacts(cfg).stages:
        if ref.stage == "pretrain":
            return ref
    return None


def _resolve_stage_encoder_checkpoint(
    current: str | None,
    *,
    stage: Any,
    stage_name: str,
) -> str | None:
    if stage_name != "pretrain":
        return current
    if isinstance(stage, dict):
        for key in ("best_checkpoint_path", "best_checkpoint", "checkpoint_path"):
            value = stage.get(key)
            if value:
                return str(value)
        checkpoint_dir = stage.get("checkpoint_dir")
        if checkpoint_dir:
            return str(Path(checkpoint_dir) / "best.pt")
    for key in ("best_checkpoint_path", "best_checkpoint", "checkpoint_path"):
        value = getattr(stage, key, None)
        if value:
            return str(value)
    checkpoint_dir = getattr(stage, "checkpoint_dir", None)
    if checkpoint_dir:
        return str(Path(checkpoint_dir) / "best.pt")
    return current


def _resolve_stage_encoder_checkpoint_from_manifest(
    current: str | None,
    ref: ArtifactRef,
) -> str | None:
    if ref.stage != "pretrain":
        return current
    try:
        payload = read_manifest(ref.manifest_path)
    except Exception:
        return current
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        result = metrics.get("result")
        resolved = _resolve_stage_encoder_checkpoint(current, stage=result, stage_name=ref.stage)
        if resolved:
            return resolved
    runtime = payload.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("resolved_encoder_checkpoint"):
        return str(runtime["resolved_encoder_checkpoint"])
    if ref.stage == "pretrain":
        return str(ref.directory / "best.pt")
    return current


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        return to_plain_data(value)
    except Exception:
        return repr(value)


def _source_paths(paths: ProjectPaths, config_source: str | None) -> dict[str, str]:
    values: dict[str, str] = {
        "config": config_source or "",
        "paths_yaml": str(paths.source_yaml),
    }
    for key, path in paths.values.items():
        if path is not None:
            values[key] = str(path)
    return values


def _runtime_rl_override(runtime_num_gpus: int | None, runtime_total_envs: int) -> dict[str, int] | None:
    if runtime_num_gpus is None:
        return None
    num_gpus = int(runtime_num_gpus)
    total_envs = int(runtime_total_envs)
    if num_gpus <= 0:
        raise RuntimeError(f"runtime_num_gpus must be positive, got {runtime_num_gpus!r}")
    if total_envs <= 0:
        raise RuntimeError(f"runtime_total_envs must be positive, got {runtime_total_envs!r}")
    if total_envs % num_gpus != 0:
        raise RuntimeError(
            f"runtime_total_envs ({total_envs}) must be divisible by runtime_num_gpus ({num_gpus})"
        )
    return {
        "num_gpus": num_gpus,
        "total_envs": total_envs,
        "num_envs_per_gpu": total_envs // num_gpus,
    }


def _cfg_for_runtime_validation(cfg: ExpCfg, runtime_rl: dict[str, int] | None) -> ExpCfg:
    if runtime_rl is None:
        return cfg
    validation_cfg = deepcopy(cfg)
    validation_cfg.num_gpus = runtime_rl["num_gpus"]
    validation_cfg.rl.env.num_envs = runtime_rl["num_envs_per_gpu"]
    validation_cfg.rl.launch.distributed = runtime_rl["num_gpus"] > 1
    return validation_cfg

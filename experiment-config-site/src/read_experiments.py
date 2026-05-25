#!/usr/bin/env python3
"""Read experiment configs and emit JSON for the Node backend."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def get_nested(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def first_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = get_nested(data, key)
        if value is not None:
            return value
    return default


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    replacements = {
        "True": True,
        "False": False,
        "None": None,
    }
    if raw in replacements:
        return replacements[raw]
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def parse_assignments(source: str) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*EXP_CFG(?:\.([A-Za-z_][\w.]*))?\s*=\s*(.+?)\s*$")
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        path = match.group(1) or "$root"
        raw_value = match.group(2).split("#", 1)[0].strip()
        assignments.append(
            {
                "line": line_number,
                "path": path,
                "raw": raw_value,
                "value": parse_value(raw_value),
            }
        )
    return assignments


def pretrain_reuse_target(
    config: dict[str, Any],
    reuse_resolution: dict[str, Any] | None = None,
) -> str | None:
    if reuse_resolution and reuse_resolution.get("target"):
        return str(reuse_resolution["target"])
    value = get_nested(config, "pretrain_reuse")
    return str(value) if value else None


def stage_enabled(
    config: dict[str, Any],
    stage: str,
    reuse_resolution: dict[str, Any] | None = None,
) -> bool:
    if stage == "contact_gen" and pretrain_reuse_target(config, reuse_resolution):
        return False
    if stage == "pretrain" and pretrain_reuse_target(config, reuse_resolution):
        return False
    return bool(get_nested(config, f"{stage}.enabled", False))


def stage_status(
    config: dict[str, Any],
    stage: str,
    reuse_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direct_reuse = get_nested(config, "pretrain_reuse")
    reuse = pretrain_reuse_target(config, reuse_resolution)
    requested = bool(get_nested(config, f"{stage}.enabled", False))
    enabled = stage_enabled(config, stage, reuse_resolution)
    if stage in {"contact_gen", "pretrain"} and direct_reuse:
        chain = reuse_resolution.get("chain", []) if reuse_resolution else []
        return {
            "enabled": False,
            "requested": requested,
            "status": "reused",
            "statusText": f"复用 {reuse}",
            "reason": "pretrain_reuse",
            "reuseTarget": reuse,
            "reuseDirect": str(direct_reuse),
            "reuseChain": chain,
        }
    return {
        "enabled": enabled,
        "requested": requested,
        "status": "enabled" if enabled else "skipped",
        "statusText": "启动" if enabled else "跳过",
        "reason": None,
        "reuseTarget": None,
        "reuseDirect": None,
        "reuseChain": [],
    }


def compact_params(config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for dotted in keys:
        value = get_nested(config, dotted)
        if value is not None:
            params[dotted] = value
    return params


def display_config_ref(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    experiments_dir = repo_root / "configs" / "experiments"
    try:
        return path.resolve().relative_to(experiments_dir.resolve()).name
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def stage_ref_summary(ref: Any | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "stage": getattr(ref, "stage", None),
        "artifactType": getattr(ref, "artifact_type", None),
        "artifactName": getattr(ref, "artifact_name", None),
        "path": str(getattr(ref, "directory", "")),
        "manifestPath": str(getattr(ref, "manifest_path", "")),
        "enabled": getattr(ref, "enabled", None),
        "status": getattr(ref, "status", None),
    }


def artifact_stage_ref(cfg_obj: Any, stage_name: str) -> Any | None:
    from utils.artifacts.resolver import resolve_artifacts

    for ref in resolve_artifacts(cfg_obj).stages:
        if ref.stage == stage_name:
            return ref
    return None


def resolve_paths_yaml(repo_root: pathlib.Path, paths_yaml: str | None) -> pathlib.Path | None:
    if not paths_yaml:
        return None
    path = pathlib.Path(paths_yaml).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_dataset_manifest_path(
    repo_root: pathlib.Path,
    cfg_obj: Any,
    dataset_manifest: str | None,
) -> pathlib.Path | None:
    if not dataset_manifest:
        return None
    path = pathlib.Path(dataset_manifest).expanduser()
    if path.is_absolute():
        return path
    paths_yaml = resolve_paths_yaml(repo_root, getattr(cfg_obj, "paths_yaml", None))
    if paths_yaml:
        return (paths_yaml.parent / path).resolve()
    return (repo_root / path).resolve()


def looks_like_contact_artifact(path: pathlib.Path) -> bool:
    manifest = path / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload.get("artifact_type") == "contact" and payload.get("status") in {"complete", "found"}:
            return True
    try:
        next(path.rglob("*.pt.manifest.json"))
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def latest_existing_contact_artifact(parent: pathlib.Path) -> pathlib.Path | None:
    if not parent.exists():
        return None
    try:
        candidates = [
            path
            for path in parent.iterdir()
            if path.is_dir() and looks_like_contact_artifact(path)
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def contact_data_source(
    repo_root: pathlib.Path,
    cfg_obj: Any,
    config_id: str,
) -> dict[str, Any]:
    dataset_manifest = getattr(getattr(cfg_obj, "pretrain", None), "dataset_manifest", None)
    contact_ref = artifact_stage_ref(cfg_obj, "contact_gen")
    planned_path = pathlib.Path(contact_ref.directory) if contact_ref is not None else None

    if dataset_manifest:
        actual = resolve_dataset_manifest_path(repo_root, cfg_obj, str(dataset_manifest))
        return {
            "config": config_id,
            "source": "dataset_manifest",
            "datasetManifest": str(actual) if actual else str(dataset_manifest),
            "path": str(actual) if actual else str(dataset_manifest),
            "exists": bool(actual and actual.exists()),
            "plannedContact": stage_ref_summary(contact_ref),
        }

    actual_path = planned_path
    source = "planned_contact_artifact"
    inferred_from: str | None = None
    if planned_path is not None and not planned_path.exists():
        inferred = latest_existing_contact_artifact(planned_path.parent)
        if inferred is not None:
            actual_path = inferred
            source = "inferred_existing_sibling"
            inferred_from = str(planned_path.parent)

    return {
        "config": config_id,
        "source": source,
        "path": str(actual_path) if actual_path else None,
        "expectedPath": str(planned_path) if planned_path else None,
        "inferredFrom": inferred_from,
        "exists": bool(actual_path and actual_path.exists()),
        "plannedContact": stage_ref_summary(contact_ref),
    }


def resolve_reuse_config_path(
    reuse: str,
    source_path: pathlib.Path,
    repo_root: pathlib.Path,
) -> pathlib.Path | None:
    path = pathlib.Path(reuse).expanduser()
    if path.is_absolute():
        return path if path.exists() else None
    candidates: list[pathlib.Path] = []
    if path.suffix == ".py":
        candidates.extend(
            [
                source_path.resolve().parent / path,
                repo_root / path,
                repo_root / "configs" / "experiments" / path,
            ]
        )
    else:
        module_path = pathlib.Path(*reuse.split(".")).with_suffix(".py")
        candidates.extend(
            [
                repo_root / module_path,
                repo_root / "configs" / "experiments" / f"{reuse}.py",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_config_from_path(repo_root: pathlib.Path, path: pathlib.Path) -> dict[str, Any]:
    return to_jsonable(load_config_object_from_path(repo_root, path))


def load_config_object_from_path(repo_root: pathlib.Path, path: pathlib.Path) -> Any:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        if rel.suffix == ".py":
            module_name = ".".join(rel.with_suffix("").parts)
            module = importlib.import_module(module_name)
            return getattr(module, "EXP_CFG")
    except ValueError:
        pass
    module_name = f"configs.experiments.{path.stem}"
    module = importlib.import_module(module_name)
    return getattr(module, "EXP_CFG")


def resolve_pretrain_reuse(
    repo_root: pathlib.Path,
    config: dict[str, Any],
    source_path: pathlib.Path,
) -> dict[str, Any] | None:
    reuse = get_nested(config, "pretrain_reuse")
    if not reuse:
        return None

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_reuse = str(reuse)
    current_source = source_path
    target: str | None = None
    error: str | None = None

    while current_reuse:
        reuse_path = resolve_reuse_config_path(current_reuse, current_source, repo_root)
        if reuse_path is None:
            error = f"Unable to resolve pretrain_reuse={current_reuse!r}"
            break

        key = str(reuse_path.resolve())
        if key in seen:
            error = f"Cyclic pretrain_reuse reference: {display_config_ref(reuse_path, repo_root)}"
            break
        seen.add(key)

        try:
            reuse_config = load_config_from_path(repo_root, reuse_path)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

        next_reuse = get_nested(reuse_config, "pretrain_reuse")
        pretrain_enabled = bool(get_nested(reuse_config, "pretrain.enabled", False))
        display_ref = display_config_ref(reuse_path, repo_root)
        chain.append(
            {
                "ref": current_reuse,
                "config": display_ref,
                "path": str(reuse_path),
                "pretrainEnabled": pretrain_enabled,
                "pretrainReuse": str(next_reuse) if next_reuse else None,
            }
        )
        if next_reuse:
            current_reuse = str(next_reuse)
            current_source = reuse_path
            continue

        target = display_ref
        break

    if target is None and chain:
        target = chain[-1]["config"]
    return {
        "direct": str(reuse),
        "target": target or str(reuse),
        "targetPath": str(reuse_path) if "reuse_path" in locals() and reuse_path is not None else None,
        "chain": chain,
        "error": error,
    }


def first_config_checkpoint(config: dict[str, Any]) -> str | None:
    return first_nested(
        config,
        [
            "model.pretrained_encoder.checkpoint_path",
            "model.encoder.checkpoint_path",
            "rl.encoder_checkpoint",
        ],
    )


def pretrain_resume_checkpoint(config: dict[str, Any]) -> str | None:
    return get_nested(config, "pretrain.checkpoint_policy.resume_checkpoint")


def encoder_contact_source(
    repo_root: pathlib.Path,
    cfg_obj: Any,
    config: dict[str, Any],
    source_path: pathlib.Path,
    reuse_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    direct_checkpoint = first_config_checkpoint(config)
    pretrain_enabled = bool(get_nested(config, "pretrain.enabled", False))
    config_ref = display_config_ref(source_path, repo_root)

    if reuse_resolution and reuse_resolution.get("targetPath"):
        target_path = pathlib.Path(str(reuse_resolution["targetPath"]))
        try:
            target_cfg_obj = load_config_object_from_path(repo_root, target_path)
            target_config = to_jsonable(target_cfg_obj)
            target_ref = display_config_ref(target_path, repo_root)
            return {
                "type": "pretrain_reuse",
                "encoderConfig": target_ref,
                "encoderConfigPath": str(target_path),
                "directReuse": reuse_resolution.get("direct"),
                "reuseChain": reuse_resolution.get("chain", []),
                "encoderArtifact": stage_ref_summary(artifact_stage_ref(target_cfg_obj, "pretrain")),
                "contactData": contact_data_source(repo_root, target_cfg_obj, target_ref),
                "pretrainResumeCheckpoint": pretrain_resume_checkpoint(target_config),
                "directCheckpoint": direct_checkpoint,
                "error": reuse_resolution.get("error"),
            }
        except Exception as exc:
            return {
                "type": "pretrain_reuse",
                "encoderConfig": reuse_resolution.get("target"),
                "directReuse": reuse_resolution.get("direct"),
                "reuseChain": reuse_resolution.get("chain", []),
                "encoderArtifact": None,
                "contactData": None,
                "directCheckpoint": direct_checkpoint,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if pretrain_enabled:
        return {
            "type": "local_pretrain",
            "encoderConfig": config_ref,
            "encoderConfigPath": str(source_path),
            "encoderArtifact": stage_ref_summary(artifact_stage_ref(cfg_obj, "pretrain")),
            "contactData": contact_data_source(repo_root, cfg_obj, config_ref),
            "pretrainResumeCheckpoint": pretrain_resume_checkpoint(config),
            "directCheckpoint": direct_checkpoint,
            "error": None,
        }

    if direct_checkpoint:
        return {
            "type": "checkpoint",
            "encoderConfig": None,
            "encoderArtifact": None,
            "contactData": None,
            "directCheckpoint": direct_checkpoint,
            "error": None,
        }

    return {
        "type": "none",
        "encoderConfig": None,
        "encoderArtifact": None,
        "contactData": None,
        "directCheckpoint": None,
        "error": None,
    }


def build_stages(
    config: dict[str, Any],
    reuse_resolution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    contact_status = stage_status(config, "contact_gen", reuse_resolution)
    pretrain_status = stage_status(config, "pretrain", reuse_resolution)
    rl_status = stage_status(config, "rl", reuse_resolution)

    stages.append(
        {
            "key": "config",
            "title": "配置入口",
            "enabled": True,
            "requested": True,
            "status": "enabled",
            "statusText": "启动",
            "reason": None,
            "summary": "加载实验配置并解析全局参数",
            "params": compact_params(
                config,
                [
                    "name",
                    "general.name",
                    "paths_yaml",
                    "num_gpus",
                    "artifact_policy",
                    "pretrain_reuse",
                ],
            ),
        }
    )

    stages.append(
        {
            "key": "contact_gen",
            "title": "接触数据生成",
            **contact_status,
            "summary": "生成工具与物体接触样本",
            "params": compact_params(
                config,
                [
                    "contact_gen.name",
                    "contact_gen.num_pairs",
                    "contact_gen.num_object_poses",
                    "contact_gen.M",
                    "contact_gen.B",
                    "contact_gen.chunk_B",
                    "contact_gen.max_contacts_per_pair",
                    "contact_gen.physics.t_stabilize",
                    "contact_gen.physics.t_postcontact",
                    "contact_gen.visualization.enabled",
                ],
            ),
        }
    )

    stages.append(
        {
            "key": "pretrain",
            "title": "预训练",
            **pretrain_status,
            "summary": "训练 SDF / Diffusion / Post-contact 相关头",
            "params": compact_params(
                config,
                [
                    "pretrain.name",
                    "pretrain.enabled_heads",
                    "pretrain.tasks.sdf",
                    "pretrain.tasks.diffusion",
                    "pretrain.tasks.postcontact",
                    "pretrain.epochs",
                    "pretrain.batch.batch_size",
                    "pretrain.optimizer.learning_rate",
                    "pretrain.optimizer.min_learning_rate",
                    "pretrain.loss.sdf_relative_loss",
                    "pretrain.loss.sdf_relative_eps",
                    "pretrain.condition_normalization",
                    "pretrain.logger",
                    "pretrain.wandb_project",
                    "pretrain.wandb_run_name",
                    "pretrain.checkpoint_policy.resume_checkpoint",
                ],
            ),
        }
    )

    stages.append(
        {
            "key": "rl",
            "title": "强化学习",
            **rl_status,
            "summary": "启动策略训练与评估流程",
            "params": compact_params(
                config,
                [
                    "rl.name",
                    "rl.env.num_envs",
                    "rl.launch.distributed",
                    "rl.launch.logger",
                    "rl.launch.wandb_project",
                    "rl.launch.run_name",
                    "rl.ppo.max_iterations",
                    "rl.ppo.save_interval",
                    "rl.actor_critic_class",
                    "rl.freeze_encoder",
                    "rl.separate_actor_critic_fusion",
                    "rl.reward.object_goal_tracking_term_weight",
                    "rl.reward.object_goal_tracking_fine_term_weight",
                    "rl.reward.task_success_term_weight",
                    "rl.reward.rotation_distance_divisor",
                ],
            ),
        }
    )

    return stages


def manifest_patterns(artifact_root: pathlib.Path) -> list[tuple[str, pathlib.Path, str]]:
    patterns = [
        ("experiment", artifact_root / "experiment", "*/*/manifest.json"),
        ("rl", artifact_root / "RL", "*/*/*/*/*/manifest.json"),
    ]
    if str(sys.argv).find("--include-stage-artifacts") >= 0:
        patterns.extend(
            [
                ("encoder", artifact_root / "encoder", "*/*/*/*/manifest.json"),
            ]
        )
    if str(sys.argv).find("--include-contact-artifacts") >= 0:
        patterns.append(("contact", artifact_root / "contact", "*/*/*/manifest.json"))
    return patterns


def iter_manifest_paths(base: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    depth_by_pattern = {
        "*/*/manifest.json": 3,
        "*/*/*/*/*/manifest.json": 6,
        "*/*/*/*/manifest.json": 5,
        "*/*/*/manifest.json": 4,
    }
    depth = depth_by_pattern.get(pattern)
    if depth is None:
        return list(base.glob(pattern))
    try:
        output = subprocess.check_output(
            [
                "find",
                str(base),
                "-mindepth",
                str(depth),
                "-maxdepth",
                str(depth),
                "-name",
                "manifest.json",
                "-type",
                "f",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return list(base.glob(pattern))
    return [pathlib.Path(line) for line in output.splitlines() if line]


def parse_wandb_dir_name(path: pathlib.Path) -> tuple[str | None, datetime | None]:
    match = re.match(r"^run-(\d{8})_(\d{6})-([A-Za-z0-9_-]+)$", path.name)
    if not match:
        return None, None
    run_id = match.group(3)
    started = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    return run_id, started


def read_text_limited(path: pathlib.Path, max_bytes: int = 2_000_000) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def build_wandb_index(repo_root: pathlib.Path) -> dict[str, Any]:
    wandb_root = repo_root / "wandb"
    by_artifact: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    if not wandb_root.exists():
        return {"byArtifact": by_artifact, "runs": runs}

    for run_dir in wandb_root.glob("run-*"):
        if not run_dir.is_dir():
            continue
        run_id, started_at = parse_wandb_dir_name(run_dir)
        if not run_id:
            continue

        output_log = read_text_limited(run_dir / "files" / "output.log", 200_000)
        internal_log = read_text_limited(run_dir / "logs" / "debug-internal.log", 200_000)
        log_dir_match = re.search(
            r"Storing git diff for '[^']+' in: ([^\n]+)|'log_dir': '([^']+)'|\"log_dir\"\s*:\s*\"([^\"]+)\"",
            output_log,
        )
        entity_project_match = re.search(r"/files/([^/\"]+)/([^/\"]+)/" + re.escape(run_id), internal_log)

        project = entity_project_match.group(2) if entity_project_match else None
        entity = entity_project_match.group(1) if entity_project_match else None
        log_dir = (
            log_dir_match.group(1) or log_dir_match.group(2) or log_dir_match.group(3)
        ) if log_dir_match else None
        if log_dir and log_dir.endswith("/git/tool-generalist.diff"):
            log_dir = str(pathlib.Path(log_dir).parents[1])

        item = {
            "runId": run_id,
            "runName": None,
            "project": project,
            "entity": entity,
            "startedAt": started_at.isoformat() if started_at else None,
            "localDir": str(run_dir),
            "artifactPath": log_dir,
        }
        runs.append(item)
        if log_dir:
            by_artifact[str(pathlib.Path(log_dir))] = item

    return {"byArtifact": by_artifact, "runs": runs}


def parse_timestamp_from_path(path: pathlib.Path) -> str | None:
    for part in reversed(path.parts):
        if re.match(r"^\d{8}T\d{6}Z$", part):
            return part
    return None


def direct_dir_stats(directory: pathlib.Path) -> dict[str, Any]:
    return {
        "fileCount": 0,
        "totalBytes": 0,
        "checkpointCount": 0,
        "latestCheckpoint": None,
        "hasTensorboard": False,
        "hasBestCheckpoint": False,
    }


def checkpoint_stats(directory: pathlib.Path) -> dict[str, Any]:
    stats = {
        "checkpointCount": 0,
        "latestCheckpoint": None,
        "hasTensorboard": False,
        "hasBestCheckpoint": False,
        "fileCount": 0,
        "totalBytes": 0,
        "isTrivialCheckpointRun": False,
    }
    latest_iteration: int | None = None
    try:
        names = [item.name for item in directory.iterdir() if item.is_file()]
    except OSError:
        return stats

    stats["fileCount"] = len(names)
    for name in names:
        if name.startswith("events.out.tfevents"):
            stats["hasTensorboard"] = True
        if name == "best.pt":
            stats["hasBestCheckpoint"] = True
            stats["checkpointCount"] += 1
        match = re.match(r"^model_(\d+)\.pt$", name)
        if match:
            stats["checkpointCount"] += 1
            iteration = int(match.group(1))
            if latest_iteration is None or iteration > latest_iteration:
                latest_iteration = iteration

    stats["latestCheckpoint"] = latest_iteration
    stats["isTrivialCheckpointRun"] = not (latest_iteration is not None and latest_iteration > 0)
    return stats


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def tool_group_name(name: str) -> str:
    return re.sub(r"_var_\d+$", "", name)


def aggregate_eval_rows(rows: list[dict[str, Any]], *, group_variants: bool) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_name = str(row.get("name") or row.get("tool") or row.get("object") or "unknown")
        name = tool_group_name(raw_name) if group_variants else raw_name
        item = grouped.setdefault(
            name,
            {
                "name": name,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "variants": 0,
                "rawNames": [],
            },
        )
        episodes = safe_int(row.get("episodes"))
        successes = safe_int(row.get("successes"))
        item["episodes"] += episodes
        item["successes"] += successes
        item["variants"] += 1
        if raw_name not in item["rawNames"]:
            item["rawNames"].append(raw_name)

    results = []
    for item in grouped.values():
        episodes = safe_int(item["episodes"])
        successes = safe_int(item["successes"])
        item["success_rate"] = float(successes) / float(episodes) if episodes > 0 else 0.0
        if not group_variants:
            item.pop("variants", None)
            item.pop("rawNames", None)
        results.append(item)
    return sorted(results, key=lambda item: item["name"])


def normalize_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows):
        raw_name = str(row.get("name") or row.get("tool") or row.get("object") or f"item_{index:04d}")
        episodes = safe_int(row.get("episodes"))
        successes = safe_int(row.get("successes"))
        success_rate = safe_float(row.get("success_rate"))
        if episodes > 0 and row.get("success_rate") is None:
            success_rate = float(successes) / float(episodes)
        results.append(
            {
                "name": raw_name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                "rank": row.get("rank"),
            }
        )
    return sorted(results, key=lambda item: item["name"])


def normalize_eval_summary(path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("per_tool"), list):
        kind = "multi_tool"
        rows = aggregate_eval_rows(payload["per_tool"], group_variants=True)
        raw_count = len(payload["per_tool"])
        item_count = len(rows)
        item_label = "工具"
        chart_title = "各工具成功率（按 variants 聚合）"
    elif isinstance(payload.get("per_object"), list):
        kind = "single_tool"
        rows = normalize_eval_rows(payload["per_object"])
        raw_count = len(payload["per_object"])
        item_count = len(rows)
        item_label = "物体"
        chart_title = "各物体成功率"
    else:
        return None

    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        modified_at = None

    episodes = safe_int(payload.get("episodes"))
    successes = safe_int(payload.get("successes"))
    success_rate = safe_float(payload.get("success_rate"))
    if episodes > 0 and (payload.get("success_rate") is None):
        success_rate = float(successes) / float(episodes)

    return {
        "kind": kind,
        "file": path.name,
        "path": str(path),
        "modifiedAt": modified_at,
        "task": payload.get("task"),
        "checkpoint": payload.get("checkpoint"),
        "tool": payload.get("tool"),
        "worldSize": payload.get("world_size"),
        "numEnvsPerRank": payload.get("num_envs_per_rank"),
        "episodes": episodes,
        "successes": successes,
        "successRate": success_rate,
        "itemCount": item_count,
        "rawItemCount": raw_count,
        "itemLabel": item_label,
        "chartTitle": chart_title,
        "episodesPerTool": payload.get("episodes_per_tool"),
        "episodesPerObject": payload.get("episodes_per_object"),
        "randomizeObjects": payload.get("randomize_objects"),
        "objectRandomSeed": payload.get("object_random_seed"),
        "rows": rows,
    }


def read_eval_summaries(directory: pathlib.Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    paths = [directory / "eval_tools_summary.json"]
    paths.extend(sorted(directory.glob("eval_single_tool_*_summary.json")))
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        summary = normalize_eval_summary(path, payload)
        if summary:
            summaries.append(summary)
    summaries.sort(key=lambda item: item.get("modifiedAt") or item.get("file") or "", reverse=True)
    return summaries


def read_manifest_snippet(path: pathlib.Path, tail_bytes: int = 24000) -> str:
    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            tail = handle.read(tail_bytes)
        return (head + b"\n" + tail).decode("utf-8", errors="replace")
    except OSError:
        return ""


def extract_scalar(text: str, key: str) -> Any:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*("([^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1)
    try:
        return json.loads(raw)
    except Exception:
        return raw.strip('"')


def extract_source_config(text: str) -> str | None:
    match = re.search(r'"source_paths"\s*:\s*\{.*?"config"\s*:\s*"([^"]+)"', text, re.S)
    if match:
        return match.group(1)
    match = re.search(r'"config"\s*:\s*"(configs/experiments/[^"]+)"', text)
    return match.group(1) if match else None


def path_summary(path: pathlib.Path, artifact_root: pathlib.Path, category: str) -> dict[str, Any]:
    try:
        rel = path.parent.relative_to(artifact_root)
    except ValueError:
        rel = path.parent
    parts = rel.parts
    summary: dict[str, Any] = {}
    if category == "rl" and len(parts) >= 6:
        config_id = parts[4] if parts[4] != "rl_default" else parts[1]
        summary.update(
            {
                "expCfgName": config_id,
                "artifactConfigGroup": parts[1],
                "contactGen": parts[2],
                "model": parts[3],
                "runName": parts[4],
                "id": parts[5],
            }
        )
    elif category == "encoder" and len(parts) >= 5:
        summary.update(
            {
                "expCfgName": parts[1],
                "contactGen": parts[2],
                "pretrainName": parts[3],
                "id": parts[4],
            }
        )
    elif category == "experiment" and len(parts) >= 3:
        summary.update({"expCfgName": parts[1], "id": parts[2]})
    elif category == "contact" and len(parts) >= 4:
        summary.update({"expCfgName": parts[1], "contactGen": parts[2], "id": parts[3]})
    return summary


def summarize_manifest(path: pathlib.Path, artifact_root: pathlib.Path, category: str) -> dict[str, Any] | None:
    text = read_manifest_snippet(path)
    if not text:
        return None
    derived = path_summary(path, artifact_root, category)
    source_config = extract_source_config(text)
    source_config_id = pathlib.Path(source_config).stem if source_config else None
    directory = path.parent
    stat = direct_dir_stats(directory)
    artifact_name = extract_scalar(text, "artifact_name") or str(directory.relative_to(artifact_root))
    exp_cfg_name = extract_scalar(text, "exp_cfg_name") or derived.get("expCfgName")

    result = {
        "id": derived.get("id") or directory.name,
        "category": category,
        "artifactType": extract_scalar(text, "artifact_type") or category,
        "artifactName": artifact_name,
        "path": str(directory),
        "manifestPath": str(path),
        "sourceConfig": source_config,
        "sourceConfigId": source_config_id,
        "expCfgName": exp_cfg_name,
        "status": extract_scalar(text, "status"),
        "createdAt": extract_scalar(text, "created_at") or parse_timestamp_from_path(path),
        "configHash": extract_scalar(text, "config_hash"),
        "gitCommit": extract_scalar(text, "git_commit"),
        "gitDirty": extract_scalar(text, "git_dirty"),
        "stage": extract_scalar(text, "stage") or category,
        "mode": extract_scalar(text, "mode"),
        "action": extract_scalar(text, "action"),
        "requested": extract_scalar(text, "requested"),
        "executed": extract_scalar(text, "executed"),
        "reused": extract_scalar(text, "reused"),
        "result": extract_scalar(text, "result"),
        "error": extract_scalar(text, "error"),
        "dependencyReason": extract_scalar(text, "dependency_reason"),
        "resolvedEncoderCheckpoint": extract_scalar(text, "resolved_encoder_checkpoint"),
        "model": derived.get("model"),
        "contactGen": derived.get("contactGen"),
        "pretrainName": derived.get("pretrainName"),
        "pretrainHeads": None,
        "runName": derived.get("runName"),
        "wandbProject": None,
        "numGpus": None,
        "numEnv": None,
        "maxIterations": None,
        "saveInterval": None,
        "checkpointCount": stat["checkpointCount"],
        "latestCheckpoint": stat["latestCheckpoint"],
        "hasTensorboard": stat["hasTensorboard"],
        "hasBestCheckpoint": stat["hasBestCheckpoint"],
        "fileCount": stat["fileCount"],
        "totalBytes": stat["totalBytes"],
        "evals": read_eval_summaries(directory) if category == "rl" else [],
    }
    return result


def summarize_manifest_path(path: pathlib.Path, artifact_root: pathlib.Path, category: str) -> dict[str, Any]:
    directory = path.parent
    derived = path_summary(path, artifact_root, category)
    try:
        artifact_name = str(directory.relative_to(artifact_root))
    except ValueError:
        artifact_name = str(directory)
    exp_cfg_name = derived.get("expCfgName")
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        modified_at = None
    runtime_spec: dict[str, Any] = {}
    if category == "rl":
        try:
            runtime_spec = json.loads((directory / "rl_runtime_spec.json").read_text(encoding="utf-8"))
        except Exception:
            runtime_spec = {}
    stat = checkpoint_stats(directory) if category == "rl" else direct_dir_stats(directory)
    launch_params = runtime_spec.get("launch_params") if isinstance(runtime_spec.get("launch_params"), dict) else {}
    return {
        "id": derived.get("id") or directory.name,
        "category": category,
        "artifactType": category,
        "artifactName": artifact_name,
        "path": str(directory),
        "manifestPath": str(path),
        "sourceConfig": f"configs/experiments/{exp_cfg_name}.py" if exp_cfg_name else None,
        "sourceConfigId": exp_cfg_name,
        "expCfgName": exp_cfg_name,
        "status": "found",
        "createdAt": parse_timestamp_from_path(path) or modified_at,
        "configHash": directory.name if re.match(r"^[0-9a-f]{32,}$", directory.name) else None,
        "gitCommit": None,
        "gitDirty": None,
        "stage": category,
        "mode": None,
        "action": None,
        "requested": None,
        "executed": None,
        "reused": None,
        "result": None,
        "error": None,
        "dependencyReason": None,
        "resolvedEncoderCheckpoint": None,
        "model": derived.get("model"),
        "contactGen": derived.get("contactGen"),
        "pretrainName": derived.get("pretrainName"),
        "pretrainHeads": None,
        "runName": launch_params.get("run_name") or derived.get("runName"),
        "wandbProject": launch_params.get("wandb_project"),
        "numGpus": runtime_spec.get("num_gpus"),
        "numEnv": runtime_spec.get("num_envs"),
        "maxIterations": runtime_spec.get("max_iterations"),
        "saveInterval": runtime_spec.get("save_interval"),
        "wandb": None,
        "checkpointCount": stat["checkpointCount"],
        "latestCheckpoint": stat["latestCheckpoint"],
        "hasTensorboard": stat["hasTensorboard"],
        "hasBestCheckpoint": stat["hasBestCheckpoint"],
        "fileCount": stat["fileCount"],
        "totalBytes": stat["totalBytes"],
        "isTrivialCheckpointRun": stat.get("isTrivialCheckpointRun", False),
        "hiddenByDefault": stat.get("isTrivialCheckpointRun", False),
        "evals": read_eval_summaries(directory) if category == "rl" else [],
    }


def read_artifacts(artifact_root: pathlib.Path) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    total = 0
    detailed = str(sys.argv).find("--detailed-artifacts") >= 0
    repo_root = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path.cwd()
    wandb_index = build_wandb_index(repo_root)
    if not artifact_root.exists():
        return {"root": str(artifact_root), "total": 0, "byConfig": by_config, "errors": errors}

    for category, base, pattern in manifest_patterns(artifact_root):
        if not base.exists():
            continue
        for path in iter_manifest_paths(base, pattern):
            total += 1
            result = (
                summarize_manifest(path, artifact_root, category)
                if detailed
                else summarize_manifest_path(path, artifact_root, category)
            )
            if not result:
                errors.append({"file": str(path), "error": "Unable to parse manifest"})
                continue
            if result.get("category") == "rl":
                wandb_run = wandb_index["byArtifact"].get(result.get("path"))
                if wandb_run:
                    entity = wandb_run.get("entity")
                    project = wandb_run.get("project") or result.get("wandbProject")
                    run_id = wandb_run.get("runId")
                    run_url = (
                        f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                        if entity and project and run_id
                        else None
                    )
                    result["wandb"] = {
                        **wandb_run,
                        "runName": wandb_run.get("runName") or result.get("runName"),
                        "project": project,
                        "panelUrl": run_url,
                        "runUrl": run_url,
                    }
            config_id = result.get("sourceConfigId") or result.get("expCfgName")
            if not config_id:
                errors.append({"file": str(path), "error": "Missing source config"})
                continue
            by_config.setdefault(str(config_id), []).append(result)

    for results in by_config.values():
        results.sort(key=lambda item: item.get("createdAt") or "", reverse=True)

    return {"root": str(artifact_root), "total": total, "byConfig": by_config, "errors": errors}


def summarize_artifacts(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_status: dict[str, int] = {}
    latest = None
    for item in results:
        category = item.get("category") or "unknown"
        status = item.get("status") or "unknown"
        by_category[category] = by_category.get(category, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        if latest is None or (item.get("createdAt") or "") > (latest.get("createdAt") or ""):
            latest = item
    return {
        "total": len(results),
        "byCategory": by_category,
        "byStatus": by_status,
        "latestCreatedAt": latest.get("createdAt") if latest else None,
        "latestStatus": latest.get("status") if latest else None,
        "latestStage": latest.get("stage") if latest else None,
    }


def summarize_evals(results: list[dict[str, Any]]) -> dict[str, Any]:
    evals = [
        eval_summary
        for artifact in results
        for eval_summary in artifact.get("evals", [])
        if isinstance(eval_summary, dict)
    ]
    latest = None
    best = None
    by_kind: dict[str, int] = {}
    for item in evals:
        kind = str(item.get("kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if latest is None or (item.get("modifiedAt") or "") > (latest.get("modifiedAt") or ""):
            latest = item
        if best is None or safe_float(item.get("successRate")) > safe_float(best.get("successRate")):
            best = item
    return {
        "total": len(evals),
        "byKind": by_kind,
        "latestModifiedAt": latest.get("modifiedAt") if latest else None,
        "latestSuccessRate": latest.get("successRate") if latest else None,
        "bestSuccessRate": best.get("successRate") if best else None,
        "bestFile": best.get("file") if best else None,
    }


def build_summary(
    path: pathlib.Path,
    config: dict[str, Any],
    source: str,
    reuse_resolution: dict[str, Any] | None = None,
    encoder_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = ast.get_docstring(ast.parse(source)) or ""
    assignments = parse_assignments(source)
    enabled_stages = [
        stage["key"]
        for stage in build_stages(config, reuse_resolution)
        if stage["key"] != "config" and stage["enabled"]
    ]
    name = get_nested(config, "name") or get_nested(config, "general.name") or path.stem
    return {
        "id": path.stem,
        "file": path.name,
        "path": str(path),
        "name": name,
        "description": doc,
        "model": get_nested(config, "model.name"),
        "pathsYaml": get_nested(config, "paths_yaml"),
        "numGpus": get_nested(config, "num_gpus"),
        "pretrainReuse": get_nested(config, "pretrain_reuse"),
        "pretrainReuseResolution": reuse_resolution,
        "encoderSource": encoder_source,
        "runName": get_nested(config, "rl.launch.run_name") or get_nested(config, "pretrain.wandb_run_name"),
        "wandbProject": get_nested(config, "rl.launch.wandb_project") or get_nested(config, "pretrain.wandb_project"),
        "enabledStages": enabled_stages,
        "stages": build_stages(config, reuse_resolution),
        "assignments": assignments,
        "assignmentCount": len(assignments),
        "artifactSummary": {"total": 0, "byCategory": {}, "byStatus": {}},
        "evalSummary": {"total": 0, "byKind": {}},
        "artifacts": [],
        "fullConfig": config,
        "sourceText": source,
    }


def read_experiment(repo_root: pathlib.Path, path: pathlib.Path) -> dict[str, Any]:
    module_name = f"configs.experiments.{path.stem}"
    source = path.read_text(encoding="utf-8")
    module = importlib.import_module(module_name)
    cfg_obj = getattr(module, "EXP_CFG")
    config = to_jsonable(cfg_obj)
    reuse_resolution = resolve_pretrain_reuse(repo_root, config, path)
    encoder_source = encoder_contact_source(repo_root, cfg_obj, config, path, reuse_resolution)
    return build_summary(path.relative_to(repo_root), config, source, reuse_resolution, encoder_source)


def main() -> int:
    repo_root = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path.cwd()
    sys.path.insert(0, str(repo_root))
    experiments_dir = repo_root / "configs" / "experiments"
    artifact_root = pathlib.Path("/mnt/project/world_model/tool_generalist/artifacts")
    artifacts = read_artifacts(artifact_root)
    experiments: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(experiments_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            experiment = read_experiment(repo_root, path)
        except Exception as error:
            source = path.read_text(encoding="utf-8", errors="replace")
            fallback_config = {"name": path.stem}
            experiment = build_summary(path.relative_to(repo_root), fallback_config, source)
            experiment["loadError"] = f"{type(error).__name__}: {error}"
            errors.append({"file": path.name, "error": experiment["loadError"]})

        experiment_artifacts = artifacts["byConfig"].get(path.stem, [])
        visible_rl_artifacts = [
            item
            for item in experiment_artifacts
            if item.get("category") == "rl" and not item.get("hiddenByDefault")
        ]
        experiment["artifacts"] = experiment_artifacts
        experiment["artifactSummary"] = summarize_artifacts(visible_rl_artifacts)
        experiment["artifactSummaryAll"] = summarize_artifacts(experiment_artifacts)
        experiment["evalSummary"] = summarize_evals(experiment_artifacts)
        experiments.append(experiment)

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "repoRoot": str(repo_root),
        "artifactRoot": artifacts["root"],
        "artifactCount": artifacts["total"],
        "experiments": experiments,
        "errors": errors + artifacts["errors"],
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

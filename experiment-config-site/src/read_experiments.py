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


def stage_enabled(config: dict[str, Any], stage: str) -> bool:
    return bool(get_nested(config, f"{stage}.enabled", False))


def compact_params(config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for dotted in keys:
        value = get_nested(config, dotted)
        if value is not None:
            params[dotted] = value
    return params


def build_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []

    stages.append(
        {
            "key": "config",
            "title": "配置入口",
            "enabled": True,
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
            "enabled": stage_enabled(config, "contact_gen"),
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
            "enabled": stage_enabled(config, "pretrain"),
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
            "enabled": stage_enabled(config, "rl"),
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
    ]
    if str(sys.argv).find("--include-stage-artifacts") >= 0:
        patterns.extend(
            [
                ("rl", artifact_root / "RL", "*/*/*/*/*/manifest.json"),
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
        summary.update(
            {
                "expCfgName": parts[1],
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
        "runName": derived.get("runName"),
        "wandbProject": None,
        "numGpus": None,
        "numEnv": None,
        "maxIterations": None,
        "saveInterval": None,
        "checkpointCount": 0,
        "latestCheckpoint": None,
        "hasTensorboard": False,
        "hasBestCheckpoint": False,
        "fileCount": 0,
        "totalBytes": 0,
    }


def read_artifacts(artifact_root: pathlib.Path) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    total = 0
    detailed = str(sys.argv).find("--detailed-artifacts") >= 0
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


def build_summary(path: pathlib.Path, config: dict[str, Any], source: str) -> dict[str, Any]:
    doc = ast.get_docstring(ast.parse(source)) or ""
    assignments = parse_assignments(source)
    enabled_stages = [
        stage["key"]
        for stage in build_stages(config)
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
        "runName": get_nested(config, "rl.launch.run_name") or get_nested(config, "pretrain.wandb_run_name"),
        "wandbProject": get_nested(config, "rl.launch.wandb_project") or get_nested(config, "pretrain.wandb_project"),
        "enabledStages": enabled_stages,
        "stages": build_stages(config),
        "assignments": assignments,
        "assignmentCount": len(assignments),
        "artifactSummary": {"total": 0, "byCategory": {}, "byStatus": {}},
        "artifacts": [],
        "fullConfig": config,
        "sourceText": source,
    }


def read_experiment(repo_root: pathlib.Path, path: pathlib.Path) -> dict[str, Any]:
    module_name = f"configs.experiments.{path.stem}"
    source = path.read_text(encoding="utf-8")
    module = importlib.import_module(module_name)
    config = to_jsonable(getattr(module, "EXP_CFG"))
    return build_summary(path.relative_to(repo_root), config, source)


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
        experiment["artifacts"] = experiment_artifacts
        experiment["artifactSummary"] = summarize_artifacts(experiment_artifacts)
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

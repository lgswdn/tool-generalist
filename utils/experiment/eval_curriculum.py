"""Runtime object-list curriculum derived from eval_objects outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from utils.config.paths import ProjectPaths


def normalize_success_rate_threshold(value: float) -> float:
    threshold = float(value)
    if threshold > 1.0:
        threshold = threshold / 100.0
    if threshold < 0.0 or threshold > 1.0:
        raise RuntimeError(
            "curriculum success-rate threshold must be in [0, 1] or a percentage in [0, 100]"
        )
    return threshold


def materialize_curriculum_object_manifest(
    paths: ProjectPaths,
    rl_artifact_dir: Path,
    *,
    threshold: float,
) -> Path:
    summary_path = latest_eval_objects_summary(rl_artifact_dir)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise RuntimeError(f"eval_objects_summary.json must contain a JSON object: {summary_path}")

    per_object = summary.get("per_object")
    if not isinstance(per_object, list):
        raise RuntimeError(f"eval_objects_summary.json is missing per_object list: {summary_path}")

    selected_object_names = {
        str(row["name"])
        for row in per_object
        if isinstance(row, dict)
        and "name" in row
        and float(row.get("success_rate", 0.0)) <= threshold
    }
    if not selected_object_names:
        raise RuntimeError(
            f"No objects in {summary_path} have success_rate <= {threshold:.4f}; "
            "refusing to create an empty RL curriculum object list."
        )

    source_paths_yaml = _summary_source_paths_yaml(summary) or paths.source_yaml
    candidates = _load_object_candidates_from_paths_yaml(source_paths_yaml)
    selected_candidates = [
        candidate
        for candidate in candidates
        if _candidate_object_name(candidate) in selected_object_names
    ]
    if not selected_candidates:
        raise RuntimeError(
            f"Objects selected from {summary_path} were not found in candidates_json "
            f"from {source_paths_yaml}."
        )

    output_path = (rl_artifact_dir / "curriculum_objects_from_eval.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(selected_candidates, f, ensure_ascii=False, indent=2)
    return output_path


def latest_eval_objects_summary(current_rl_artifact_dir: Path) -> Path:
    run_root = current_rl_artifact_dir.parent
    candidates: list[tuple[float, str, Path]] = []
    if run_root.is_dir():
        for path in run_root.iterdir():
            summary_path = path / "eval_objects_summary.json"
            if path.is_dir() and summary_path.is_file():
                candidates.append((summary_path.stat().st_mtime, path.name, summary_path))
    if not candidates:
        raise RuntimeError(
            "No evaluated RL run was found for this experiment. Expected at least one "
            f"eval_objects_summary.json under {run_root}. Run ./eval.bash <config> first."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def latest_eval_checkpoint(current_rl_artifact_dir: Path) -> Path:
    summary_path = latest_eval_objects_summary(current_rl_artifact_dir)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise RuntimeError(f"eval_objects_summary.json must contain a JSON object: {summary_path}")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise RuntimeError(f"eval_objects_summary.json is missing checkpoint: {summary_path}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint from eval_objects_summary.json does not exist: {checkpoint_path}"
        )
    return checkpoint_path


def _summary_source_paths_yaml(summary: dict[str, Any]) -> Path | None:
    for key in ("effective_source_paths_yaml", "runtime_spec_paths_yaml"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if path.is_file():
                return path.resolve()
    return None


def _load_object_candidates_from_paths_yaml(paths_yaml: str | Path) -> list[str]:
    paths_yaml = Path(paths_yaml).expanduser().resolve()
    with paths_yaml.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"paths.yaml must contain a mapping: {paths_yaml}")

    candidates_value = None
    if isinstance(payload.get("dgn"), dict):
        candidates_value = payload["dgn"].get("candidates_json")
    if candidates_value is None and isinstance(payload.get("objects"), dict):
        candidates_value = payload["objects"].get("candidates_json")
    if not isinstance(candidates_value, str) or not candidates_value.strip():
        raise RuntimeError(f"paths.yaml must define dgn.candidates_json or objects.candidates_json: {paths_yaml}")

    candidates_path = Path(candidates_value).expanduser()
    if not candidates_path.is_absolute():
        candidates_path = paths_yaml.parent / candidates_path
    with candidates_path.resolve().open("r", encoding="utf-8") as f:
        candidates = json.load(f)
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise RuntimeError(f"Object candidates JSON must contain a list of strings: {candidates_path}")
    if not candidates:
        raise RuntimeError(f"Object candidates JSON is empty: {candidates_path}")
    return candidates


def _candidate_object_name(candidate: str) -> str:
    if "-" not in candidate:
        return candidate
    return candidate.rsplit("-", 1)[0]

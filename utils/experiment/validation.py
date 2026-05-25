"""Lightweight validation for experiment planning and running."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from configs.config_contact_gen import TOOL_SOURCE_OBJECTS, TOOL_SOURCE_SELECTED_TOOLS
from configs.config_exp import ConfigValidationError, ExpCfg
from utils.config.paths import PathsConfigError, ProjectPaths, require_path
from utils.experiment.effective_paths import apply_experiment_path_overrides
from utils.experiment.runtime import visible_gpu_count
from utils.experiment.stages import contact_stage_required
from utils.io import read_json


class ExperimentValidationError(ValueError):
    """Raised when a config cannot be planned for the current workspace."""


def validate_for_plan(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    cuda_visible_devices: str | None = None,
) -> None:
    """Run only cheap checks needed to print a plan.

    Plan mode must avoid filesystem probes beyond loading paths.yaml: no
    manifest reads, no JSON reads, no checkpoint inspection, and no
    ``Path.exists`` checks for configured paths.
    """

    paths = apply_experiment_path_overrides(cfg, paths)
    errors: list[str] = []
    try:
        cfg.validate()
    except ConfigValidationError as exc:
        errors.append(str(exc))

    errors.extend(validate_cuda_visible_devices_gpu_count(cfg, cuda_visible_devices))
    errors.extend(validate_required_path_keys_for_plan(cfg, paths))

    if errors:
        raise ExperimentValidationError("; ".join(errors))


def validate_for_run(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    cuda_visible_devices: str | None = None,
) -> None:
    validate_full_config(
        cfg,
        paths,
        strict_paths=True,
        cuda_visible_devices=cuda_visible_devices,
    )


def validate_for_planning(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    strict_paths: bool = True,
    cuda_visible_devices: str | None = None,
) -> None:
    """Backward-compatible wrapper.

    ``strict_paths=False`` maps to the new fast plan validation.  Existing
    callers that requested strict validation keep run-mode semantics.
    """

    if strict_paths:
        validate_for_run(cfg, paths, cuda_visible_devices=cuda_visible_devices)
    else:
        validate_for_plan(cfg, paths, cuda_visible_devices=cuda_visible_devices)


def validate_full_config(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    strict_paths: bool = True,
    cuda_visible_devices: str | None = None,
) -> None:
    """Run light full-config checks without importing training or Isaac code."""

    paths = apply_experiment_path_overrides(cfg, paths)
    errors: list[str] = []
    try:
        cfg.validate()
    except ConfigValidationError as exc:
        errors.append(str(exc))

    errors.extend(
        validate_cuda_visible_devices_gpu_count(
            cfg,
            cuda_visible_devices,
        )
    )
    errors.extend(validate_object_tool_manifests_non_empty(cfg, paths, strict_paths=strict_paths))
    errors.extend(validate_model_general_num_points_match(cfg))
    errors.extend(
        validate_encoder_checkpoint_path_and_declared_dims(
            cfg,
            strict_paths=strict_paths,
        )
    )
    errors.extend(validate_contact_schema_version(cfg))
    errors.extend(validate_isaac_task_and_rsl_rl_entrypoint_strings(cfg))

    if errors:
        raise ExperimentValidationError("; ".join(errors))


def validate_required_path_keys_for_plan(cfg: ExpCfg, paths: ProjectPaths) -> list[str]:
    errors: list[str] = []
    for key in _required_plan_path_keys(cfg):
        if paths.get(key) is None:
            errors.append(f"Missing required paths.yaml key '{key}'")
    return errors


def _required_plan_path_keys(cfg: ExpCfg) -> tuple[str, ...]:
    keys: list[str] = []
    if contact_stage_required(cfg) or cfg.rl.enabled:
        keys.append("objects.candidates_json")
        if cfg.contact_gen.tool_source == TOOL_SOURCE_OBJECTS:
            keys.append("objects.obj_dir")
        if cfg.rl.enabled or cfg.contact_gen.tool_source == TOOL_SOURCE_SELECTED_TOOLS:
            keys.extend(
                [
                    "tools.tools_selected_json",
                    "tools.tools_adjusted_json",
                    "tools.meshdata_adjusted_root",
                ]
            )
    return tuple(dict.fromkeys(keys))


def validate_cuda_visible_devices_gpu_count(
    cfg: ExpCfg,
    cuda_visible_devices: str | None,
) -> list[str]:
    errors: list[str] = []
    visible = visible_gpu_count(cuda_visible_devices)
    if visible is not None and cfg.num_gpus > visible:
        errors.append(
            f"ExpCfg.num_gpus={cfg.num_gpus} exceeds CUDA_VISIBLE_DEVICES count {visible}"
        )
    return errors


def validate_object_tool_manifests_non_empty(
    cfg: ExpCfg,
    paths: ProjectPaths,
    *,
    strict_paths: bool = True,
) -> list[str]:
    errors: list[str] = []
    require_selected_tools = (
        cfg.rl.enabled
        or (contact_stage_required(cfg) and cfg.contact_gen.tool_source == TOOL_SOURCE_SELECTED_TOOLS)
    )
    if contact_stage_required(cfg) or cfg.rl.enabled:
        _require_json_non_empty(errors, paths, "objects.candidates_json", strict_paths)
    if require_selected_tools:
        _require_json_non_empty(errors, paths, "tools.tools_selected_json", strict_paths)
        _require_json_non_empty(errors, paths, "tools.tools_adjusted_json", strict_paths)
        _require_path(errors, paths, "tools.meshdata_adjusted_root", strict_paths)
    if contact_stage_required(cfg) and cfg.contact_gen.tool_source == TOOL_SOURCE_OBJECTS:
        _require_path(errors, paths, "objects.obj_dir", strict_paths)
    if cfg.general.objects_manifest:
        _require_existing_json_non_empty(
            errors,
            Path(cfg.general.objects_manifest),
            "GeneralCfg.objects_manifest",
            strict_paths,
        )
    if cfg.contact_gen.object_tool_manifest:
        _require_existing_json_non_empty(
            errors,
            _resolve_config_path(cfg.contact_gen.object_tool_manifest, paths.source_yaml.parent),
            "ContactGenCfg.object_tool_manifest",
            strict_paths,
        )
    if cfg.general.tools_selected_json:
        _require_existing_json_non_empty(
            errors,
            Path(cfg.general.tools_selected_json),
            "GeneralCfg.tools_selected_json",
            strict_paths,
        )
    if cfg.general.tools_manifest:
        _require_existing_json_non_empty(
            errors,
            Path(cfg.general.tools_manifest),
            "GeneralCfg.tools_manifest",
            strict_paths,
        )
    return errors


def validate_model_general_num_points_match(cfg: ExpCfg) -> list[str]:
    errors: list[str] = []
    try:
        encoder = cfg.model.encoder
    except ValueError as exc:
        return [str(exc)]
    if encoder.num_points != cfg.general.num_points:
        errors.append(
            "ModelCfg.encoder.num_points must match GeneralCfg.num_points "
            f"({encoder.num_points} != {cfg.general.num_points})"
        )
    return errors


def validate_encoder_checkpoint_path_and_declared_dims(
    cfg: ExpCfg,
    *,
    strict_paths: bool = True,
) -> list[str]:
    errors: list[str] = []
    try:
        active_encoder_checkpoint = cfg.model.encoder.checkpoint_path
    except ValueError as exc:
        errors.append(str(exc))
        active_encoder_checkpoint = None
    checkpoint_specs = [
        ("ModelCfg.encoder.checkpoint_path", active_encoder_checkpoint),
        ("ModelCfg.tce.checkpoint_path", cfg.model.tce.checkpoint_path),
        ("ModelCfg.p2v.checkpoint_path", cfg.model.p2v.checkpoint_path),
        ("ModelCfg.icp.checkpoint_path", cfg.model.icp.checkpoint_path),
        (
            "ModelCfg.pretrained_encoder.checkpoint_path",
            cfg.model.pretrained_encoder.checkpoint_path,
        ),
        (
            "CheckpointPolicyCfg.resume_checkpoint",
            cfg.pretrain.checkpoint_policy.resume_checkpoint,
        ),
    ]
    if cfg.rl.enabled:
        checkpoint_specs.append(("RLCfg.encoder_checkpoint", cfg.rl.encoder_checkpoint))
    for field_name, value in checkpoint_specs:
        if not value:
            continue
        path = Path(value)
        _require_existing_file(errors, path, strict_paths, field_name)
        if _should_validate_tce_manifest(cfg, field_name):
            if cfg.model.tce.output_dim <= 0:
                errors.append(
                    "TCECfg.output_dim must be > 0 when encoder checkpoint is set"
                )
            if not cfg.model.tce.encoder_type.strip():
                errors.append(
                    "TCECfg.encoder_type must be non-empty when encoder checkpoint is set"
                )
            if strict_paths and path.exists():
                _validate_encoder_manifest_dims(errors, cfg, path)
    return errors


def _should_validate_tce_manifest(cfg: ExpCfg, field_name: str) -> bool:
    backend = cfg.model.encoder_backend.strip().lower()
    if backend in {"tg"}:
        backend = "tce"
    if backend in {"p2v"}:
        backend = "point2vec"
    if field_name == "ModelCfg.encoder.checkpoint_path":
        return backend == "tce"
    if field_name == "ModelCfg.tce.checkpoint_path":
        return True
    if field_name == "ModelCfg.p2v.checkpoint_path":
        return False
    if field_name == "ModelCfg.icp.checkpoint_path":
        return False
    if field_name == "ModelCfg.pretrained_encoder.checkpoint_path":
        return cfg.model.pretrained_encoder.adapter == "tce_strict"
    if field_name == "RLCfg.encoder_checkpoint":
        return backend == "tce"
    return False


def validate_contact_schema_version(cfg: ExpCfg) -> list[str]:
    errors: list[str] = []
    expected = "contact_pt_v1"
    if cfg.contact_gen.schema_version != expected:
        errors.append(
            f"ContactGenCfg.schema_version must be {expected!r}, "
            f"got {cfg.contact_gen.schema_version!r}"
        )
    return errors


def validate_isaac_task_and_rsl_rl_entrypoint_strings(cfg: ExpCfg) -> list[str]:
    errors: list[str] = []
    if not cfg.rl.enabled:
        return errors
    task_id = cfg.rl.isaac_task_id or cfg.rl.task_id or cfg.rl.task_name
    if not _non_empty_string(task_id):
        errors.append("RLCfg.enabled requires a non-empty Isaac task id string")
    if not _non_empty_string(cfg.rl.rsl_rl_cfg_entry_point):
        errors.append("RLCfg.enabled requires a non-empty rsl_rl_cfg_entry_point string")
    return errors


def _validate_encoder_manifest_dims(errors: list[str], cfg: ExpCfg, checkpoint_path: Path) -> None:
    manifest = _read_checkpoint_manifest(checkpoint_path)
    if manifest is None:
        return
    declared = _extract_declared_dims(manifest)
    if declared is None:
        errors.append(
            f"Encoder checkpoint manifest lacks declared model dims: {checkpoint_path}"
        )
        return
    _compare_declared_dim(
        errors,
        declared,
        ("num_points", "num_pts"),
        cfg.model.tce.num_points,
        checkpoint_path,
    )
    _compare_declared_dim(errors, declared, ("patch_size",), cfg.model.tce.patch_size, checkpoint_path)
    _compare_declared_dim(
        errors,
        declared,
        ("encoder_channel",),
        cfg.model.tce.encoder_channel,
        checkpoint_path,
    )
    _compare_declared_dim(errors, declared, ("vit_depth",), cfg.model.tce.vit_depth, checkpoint_path)
    _compare_declared_dim(errors, declared, ("vit_heads",), cfg.model.tce.vit_heads, checkpoint_path)


def _read_checkpoint_manifest(checkpoint_path: Path) -> Mapping[str, Any] | None:
    candidates = []
    if checkpoint_path.suffix == ".json":
        candidates.append(checkpoint_path)
    candidates.extend(
        [
            checkpoint_path.with_suffix(".manifest.json"),
            checkpoint_path.parent / "manifest.json",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = read_json(candidate)
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
    return None


def _extract_declared_dims(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("model_dims", "encoder_dims", "model_config"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            return value
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping):
        return _extract_declared_dims(metadata)
    config_dump = manifest.get("config_dump")
    if isinstance(config_dump, Mapping):
        model_cfg = config_dump.get("model")
        if isinstance(model_cfg, Mapping):
            return model_cfg
    return None


def _compare_declared_dim(
    errors: list[str],
    declared: Mapping[str, Any],
    keys: tuple[str, ...],
    expected: int,
    checkpoint_path: Path,
) -> None:
    for key in keys:
        if key not in declared:
            continue
        try:
            actual = int(declared[key])
        except (TypeError, ValueError):
            errors.append(f"Encoder checkpoint declared dim {key} is not an int: {checkpoint_path}")
            return
        if actual != expected:
            errors.append(
                f"Encoder checkpoint declared {key}={actual} does not match config value "
                f"{expected}: {checkpoint_path}"
            )
        return


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_json(errors: list[str], path: Path, label: str) -> Any | None:
    try:
        return read_json(path)
    except Exception as exc:
        errors.append(f"{label} must be readable JSON: {exc}")
        return None


def _json_non_empty(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return bool(payload)
    if isinstance(payload, list):
        return bool(payload)
    return False


def _require_json_non_empty(
    errors: list[str],
    paths: ProjectPaths,
    key: str,
    strict_paths: bool,
) -> None:
    path = _require_path(errors, paths, key, strict_paths)
    if path is None or not strict_paths:
        return
    payload = _read_json(errors, path, key)
    if payload is not None and not _json_non_empty(payload):
        errors.append(f"{key} must contain non-empty JSON")


def _require_existing_json_non_empty(
    errors: list[str],
    path: Path,
    label: str,
    strict_paths: bool,
) -> None:
    _require_existing_file(errors, path, strict_paths, label)
    if not strict_paths or not path.exists():
        return
    payload = _read_json(errors, path, label)
    if payload is not None and not _json_non_empty(payload):
        errors.append(f"{label} must contain non-empty JSON")


def _require_existing_file(
    errors: list[str],
    path: Path,
    strict_paths: bool,
    label: str = "File",
) -> None:
    if strict_paths and not path.exists():
        errors.append(f"{label} does not exist: {path}")


def _require_path(
    errors: list[str],
    paths: ProjectPaths,
    key: str,
    strict_paths: bool,
) -> Path | None:
    try:
        return require_path(paths, key, must_exist=strict_paths)
    except PathsConfigError as exc:
        errors.append(str(exc))
        return None


def _resolve_config_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()

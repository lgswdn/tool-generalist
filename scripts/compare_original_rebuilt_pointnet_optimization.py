#!/usr/bin/env python3
"""Compare original/rebuilt PointNet GG optimization without simulation.

The report aligns TensorBoard learning curves and measures DGN-to-GG parameter
drift by policy module.  It also records the known environment/configuration
differences so a curve difference is never silently attributed to the encoder.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_DGN = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/panda_general_oracle_pointcloud_pointnet_full_yes_5k/"
    "20260719T092442Z/model_best.pt"
)
OLD_GG_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/no-contact/"
    "oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_gg_from_full_yes_5k/"
    "20260719T202622Z"
)
NEW_DGN = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "ce_prl_oracle_rebuild_d12_pointnet_dgn_5k/no-contact/"
    "oracle_pointcloud_pointnet/ce_prl_oracle_rebuild_d12_pointnet_dgn_5k/"
    "20260802T053747Z/model_best.pt"
)
NEW_GG_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "ce_prl_oracle_rebuild_d12_pointnet_gg_15k/no-contact/"
    "oracle_pointcloud_pointnet/ce_prl_oracle_rebuild_d12_pointnet_gg_15k/"
    "20260802T142604Z"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/analysis/oracle_pointnet_first_round/"
    "original_vs_rebuilt_optimization.json"
)

SCALAR_TAGS = (
    "Episode/recent_success_rate",
    "Episode/success_rate",
    "Episode_Reward/object_goal_tracking",
    "Episode_Termination/reached",
    "Episode_Termination/object_dropped",
    "Policy/mean_noise_std",
    "Loss/entropy",
    "Loss/surrogate",
    "Loss/value_function",
    "Train/mean_reward",
)
CONFIG_PATHS = (
    "paths_yaml",
    "general.tools_selected_json",
    "general.rl_objects_manifest",
    "model.encoder_backend",
    "model.pretrained_encoder.checkpoint_path",
    "rl.actor_critic_class",
    "rl.freeze_encoder",
    "rl.ppo.max_iterations",
    "rl.env.generated_parallel_finger_velocity_limit_m_s",
    "rl.env.robot_mode",
    "rl.observation.tool_cloud_source",
)
GROUP_PREFIXES = (
    ("encoder", ("encoder.",)),
    ("actor_cross_attention", ("state_cross_all.",)),
    ("critic_cross_attention", ("critic_state_cross_all.",)),
    ("actor_fusion", ("fusion_mlp.",)),
    ("critic_fusion", ("critic_fusion_mlp.",)),
    ("actor_head", ("actor.",)),
    ("critic_head", ("critic.",)),
    ("exploration", ("std",)),
)


def _checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint lacks model_state_dict: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"Checkpoint has non-tensor model state entries: {path}")
    return state, {"path": str(path), "iteration": payload.get("iter")}


def _classify(key: str) -> str:
    matches = [
        name
        for name, prefixes in GROUP_PREFIXES
        if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Parameter must map to exactly one module group: {key}")
    return matches[0]


def _drift(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        raise RuntimeError(
            f"DGN/GG state keys differ; missing={missing[:10]} extra={extra[:10]}"
        )
    totals: dict[str, dict[str, float | int]] = {}
    for key in sorted(before):
        a = before[key].detach().double().reshape(-1)
        b = after[key].detach().double().reshape(-1)
        if a.shape != b.shape:
            raise RuntimeError(f"DGN/GG parameter shape differs for {key}")
        group = _classify(key)
        row = totals.setdefault(
            group,
            {
                "parameters": 0,
                "before_sq": 0.0,
                "after_sq": 0.0,
                "change_sq": 0.0,
                "dot": 0.0,
            },
        )
        row["parameters"] += a.numel()
        row["before_sq"] += float(torch.dot(a, a))
        row["after_sq"] += float(torch.dot(b, b))
        row["change_sq"] += float(torch.dot(b - a, b - a))
        row["dot"] += float(torch.dot(a, b))
    total_change = sum(float(row["change_sq"]) for row in totals.values())
    if total_change <= 0:
        raise RuntimeError("DGN and GG checkpoints have no parameter change")
    result = {}
    for group, row in totals.items():
        before_norm = math.sqrt(float(row["before_sq"]))
        after_norm = math.sqrt(float(row["after_sq"]))
        change_norm = math.sqrt(float(row["change_sq"]))
        result[group] = {
            "parameters": int(row["parameters"]),
            "before_l2": before_norm,
            "after_l2": after_norm,
            "change_l2": change_norm,
            "relative_l2_change": change_norm / max(before_norm, 1e-30),
            "cosine": float(row["dot"]) / max(before_norm * after_norm, 1e-30),
            "fraction_of_total_squared_update": float(row["change_sq"]) / total_change,
        }
    return result


def _event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one TensorBoard event file in {run_dir}, found {len(files)}"
        )
    return files[0]


def _curve_summary(events: list[Any]) -> dict[str, Any]:
    if not events:
        raise RuntimeError("Cannot summarize an empty scalar curve")
    points = [(int(item.step), float(item.value)) for item in events]
    peak_step, peak = max(points, key=lambda item: item[1])
    first_step, last_step = points[0][0], points[-1][0]
    area = 0.0
    for (step_a, value_a), (step_b, value_b) in zip(points, points[1:]):
        area += (step_b - step_a) * (value_a + value_b) * 0.5
    span = max(last_step - first_step, 1)
    milestones = {}
    for target in (1000, 3000, 5000, 10000, 15000):
        eligible = [item for item in points if item[0] <= target]
        milestones[str(target)] = eligible[-1][1] if eligible else None
    thresholds = {}
    for threshold in (0.2, 0.5, 0.75, 0.8):
        reached = [step for step, value in points if value >= threshold]
        thresholds[str(threshold)] = reached[0] if reached else None
    return {
        "points": len(points),
        "first": {"step": first_step, "value": points[0][1]},
        "last": {"step": last_step, "value": points[-1][1]},
        "peak": {"step": peak_step, "value": peak},
        "step_normalized_auc": area / span,
        "value_at_or_before_step": milestones,
        "first_step_at_threshold": thresholds,
    }


def _curves(run_dir: Path) -> dict[str, Any]:
    accumulator = EventAccumulator(
        str(_event_file(run_dir)), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    missing = sorted(set(SCALAR_TAGS) - available)
    if missing:
        raise RuntimeError(f"Required TensorBoard scalar tags are missing: {missing}")
    return {
        tag: _curve_summary(accumulator.Scalars(tag)) for tag in SCALAR_TAGS
    }


def _nested_field(payload: dict[str, Any], dotted: str) -> dict[str, Any]:
    value: Any = payload
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return {"present": False, "value": None}
        value = value[key]
    return {"present": True, "value": value}


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config_dump")
    if not isinstance(config, dict):
        raise RuntimeError(f"Manifest lacks config_dump: {path}")
    return {
        "path": str(path),
        "experiment": payload.get("exp_cfg_name"),
        "created_at": payload.get("created_at"),
        "status": payload.get("status"),
        "selected_config": {
            key: _nested_field(config, key) for key in CONFIG_PATHS
        },
    }


def _configuration_differences(
    old: dict[str, Any], new: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    old_config = old["selected_config"]
    new_config = new["selected_config"]
    return {
        key: {"original": old_config[key], "rebuilt": new_config[key]}
        for key in CONFIG_PATHS
        if old_config[key] != new_config[key]
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dgn", type=Path, default=OLD_DGN)
    parser.add_argument("--old-gg-dir", type=Path, default=OLD_GG_DIR)
    parser.add_argument("--new-dgn", type=Path, default=NEW_DGN)
    parser.add_argument("--new-gg-dir", type=Path, default=NEW_GG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    required = (
        args.old_dgn,
        args.old_gg_dir / "model_best.pt",
        args.old_gg_dir / "manifest.json",
        args.new_dgn,
        args.new_gg_dir / "model_best.pt",
        args.new_gg_dir / "manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing required inputs:\n" + "\n".join(missing))
    return args


def main() -> None:
    args = _parse()
    old_dgn, old_dgn_meta = _checkpoint(args.old_dgn)
    old_gg, old_gg_meta = _checkpoint(args.old_gg_dir / "model_best.pt")
    new_dgn, new_dgn_meta = _checkpoint(args.new_dgn)
    new_gg, new_gg_meta = _checkpoint(args.new_gg_dir / "model_best.pt")
    old_manifest = _manifest(args.old_gg_dir)
    new_manifest = _manifest(args.new_gg_dir)
    result = {
        "schema_version": "original_rebuilt_pointnet_optimization_v1",
        "scope": "checkpoint_and_learning_curve_analysis_without_simulation",
        "original": {
            "dgn_checkpoint": old_dgn_meta,
            "gg_checkpoint": old_gg_meta,
            "manifest": old_manifest,
            "dgn_to_gg_parameter_drift": _drift(old_dgn, old_gg),
            "learning_curves": _curves(args.old_gg_dir),
        },
        "rebuilt": {
            "dgn_checkpoint": new_dgn_meta,
            "gg_checkpoint": new_gg_meta,
            "manifest": new_manifest,
            "dgn_to_gg_parameter_drift": _drift(new_dgn, new_gg),
            "learning_curves": _curves(args.new_gg_dir),
        },
        "configuration_differences": _configuration_differences(
            old_manifest, new_manifest
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name in ("original", "rebuilt"):
        success = result[name]["learning_curves"]["Episode/recent_success_rate"]
        encoder = result[name]["dgn_to_gg_parameter_drift"]["encoder"]
        print(
            f"{name}: success peak={success['peak']['value']:.4f} "
            f"at={success['peak']['step']} encoder_relative_drift="
            f"{encoder['relative_l2_change']:.6f}"
        )
    print(
        "Recorded configuration differences: "
        + ", ".join(result["configuration_differences"])
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

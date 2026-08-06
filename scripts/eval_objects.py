#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy across configured object candidates and report per-object success rates."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

from utils.artifacts.resolver import resolve_artifacts
from utils.config.loader import load_exp_cfg
from utils.experiment.object_eval import (
    load_candidate_entries,
    merge_rows_by_object,
    scale_stats,
)
from utils.experiment.rl_runtime_spec import RUNTIME_SPEC_ENV_VAR, validate_runtime_spec


def _distributed_rank_info(distributed: bool) -> tuple[int, int]:
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size
    return 0, 1


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) == 0:
        raise ValueError(f"Could not create a safe filename from: {name}")
    return safe


def _resolve_checkpoint_arg(runtime_spec_path: str, checkpoint: str | None) -> str:
    if checkpoint is not None:
        return checkpoint

    spec_dir = Path(runtime_spec_path).parent
    best_path = spec_dir / "model_best.pt"
    if best_path.is_file():
        return str(best_path)

    candidates = []
    for path in spec_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])

    for name in ("model.pt", "best.pt"):
        path = spec_dir / name
        if path.is_file():
            return str(path)

    raise FileNotFoundError(
        f"--checkpoint was not provided and no model_best.pt/model_*.pt/model.pt/best.pt was found in {spec_dir}"
    )


def _latest_runtime_spec_for_config(config: str) -> str:
    cfg = load_exp_cfg(config)
    rl_refs = [ref for ref in resolve_artifacts(cfg).stages if ref.stage == "rl"]
    if not rl_refs:
        raise FileNotFoundError(f"No RL artifact location could be resolved for config: {config}")
    run_root = rl_refs[0].directory.parent
    candidates = []
    if run_root.is_dir():
        for path in run_root.iterdir():
            spec_path = path / "rl_runtime_spec.json"
            if path.is_dir() and spec_path.is_file():
                candidates.append((path.stat().st_mtime, path.name, spec_path))
    if not candidates:
        raise FileNotFoundError(
            f"No rl_runtime_spec.json found under latest-run root for config {config}: {run_root}"
        )
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


def _resolve_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _get_candidates_json(paths_cfg: dict) -> str:
    if isinstance(paths_cfg.get("dgn"), dict) and paths_cfg["dgn"].get("candidates_json"):
        return str(paths_cfg["dgn"]["candidates_json"])
    if isinstance(paths_cfg.get("objects"), dict) and paths_cfg["objects"].get("candidates_json"):
        return str(paths_cfg["objects"]["candidates_json"])
    raise ValueError("paths_yaml must define dgn.candidates_json or objects.candidates_json")


def _ensure_legacy_aliases(paths_cfg: dict) -> None:
    if "dgn" not in paths_cfg and isinstance(paths_cfg.get("objects"), dict):
        paths_cfg["dgn"] = copy.deepcopy(paths_cfg["objects"])
    if isinstance(paths_cfg.get("objects"), dict) and isinstance(paths_cfg.get("dgn"), dict):
        paths_cfg["objects"]["candidates_json"] = paths_cfg["dgn"].get("candidates_json")

    tools_cfg = paths_cfg.get("tools")
    if isinstance(tools_cfg, dict):
        if "robots_usd_dir" not in tools_cfg and tools_cfg.get("robots_usd_root"):
            tools_cfg["robots_usd_dir"] = tools_cfg["robots_usd_root"]
        if "tools_adjusted_json" not in tools_cfg and tools_cfg.get("tools_json"):
            tools_cfg["tools_adjusted_json"] = tools_cfg["tools_json"]


def _prepare_paths_yaml_for_rank(
    paths_yaml: str,
    rank: int,
    world_size: int,
    *,
    replicate_objects_across_ranks: bool,
) -> tuple[str, list[str | dict[str, Any]], list[str | dict[str, Any]]]:
    source_yaml = Path(paths_yaml).expanduser().resolve()
    with source_yaml.open("r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f)
    if not isinstance(paths_cfg, dict):
        raise ValueError(f"Expected paths yaml to contain a mapping: {source_yaml}")
    _ensure_legacy_aliases(paths_cfg)

    base_dir = source_yaml.parent
    candidates_json = _resolve_path(_get_candidates_json(paths_cfg), base_dir)
    all_candidates = load_candidate_entries(candidates_json, "candidates_json")
    rank_candidates = (
        list(all_candidates)
        if replicate_objects_across_ranks
        else all_candidates[rank::world_size]
    )
    if len(rank_candidates) == 0:
        raise ValueError(
            f"Rank {rank} received no objects from {len(all_candidates)} candidates "
            f"with world_size={world_size}."
        )

    temp_root = os.path.join(tempfile.gettempdir(), "tool_generalist_eval_objects")
    os.makedirs(temp_root, exist_ok=True)
    rank_candidates_json = os.path.join(temp_root, f"candidates_rank_{rank}_of_{world_size}.json")
    rank_paths_yaml = os.path.join(temp_root, f"paths_rank_{rank}_of_{world_size}.yaml")

    with open(rank_candidates_json, "w", encoding="utf-8") as f:
        json.dump(rank_candidates, f, ensure_ascii=False, indent=2)

    rank_paths_cfg = copy.deepcopy(paths_cfg)
    _ensure_legacy_aliases(rank_paths_cfg)
    rank_paths_cfg["dgn"]["candidates_json"] = rank_candidates_json
    if isinstance(rank_paths_cfg.get("objects"), dict):
        rank_paths_cfg["objects"]["candidates_json"] = rank_candidates_json

    with open(rank_paths_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(rank_paths_cfg, f, sort_keys=False)

    return rank_paths_yaml, all_candidates, rank_candidates


def _backfill_runtime_spec_defaults(spec: dict) -> None:
    """Fill policy fields that older runtime specs omitted but current configs default."""

    policy = spec.get("policy_params")
    if not isinstance(policy, dict):
        return

    observation = spec.get("observation_params")
    if not isinstance(observation, dict):
        observation = {}

    policy.setdefault(
        "model_input_centering",
        observation.get("model_input_centering", "bbox_center"),
    )
    policy.setdefault("relative_translation_query_tokens", 2)
    policy.setdefault("reuse_pretrain_pose_cross_attn", False)
    if "vit_attention_mode" not in policy:
        raise ValueError(
            "Runtime spec is missing required policy_params.vit_attention_mode"
        )


parser = argparse.ArgumentParser(
    description="Evaluate the configured policy and write a per-object success-rate table."
)
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--config",
    type=str,
    default=None,
    help="Experiment config exposing EXP_CFG. Uses the latest RL run/checkpoint for this config.",
)
source_group.add_argument(
    "--runtime_spec",
    type=str,
    help="Path to the rl_runtime_spec.json written with the checkpoint.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the RSL-RL checkpoint. Defaults to model_best.pt beside the runtime spec.",
)
parser.add_argument(
    "--paths_yaml",
    type=str,
    default=None,
    help="Optional paths.yaml override for the object candidate set and assets.",
)
parser.add_argument("--task", type=str, default=None, help="Task name. Defaults to runtime_spec['task_id'].")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate per rank.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of completed episodes per object.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run evaluation across multiple GPUs.")
parser.add_argument(
    "--replicate_objects_across_ranks",
    action="store_true",
    default=False,
    help=(
        "Load the complete object candidate set on every distributed rank instead of sharding it. "
        "Use with --require_one_env_per_object and --num_envs equal to the candidate count for an "
        "exact one-object-per-env layout on every GPU."
    ),
)
parser.add_argument(
    "--require_one_env_per_object",
    action="store_true",
    default=False,
    help="Fail unless the number of loaded objects exactly equals --num_envs on every rank.",
)
parser.add_argument(
    "--randomize_grippers",
    action="store_true",
    default=False,
    help=(
        "Randomly assign a generated gripper/tool asset to every env using the runtime-spec seed "
        "and rank-aware global env ids. The assignment remains fixed for that env during evaluation."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be positive")

runtime_spec_source = args_cli.runtime_spec or _latest_runtime_spec_for_config(args_cli.config)
runtime_spec_path = os.path.abspath(os.path.normpath(runtime_spec_source))
with open(runtime_spec_path, "r", encoding="utf-8") as f:
    runtime_spec = json.load(f)

runtime_robot_mode = str(runtime_spec.get("env_params", {}).get("robot_mode", "tool"))
if runtime_robot_mode not in {"tool", "bare_franka", "official_panda_gripper", "generated_gripper"}:
    parser.error(
        "runtime_spec env_params.robot_mode must be one of: "
        "tool, bare_franka, official_panda_gripper, generated_gripper"
    )

if args_cli.task is None:
    args_cli.task = runtime_spec.get("task_id")
if not args_cli.task:
    parser.error("--task is required when runtime_spec does not contain task_id")

runtime_spec_paths_yaml = runtime_spec.get("paths_yaml")
if not runtime_spec_paths_yaml:
    parser.error("runtime_spec must contain paths_yaml")
runtime_spec_paths_yaml = os.path.abspath(os.path.normpath(runtime_spec_paths_yaml))
paths_yaml = (
    os.path.abspath(os.path.normpath(args_cli.paths_yaml))
    if args_cli.paths_yaml is not None
    else runtime_spec_paths_yaml
)
checkpoint_arg = _resolve_checkpoint_arg(runtime_spec_path, args_cli.checkpoint)

rank, world_size = _distributed_rank_info(args_cli.distributed)
local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args_cli.distributed else 0
os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = str(rank)
os.environ["TOOL_GENERALIST_LOCAL_RANK"] = str(local_rank)
os.environ["TOOL_GENERALIST_WORLD_SIZE"] = str(world_size)
rank_paths_yaml, all_candidate_entries, rank_candidate_entries = _prepare_paths_yaml_for_rank(
    paths_yaml,
    rank,
    world_size,
    replicate_objects_across_ranks=bool(args_cli.replicate_objects_across_ranks),
)
os.environ["TOOL_GENERALIST_PATHS_YAML"] = rank_paths_yaml

eval_runtime_spec = copy.deepcopy(runtime_spec)
eval_runtime_spec["num_envs"] = args_cli.num_envs
if isinstance(eval_runtime_spec.get("env_params"), dict):
    eval_runtime_spec["env_params"]["num_envs"] = args_cli.num_envs
eval_runtime_spec["paths_yaml"] = rank_paths_yaml
asset_assignment = eval_runtime_spec.setdefault("asset_assignment_params", {})
asset_assignment["randomize_object_assignment"] = False
if args_cli.randomize_grippers:
    if runtime_robot_mode not in {"tool", "generated_gripper"}:
        parser.error(
            "--randomize_grippers requires runtime_spec env_params.robot_mode to be "
            "'tool' or 'generated_gripper'"
        )
    asset_assignment["randomize_tool_assignment"] = True
_backfill_runtime_spec_defaults(eval_runtime_spec)
eval_runtime_spec_path = os.path.join(
    tempfile.gettempdir(),
    "tool_generalist_eval_objects",
    f"rl_runtime_spec_rank_{rank}_of_{world_size}_envs_{args_cli.num_envs}.json",
)
os.makedirs(os.path.dirname(eval_runtime_spec_path), exist_ok=True)
validate_runtime_spec(eval_runtime_spec, eval_runtime_spec_path)
with open(eval_runtime_spec_path, "w", encoding="utf-8") as f:
    json.dump(eval_runtime_spec, f, ensure_ascii=False, indent=2)
os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path

print(
    f"[INFO][rank {rank}]: Evaluating {len(rank_candidate_entries)}/{len(all_candidate_entries)} "
    f"object candidates for this rank "
    f"(replicated={bool(args_cli.replicate_objects_across_ranks)}).",
    flush=True,
)
print(
    f"[INFO][rank {rank}]: Random gripper assignment: "
    f"{bool(asset_assignment.get('randomize_tool_assignment', False))}",
    flush=True,
)
if args_cli.paths_yaml is not None:
    print(
        f"[INFO][rank {rank}]: Overriding runtime spec paths_yaml: "
        f"{runtime_spec_paths_yaml} -> {paths_yaml}",
        flush=True,
    )

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.distributed as dist

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import env_tool as env_tool_module
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    GENERATED_GRIPPER_ASSET_INDICES_BY_ENV,
    GENERATED_GRIPPER_DATA,
    OBJECT_ASSET_CFGS,
    TOOL_ASSET_INDICES_BY_ENV,
    TOOL_DATA,
    get_object_index_for_env,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.events import (
    get_rigid_body_scale,
)


def _object_names_from_loaded_data() -> list[str]:
    object_names = []
    for index, cfg in enumerate(OBJECT_ASSET_CFGS):
        obj_path = getattr(cfg, "obj_path", None)
        usd_path = getattr(cfg, "usd_path", None)
        if obj_path:
            object_names.append(Path(str(obj_path)).stem)
        elif usd_path:
            object_names.append(Path(str(usd_path)).stem)
        else:
            object_names.append(f"object_{index:04d}")
    if len(object_names) == 0:
        raise ValueError("No objects were loaded into OBJECT_ASSET_CFGS.")
    return object_names


def _init_object_rows(object_names: list[str]) -> list[dict]:
    return [{"name": name, "episodes": 0, "successes": 0, "episode_scales": []} for name in object_names]


def _objects_finished(object_rows: list[dict], object_indices: list[int], episodes_per_object: int) -> bool:
    for object_idx in object_indices:
        if int(object_rows[object_idx]["episodes"]) < episodes_per_object:
            return False
    return True


def _count_finished_episodes(object_rows: list[dict], episodes_per_object: int) -> int:
    count = 0
    for row in object_rows:
        count += min(int(row["episodes"]), episodes_per_object)
    return count


def _add_success_rates(object_rows: list[dict], row_rank: int) -> list[dict]:
    rows_with_rates = []
    for row in object_rows:
        episodes = int(row["episodes"])
        successes = int(row["successes"])
        success_rate = float(successes) / float(episodes) if episodes > 0 else 0.0
        row_scale_stats = scale_stats(row.get("episode_scales", []))
        rows_with_rates.append(
            {
                "name": row["name"],
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                **row_scale_stats,
                "rank": row_rank,
                "ranks": [row_rank],
            }
        )
    return rows_with_rates


def _format_scale_value(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _write_per_object_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "object",
                "episodes",
                "successes",
                "success_rate",
                "scale_mean",
                "scale_min",
                "scale_max",
                "scale_values",
                "rank",
                "ranks",
            ]
        )
        for row in sorted(rows, key=lambda item: item["name"]):
            writer.writerow(
                [
                    row["name"],
                    row["episodes"],
                    row["successes"],
                    row["success_rate"],
                    _format_scale_value(row.get("scale_mean")),
                    _format_scale_value(row.get("scale_min")),
                    _format_scale_value(row.get("scale_max")),
                    " ".join(_format_scale_value(value) for value in row.get("scale_values", [])),
                    row.get("rank"),
                    " ".join(str(value) for value in row.get("ranks", [])),
                ]
            )


def _print_table(rows: list[dict]) -> None:
    sorted_rows = sorted(rows, key=lambda item: item["name"])
    object_width = max([len("object")] + [len(str(row["name"])) for row in sorted_rows])
    print("\n========== Per-Object Success Rate ==========")
    print(f"{'object':<{object_width}}  episodes  successes  success_rate  scale_mean  scale_range")
    print(f"{'-' * object_width}  --------  ---------  ------------  ----------  -----------")
    for row in sorted_rows:
        scale_min = row.get("scale_min")
        scale_max = row.get("scale_max")
        scale_range = (
            ""
            if scale_min is None or scale_max is None
            else f"{float(scale_min):.4f}-{float(scale_max):.4f}"
        )
        print(
            f"{row['name']:<{object_width}}  "
            f"{int(row['episodes']):>8}  "
            f"{int(row['successes']):>9}  "
            f"{float(row['success_rate']) * 100.0:>11.2f}%  "
            f"{_format_scale_value(row.get('scale_mean')):>10}  "
            f"{scale_range:>11}"
        )
    print("=============================================\n")


def _write_summary(log_dir: str, resume_path: str, rows: list[dict]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    summary_path = os.path.join(log_dir, "eval_objects_summary.json")
    csv_path = os.path.join(log_dir, "eval_objects_per_object.csv")
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "runtime_spec": runtime_spec_path,
        "effective_runtime_spec": eval_runtime_spec_path,
        "runtime_spec_paths_yaml": runtime_spec_paths_yaml,
        "effective_source_paths_yaml": paths_yaml,
        "paths_yaml_override": args_cli.paths_yaml,
        "rank_paths_yaml": rank_paths_yaml,
        "robot_mode": runtime_robot_mode,
        "world_size": world_size,
        "num_envs_per_rank": args_cli.num_envs,
        "episodes_per_object": args_cli.num_episodes,
        "aggregate_episodes_per_object": (
            args_cli.num_episodes * world_size
            if args_cli.replicate_objects_across_ranks
            else args_cli.num_episodes
        ),
        "object_batch_size": args_cli.num_envs,
        "source_candidate_count": len(all_candidate_entries),
        "rank_candidate_count": len(rank_candidate_entries),
        "replicate_objects_across_ranks": bool(args_cli.replicate_objects_across_ranks),
        "require_one_env_per_object": bool(args_cli.require_one_env_per_object),
        "randomize_grippers": bool(args_cli.randomize_grippers),
        "effective_randomize_tool_assignment": bool(
            eval_runtime_spec.get("asset_assignment_params", {}).get("randomize_tool_assignment", False)
        ),
        "objects": len(rows),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": success_rate,
        "per_object": sorted(rows, key=lambda item: item["name"]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_per_object_csv(csv_path, rows)
    _print_table(rows)

    print("========== Object Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"World Size: {world_size}")
    print(f"Objects: {len(rows)}")
    print(f"Episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100.0:.2f}%")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    print("===============================================\n")


def _apply_object_batch(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    active_object_indices: list[int],
    num_envs: int,
) -> list[int]:
    if len(active_object_indices) == 0:
        raise ValueError("active_object_indices must be non-empty")

    object_indices = [
        active_object_indices[env_id % len(active_object_indices)]
        for env_id in range(num_envs)
    ]
    object_spawn_cfgs = [env_tool_module.OBJECT_ASSET_CFGS[index] for index in object_indices]
    env_tool_module.OBJECT_ASSET_INDICES_BY_ENV[:] = object_indices
    env_tool_module.OBJECT_ASSET_CFGS_BY_ENV[:] = object_spawn_cfgs
    env_tool_module.OBJECT_SPAWN_ASSET_INDICES[:] = object_indices
    env_tool_module.OBJECT_ASSET_CFGS_FOR_SPAWN[:] = object_spawn_cfgs
    if hasattr(env_cfg.scene, "object"):
        env_cfg.scene.object.spawn.assets_cfg = env_tool_module.OBJECT_ASSET_CFGS_FOR_SPAWN
    return object_indices


def _env_object_scales(base_env, num_envs: int) -> torch.Tensor:
    env_ids = torch.arange(num_envs, device=base_env.device, dtype=torch.long)
    scales = get_rigid_body_scale(base_env, SceneEntityCfg("object"), env_ids)
    return scales.detach().cpu()


def _scale_scalar(scale_xyz: torch.Tensor) -> float:
    values = [float(value) for value in scale_xyz.tolist()]
    return sum(values) / len(values)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        if agent_cfg.seed is not None:
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

    resume_path = retrieve_file_path(checkpoint_arg)
    log_dir = os.path.dirname(runtime_spec_path)

    object_names = _object_names_from_loaded_data()
    if args_cli.require_one_env_per_object:
        expected_objects = len(rank_candidate_entries)
        if len(object_names) != expected_objects:
            raise ValueError(
                "--require_one_env_per_object requires every candidate to load exactly once, but "
                f"rank {rank} loaded {len(object_names)} objects from {expected_objects} candidates."
            )
        if args_cli.num_envs != len(object_names):
            raise ValueError(
                "--require_one_env_per_object requires --num_envs to equal the loaded object count: "
                f"num_envs={args_cli.num_envs}, objects={len(object_names)}, rank={rank}."
            )

    if args_cli.randomize_grippers:
        if runtime_robot_mode == "generated_gripper":
            gripper_assignments = GENERATED_GRIPPER_ASSET_INDICES_BY_ENV
            gripper_count = len(GENERATED_GRIPPER_DATA)
        else:
            gripper_assignments = TOOL_ASSET_INDICES_BY_ENV
            gripper_count = len(TOOL_DATA)
        if len(gripper_assignments) != args_cli.num_envs:
            raise ValueError(
                "Random gripper assignment did not produce one assignment per env: "
                f"assignments={len(gripper_assignments)}, num_envs={args_cli.num_envs}, rank={rank}."
            )
        print(
            f"[INFO][rank {rank}]: Randomly assigned {len(set(gripper_assignments))}/"
            f"{gripper_count} available grippers across {len(gripper_assignments)} envs.",
            flush=True,
        )

    object_rows = _init_object_rows(object_names)
    total_required = len(object_names) * args_cli.num_episodes
    object_batches = [
        list(range(start, min(start + args_cli.num_envs, len(object_names))))
        for start in range(0, len(object_names), args_cli.num_envs)
    ]
    print(
        f"[INFO][rank {rank}]: Evaluating {len(object_names)} objects in "
        f"{len(object_batches)} batch(es) of up to {args_cli.num_envs} active objects.",
        flush=True,
    )

    total_step_count = 0
    inference_policy = None
    for batch_index, active_object_indices in enumerate(object_batches):
        if not simulation_app.is_running():
            break
        assigned_object_indices = _apply_object_batch(env_cfg, active_object_indices, args_cli.num_envs)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(
            f"[INFO][rank {rank}]: Preparing object_batch={batch_index + 1}/{len(object_batches)} "
            f"active_objects={len(active_object_indices)}",
            flush=True,
        )
        if inference_policy is None:
            print(f"[INFO][rank {rank}]: Loading model checkpoint from: {resume_path}", flush=True)
            ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(resume_path)
            inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)
            ppo_runner.env = None

        if not hasattr(env.unwrapped, "episode_success_buf"):
            raise AttributeError("Environment does not have episode_success_buf.")

        num_envs = env.unwrapped.num_envs
        env_to_object_idx = torch.tensor(
            [get_object_index_for_env(env_id) for env_id in range(num_envs)],
            dtype=torch.long,
        )
        env_to_object_scale = _env_object_scales(env.unwrapped, num_envs)
        active_set = set(active_object_indices)
        assigned_set = set(int(index) for index in assigned_object_indices)
        if not active_set.issubset(assigned_set):
            missing_names = [object_names[index] for index in sorted(active_set.difference(assigned_set))]
            raise ValueError(
                f"Rank {rank} object batch {batch_index + 1} is missing assigned envs for: {missing_names}"
            )

        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
        batch_step_count = 0

        while (
            not _objects_finished(object_rows, active_object_indices, args_cli.num_episodes)
            and simulation_app.is_running()
        ):
            start_time = time.time()
            with torch.inference_mode():
                actions = inference_policy(obs)
                obs, _, dones, _ = env.step(actions)

            ended = dones.bool()
            if torch.any(ended):
                ended_ids = torch.where(ended)[0]
                for env_id in ended_ids.tolist():
                    object_idx = int(env_to_object_idx[env_id].item())
                    row = object_rows[object_idx]
                    if not hasattr(env.unwrapped, "_episode_success_before_reset"):
                        raise AttributeError("Environment does not have _episode_success_before_reset.")
                    episode_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                    if int(row["episodes"]) < args_cli.num_episodes:
                        row["episodes"] = int(row["episodes"]) + 1
                        row["episode_scales"].append(_scale_scalar(env_to_object_scale[env_id]))
                        if episode_success:
                            row["successes"] = int(row["successes"]) + 1

            elapsed = time.time() - start_time
            total_step_count += 1
            batch_step_count += 1
            finished = _count_finished_episodes(object_rows, args_cli.num_episodes)
            remaining = max(total_required - finished, 0)
            print(
                f"[PROGRESS][rank {rank}] batch={batch_index + 1}/{len(object_batches)} "
                f"batch_step={batch_step_count} total_step={total_step_count} "
                f"step_time={elapsed:.4f}s episodes_done={finished}/{total_required} "
                f"episodes_remaining={remaining}",
                flush=True,
            )

            if args_cli.real_time and dt is not None:
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()

    local_rows = _add_success_rates(object_rows, rank)
    rank_csv_path = os.path.join(log_dir, f"eval_objects_rank_{rank}.csv")
    _write_per_object_csv(rank_csv_path, local_rows)

    if args_cli.distributed:
        if not dist.is_initialized():
            raise RuntimeError("Distributed evaluation requires OnPolicyRunner to initialize torch.distributed.")
        gathered_rows = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_rows, object_gather_list=gathered_rows, dst=0)
        if rank == 0:
            all_rows = []
            for rank_rows in gathered_rows:
                all_rows.extend(rank_rows)
            _write_summary(log_dir, resume_path, merge_rows_by_object(all_rows))
    else:
        _write_summary(log_dir, resume_path, merge_rows_by_object(local_rows))


if __name__ == "__main__":
    main()
    if args_cli.distributed and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

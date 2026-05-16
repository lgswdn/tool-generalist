# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one selected tool across all object candidates and report per-object success rates."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

from utils.assets import load_selected_tool_ids
from utils.experiment.rl_runtime_spec import RUNTIME_SPEC_ENV_VAR, validate_runtime_spec


FFMPEG_PATH = "/usr/bin/ffmpeg"


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


def _load_json_list(path: str, label: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Expected {label} to contain a JSON list of strings: {path}")
    if len(data) == 0:
        raise ValueError(f"{label} is empty: {path}")
    return data


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
    if not isinstance(tools_cfg, dict):
        raise ValueError("paths_yaml must contain a tools mapping")
    if "robots_usd_dir" not in tools_cfg and tools_cfg.get("robots_usd_root"):
        tools_cfg["robots_usd_dir"] = tools_cfg["robots_usd_root"]
    if "tools_adjusted_json" not in tools_cfg and tools_cfg.get("tools_json"):
        tools_cfg["tools_adjusted_json"] = tools_cfg["tools_json"]


def _prepare_paths_yaml_for_rank(
    paths_yaml: str,
    selected_tool: str,
    rank: int,
    world_size: int,
) -> tuple[str, list[str], list[str]]:
    source_yaml = Path(paths_yaml).expanduser().resolve()
    with source_yaml.open("r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f)
    if not isinstance(paths_cfg, dict):
        raise ValueError(f"Expected paths yaml to contain a mapping: {source_yaml}")
    _ensure_legacy_aliases(paths_cfg)

    base_dir = source_yaml.parent
    tools_cfg = paths_cfg["tools"]
    selected_json = _resolve_path(str(tools_cfg["tools_selected_json"]), base_dir)
    selected_tool_names = load_selected_tool_ids(selected_json)
    if selected_tool not in selected_tool_names:
        raise ValueError(
            f"Tool {selected_tool!r} is not listed in effective tools_selected_json: {selected_json}"
        )

    candidates_json = _resolve_path(_get_candidates_json(paths_cfg), base_dir)
    all_candidates = _load_json_list(candidates_json, "candidates_json")
    rank_candidates = all_candidates[rank::world_size]
    if len(rank_candidates) == 0:
        raise ValueError(
            f"Rank {rank} received no objects from {len(all_candidates)} candidates "
            f"with world_size={world_size}."
        )

    temp_root = os.path.join(tempfile.gettempdir(), "tool_generalist_eval_single_tool")
    os.makedirs(temp_root, exist_ok=True)
    safe_tool = _safe_filename(selected_tool)
    rank_selected_json = os.path.join(temp_root, f"tools_selected_{safe_tool}_rank_{rank}_of_{world_size}.json")
    rank_candidates_json = os.path.join(temp_root, f"candidates_{safe_tool}_rank_{rank}_of_{world_size}.json")
    rank_paths_yaml = os.path.join(temp_root, f"paths_{safe_tool}_rank_{rank}_of_{world_size}.yaml")

    with open(rank_selected_json, "w", encoding="utf-8") as f:
        json.dump([selected_tool], f, ensure_ascii=False, indent=2)
    with open(rank_candidates_json, "w", encoding="utf-8") as f:
        json.dump(rank_candidates, f, ensure_ascii=False, indent=2)

    rank_paths_cfg = copy.deepcopy(paths_cfg)
    _ensure_legacy_aliases(rank_paths_cfg)
    rank_paths_cfg["tools"]["tools_selected_json"] = rank_selected_json
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


parser = argparse.ArgumentParser(description="Evaluate one selected tool across all object candidates.")
parser.add_argument(
    "--runtime_spec",
    type=str,
    required=True,
    help="Path to the rl_runtime_spec.json written with the checkpoint.",
)
parser.add_argument(
    "--paths_yaml",
    type=str,
    default=None,
    help=(
        "Optional paths.yaml override. When set, tool/object asset sources and object selection "
        "come from this paths.yaml instead of runtime_spec['paths_yaml']."
    ),
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RSL-RL checkpoint to evaluate.")
parser.add_argument("--tool", type=str, required=True, help="Tool name from the effective paths_yaml tools_selected.json.")
parser.add_argument("--task", type=str, default=None, help="Name of the task. Defaults to runtime_spec['task_id'].")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate per rank.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate per object.")
parser.add_argument("--max_episode_steps", type=int, default=300, help="Safety cap on episode length (steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--video_length", type=int, default=400, help="Maximum length of each recorded video in steps.")
parser.add_argument("--video_width", type=int, default=128, help="Per-env tiled-camera video width.")
parser.add_argument("--video_height", type=int, default=128, help="Per-env tiled-camera video height.")
parser.add_argument("--video_fps", type=int, default=30, help="Output video frames per second.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to write success/failure MP4 files.")
parser.add_argument("--success_videos", type=int, default=0, help="Number of successful episodes to record.")
parser.add_argument("--failure_videos", type=int, default=0, help="Number of failed episodes to record.")
parser.add_argument(
    "--video_max_active_episodes",
    type=int,
    default=4,
    help="Maximum number of episode videos recorded concurrently per rank.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run evaluation across multiple GPUs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.success_videos < 0:
    parser.error("--success_videos must be >= 0")
if args_cli.failure_videos < 0:
    parser.error("--failure_videos must be >= 0")
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if args_cli.video_width <= 0:
    parser.error("--video_width must be positive")
if args_cli.video_height <= 0:
    parser.error("--video_height must be positive")
if args_cli.video_fps <= 0:
    parser.error("--video_fps must be positive")
if args_cli.video_max_active_episodes <= 0:
    parser.error("--video_max_active_episodes must be positive")

runtime_spec_path = os.path.abspath(os.path.normpath(args_cli.runtime_spec))
with open(runtime_spec_path, "r", encoding="utf-8") as f:
    runtime_spec = json.load(f)

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
rank, world_size = _distributed_rank_info(args_cli.distributed)
local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args_cli.distributed else 0
os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = str(rank)
os.environ["TOOL_GENERALIST_LOCAL_RANK"] = str(local_rank)
os.environ["TOOL_GENERALIST_WORLD_SIZE"] = str(world_size)
rank_paths_yaml, all_candidate_entries, rank_candidate_entries = _prepare_paths_yaml_for_rank(
    paths_yaml,
    args_cli.tool,
    rank,
    world_size,
)
print(
    f"[INFO][rank {rank}]: Evaluating tool {args_cli.tool!r} on "
    f"{len(rank_candidate_entries)}/{len(all_candidate_entries)} object candidates for this rank.",
    flush=True,
)
if args_cli.paths_yaml is not None:
    print(
        f"[INFO][rank {rank}]: Overriding runtime spec paths_yaml: "
        f"{runtime_spec_paths_yaml} -> {paths_yaml}",
        flush=True,
    )
os.environ["TOOL_GENERALIST_PATHS_YAML"] = rank_paths_yaml

eval_runtime_spec = copy.deepcopy(runtime_spec)
eval_runtime_spec["num_envs"] = args_cli.num_envs
if isinstance(eval_runtime_spec.get("env_params"), dict):
    eval_runtime_spec["env_params"]["num_envs"] = args_cli.num_envs
eval_runtime_spec["paths_yaml"] = rank_paths_yaml
asset_assignment = eval_runtime_spec.setdefault("asset_assignment_params", {})
asset_assignment["randomize_tool_assignment"] = False
asset_assignment["randomize_object_assignment"] = False
_backfill_runtime_spec_defaults(eval_runtime_spec)
eval_runtime_spec_path = os.path.join(
    tempfile.gettempdir(),
    "tool_generalist_eval_single_tool",
    f"rl_runtime_spec_{_safe_filename(args_cli.tool)}_rank_{rank}_of_{world_size}_envs_{args_cli.num_envs}.json",
)
os.makedirs(os.path.dirname(eval_runtime_spec_path), exist_ok=True)
validate_runtime_spec(eval_runtime_spec, eval_runtime_spec_path)
with open(eval_runtime_spec_path, "w", encoding="utf-8") as f:
    json.dump(eval_runtime_spec, f, ensure_ascii=False, indent=2)
os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path

if args_cli.success_videos + args_cli.failure_videos > 0:
    args_cli.enable_cameras = True

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
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    OBJECT_ASSET_CFGS,
    TOOL_DATA,
    get_object_index_for_env,
)


def _tool_names_from_loaded_data() -> list[str]:
    tool_names = [tool_data["name"] for tool_data in TOOL_DATA]
    if len(tool_names) != 1:
        raise ValueError(f"Expected exactly one loaded tool, got {len(tool_names)}: {tool_names}")
    if tool_names[0] != args_cli.tool:
        raise ValueError(f"Loaded tool {tool_names[0]!r} does not match requested tool {args_cli.tool!r}")
    return tool_names


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
    return [{"name": name, "episodes": 0, "successes": 0} for name in object_names]


def _all_objects_finished(object_rows: list[dict], episodes_per_object: int) -> bool:
    for row in object_rows:
        if int(row["episodes"]) < episodes_per_object:
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
        rows_with_rates.append(
            {
                "name": row["name"],
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                "rank": row_rank,
            }
        )
    return rows_with_rates


def _write_results_csv(path: str, tool_name: str, rows: list[dict]) -> None:
    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    total_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "tool", "object", "episodes", "successes", "success_rate", "rank"])
        writer.writerow(["total", tool_name, "__all__", total_episodes, total_successes, total_rate, "all"])
        for row in sorted(rows, key=lambda item: item["name"]):
            writer.writerow(
                [
                    "object",
                    tool_name,
                    row["name"],
                    row["episodes"],
                    row["successes"],
                    row["success_rate"],
                    row["rank"],
                ]
            )


def _make_record_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/EvalRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0),
            rot=(-0.3337, 0.6234, 0.6234, -0.3337),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=1.8,
            horizontal_aperture=28.0,
            clipping_range=(0.05, 20.0),
        ),
        width=args_cli.video_width,
        height=args_cli.video_height,
    )


def _start_ffmpeg_writer(path: str) -> subprocess.Popen:
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args_cli.video_width}x{args_cli.video_height}",
        "-r",
        str(args_cli.video_fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _close_ffmpeg_writer(writer: subprocess.Popen) -> None:
    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def _init_video_state(video_dir: str) -> dict:
    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG_PATH}")
    os.makedirs(video_dir, exist_ok=True)
    return {
        "video_dir": video_dir,
        "active": {},
        "next_id": 0,
        "next_env_cursor": 0,
        "success_saved": 0,
        "failure_saved": 0,
        "kept_paths": [],
    }


def _video_quota_remaining(video_state: dict) -> bool:
    return (
        int(video_state["success_saved"]) < args_cli.success_videos
        or int(video_state["failure_saved"]) < args_cli.failure_videos
    )


def _eligible_video_envs(env_to_object_idx: torch.Tensor, object_rows: list[dict]) -> list[int]:
    env_ids = []
    for env_id, object_idx in enumerate(env_to_object_idx.tolist()):
        row = object_rows[int(object_idx)]
        if int(row["episodes"]) < args_cli.num_episodes:
            env_ids.append(env_id)
    return env_ids


def _start_episode_video(video_state: dict, env_id: int, object_name: str) -> None:
    record_id = int(video_state["next_id"])
    video_state["next_id"] = record_id + 1
    tmp_path = os.path.join(
        video_state["video_dir"],
        f"rank_{rank:03d}_pending_{record_id:06d}_{_safe_filename(object_name)}.tmp.mp4",
    )
    video_state["active"][int(env_id)] = {
        "record_id": record_id,
        "env_id": int(env_id),
        "object_name": object_name,
        "tmp_path": tmp_path,
        "writer": _start_ffmpeg_writer(tmp_path),
        "frames": 0,
    }


def _activate_video_slots(
    video_state: dict,
    env_to_object_idx: torch.Tensor,
    object_names: list[str],
    object_rows: list[dict],
) -> None:
    if not _video_quota_remaining(video_state):
        return
    env_ids = _eligible_video_envs(env_to_object_idx, object_rows)
    if len(env_ids) == 0:
        return
    cursor = int(video_state["next_env_cursor"])
    attempts = 0
    while (
        len(video_state["active"]) < args_cli.video_max_active_episodes
        and _video_quota_remaining(video_state)
        and attempts < len(env_ids)
    ):
        env_id = env_ids[cursor % len(env_ids)]
        cursor += 1
        attempts += 1
        if env_id in video_state["active"]:
            continue
        object_idx = int(env_to_object_idx[env_id].item())
        _start_episode_video(video_state, env_id, object_names[object_idx])
    video_state["next_env_cursor"] = cursor


def _capture_video_frames(env, video_state: dict) -> None:
    if len(video_state["active"]) == 0:
        return
    env.unwrapped.sim.render()
    env.unwrapped.scene["eval_record_camera"].update(dt=0.0, force_recompute=True)
    rgb_all = env.unwrapped.scene["eval_record_camera"].data.output["rgb"]
    for record in list(video_state["active"].values()):
        writer = record.get("writer")
        if writer is None:
            continue
        frame_tensor = rgb_all[int(record["env_id"]), ..., :3].detach().cpu()
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = torch.clamp(frame_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
        frame = frame_tensor.contiguous().numpy()
        writer.stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1
        if int(record["frames"]) >= args_cli.video_length:
            _close_ffmpeg_writer(writer)
            record["writer"] = None


def _discard_video_tmp(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _finish_video_on_done(video_state: dict, env_id: int, episode_success: bool) -> None:
    record = video_state["active"].pop(int(env_id), None)
    if record is None:
        return
    writer = record.get("writer")
    if writer is not None:
        _close_ffmpeg_writer(writer)
        record["writer"] = None

    label = "success" if episode_success else "failure"
    if episode_success:
        if int(video_state["success_saved"]) >= args_cli.success_videos:
            _discard_video_tmp(record["tmp_path"])
            return
        slot = int(video_state["success_saved"])
        video_state["success_saved"] = slot + 1
    else:
        if int(video_state["failure_saved"]) >= args_cli.failure_videos:
            _discard_video_tmp(record["tmp_path"])
            return
        slot = int(video_state["failure_saved"])
        video_state["failure_saved"] = slot + 1

    final_path = os.path.join(
        video_state["video_dir"],
        f"rank_{rank:03d}_{label}_{slot:03d}_{_safe_filename(record['object_name'])}.mp4",
    )
    os.replace(record["tmp_path"], final_path)
    video_state["kept_paths"].append(final_path)


def _discard_video_on_done(video_state: dict, env_id: int) -> None:
    record = video_state["active"].pop(int(env_id), None)
    if record is None:
        return
    writer = record.get("writer")
    if writer is not None:
        _close_ffmpeg_writer(writer)
    _discard_video_tmp(record["tmp_path"])


def _close_video_state(video_state: dict) -> None:
    for record in list(video_state["active"].values()):
        writer = record.get("writer")
        if writer is not None:
            _close_ffmpeg_writer(writer)
        _discard_video_tmp(record["tmp_path"])
    video_state["active"].clear()


def _write_summary(log_dir: str, resume_path: str, tool_name: str, rows: list[dict]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    summary_path = os.path.join(log_dir, f"eval_single_tool_{_safe_filename(tool_name)}_summary.json")
    csv_path = os.path.join(log_dir, f"eval_single_tool_{_safe_filename(tool_name)}.csv")
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "runtime_spec_paths_yaml": runtime_spec_paths_yaml,
        "effective_source_paths_yaml": paths_yaml,
        "paths_yaml_override": args_cli.paths_yaml,
        "rank_paths_yaml": rank_paths_yaml,
        "tool": tool_name,
        "world_size": world_size,
        "num_envs_per_rank": args_cli.num_envs,
        "episodes_per_object": args_cli.num_episodes,
        "source_candidate_count": len(all_candidate_entries),
        "objects": len(rows),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": success_rate,
        "per_object": sorted(rows, key=lambda item: item["name"]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_results_csv(csv_path, tool_name, rows)

    print("\n========== Single-Tool Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"Tool: {tool_name}")
    print(f"World Size: {world_size}")
    print(f"Objects: {len(rows)}")
    print(f"Episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100.0:.2f}%")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    print("====================================================\n")


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

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    video_dir = (
        args_cli.video_dir
        if args_cli.video_dir is not None
        else os.path.join(log_dir, "videos", "eval_single_tool", _safe_filename(args_cli.tool))
    )

    want_videos = args_cli.success_videos + args_cli.failure_videos > 0
    if want_videos:
        env_cfg.scene.eval_record_camera = _make_record_camera_cfg()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO][rank {rank}]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)

    if not hasattr(env.unwrapped, "episode_success_buf"):
        raise AttributeError("Environment does not have episode_success_buf.")

    tool_name = _tool_names_from_loaded_data()[0]
    object_names = _object_names_from_loaded_data()
    object_rows = _init_object_rows(object_names)
    num_envs = env.unwrapped.num_envs
    env_to_object_idx = torch.tensor(
        [get_object_index_for_env(env_id) for env_id in range(num_envs)],
        dtype=torch.long,
    )
    env_id_by_object: dict[int, int] = {}
    for env_id, object_idx in enumerate(env_to_object_idx.tolist()):
        env_id_by_object.setdefault(int(object_idx), env_id)

    missing_object_indices = sorted(set(range(len(object_names))).difference(env_id_by_object))
    if missing_object_indices:
        missing_names = [object_names[index] for index in missing_object_indices]
        raise ValueError(
            f"Rank {rank} has {len(object_names)} objects but only {num_envs} envs, so "
            f"{len(missing_names)} objects have no assigned env. Increase --num_envs to at least "
            f"{len(object_names)} per rank or use more distributed ranks. Missing objects: {missing_names}"
        )

    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    total_required = len(object_names) * args_cli.num_episodes
    step_count = 0

    video_state = None
    if want_videos:
        video_state = _init_video_state(video_dir)
        _activate_video_slots(video_state, env_to_object_idx, object_names, object_rows)
        print(f"[INFO][rank {rank}]: Recording success/failure videos to: {video_dir}")

    while (not _all_objects_finished(object_rows, args_cli.num_episodes)) and simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = inference_policy(obs)
            obs, _, dones, _ = env.step(actions)

        if video_state is not None:
            _capture_video_frames(env, video_state)

        ended = dones.bool()
        if torch.any(ended):
            ended_ids = torch.where(ended)[0]
            for env_id in ended_ids.tolist():
                object_idx = int(env_to_object_idx[env_id].item())
                row = object_rows[object_idx]
                if not hasattr(env.unwrapped, "_episode_success_before_reset"):
                    raise AttributeError("Environment does not have _episode_success_before_reset.")
                episode_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                counted_episode = int(row["episodes"]) < args_cli.num_episodes
                if counted_episode:
                    row["episodes"] = int(row["episodes"]) + 1
                    if episode_success:
                        row["successes"] = int(row["successes"]) + 1
                if video_state is not None:
                    if counted_episode:
                        _finish_video_on_done(video_state, env_id, episode_success)
                    else:
                        _discard_video_on_done(video_state, env_id)

            if video_state is not None:
                _activate_video_slots(video_state, env_to_object_idx, object_names, object_rows)

        elapsed = time.time() - start_time
        step_count += 1
        finished = _count_finished_episodes(object_rows, args_cli.num_episodes)
        remaining = max(total_required - finished, 0)
        print(
            f"[PROGRESS][rank {rank}] step={step_count} "
            f"step_time={elapsed:.4f}s episodes_done={finished}/{total_required} "
            f"episodes_remaining={remaining}",
            flush=True,
        )

        if args_cli.real_time and dt is not None:
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if video_state is not None:
        _close_video_state(video_state)
        if int(video_state["success_saved"]) < args_cli.success_videos:
            print(
                f"[WARNING][rank {rank}]: Requested {args_cli.success_videos} success videos, "
                f"saved {video_state['success_saved']} during evaluated episodes."
            )
        if int(video_state["failure_saved"]) < args_cli.failure_videos:
            print(
                f"[WARNING][rank {rank}]: Requested {args_cli.failure_videos} failure videos, "
                f"saved {video_state['failure_saved']} during evaluated episodes."
            )

    env.close()

    local_rows = _add_success_rates(object_rows, rank)

    rank_csv_path = os.path.join(log_dir, f"eval_single_tool_{_safe_filename(tool_name)}_rank_{rank}.csv")
    _write_results_csv(rank_csv_path, tool_name, local_rows)

    if args_cli.distributed:
        if not dist.is_initialized():
            raise RuntimeError("Distributed evaluation requires OnPolicyRunner to initialize torch.distributed.")
        gathered_rows = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_rows, object_gather_list=gathered_rows, dst=0)
        if rank == 0:
            all_rows = []
            for rank_rows in gathered_rows:
                all_rows.extend(rank_rows)
            _write_summary(log_dir, resume_path, tool_name, all_rows)
    else:
        _write_summary(log_dir, resume_path, tool_name, local_rows)


if __name__ == "__main__":
    main()
    if args_cli.distributed and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

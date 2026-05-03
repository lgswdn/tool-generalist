# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a multi-tool RSL-RL checkpoint and report per-tool success rates."""

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

import yaml
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


DEFAULT_TASK = "tool-sdf-multitool-patch-v0"
DEFAULT_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/RL/multitool_sdf_patch/"
    "2026-04-29_15-14-36/model_1000.pt"
)
DEFAULT_PATHS_YAML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "paths_multitool.yaml")
)
FFMPEG_PATH = "/usr/bin/ffmpeg"


def _distributed_rank_info(distributed: bool) -> tuple[int, int]:
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size
    return 0, 1


def _prepare_paths_yaml_for_rank(paths_yaml: str, rank: int, world_size: int) -> str:
    with open(paths_yaml, "r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f)

    tools_cfg = paths_cfg["tools"]
    selected_json = tools_cfg["tools_selected_json"]
    with open(selected_json, "r", encoding="utf-8") as f:
        selected_tool_names = json.load(f)

    if not isinstance(selected_tool_names, list):
        raise ValueError(f"Expected {selected_json} to contain a JSON list.")
    if len(selected_tool_names) == 0:
        raise ValueError(f"Tool selection file is empty: {selected_json}")

    rank_tool_names = selected_tool_names[rank::world_size]
    if len(rank_tool_names) == 0:
        raise ValueError(
            f"Rank {rank} received no tools from {len(selected_tool_names)} selected tools "
            f"with world_size={world_size}."
        )

    if world_size == 1:
        return paths_yaml

    temp_root = os.path.join(tempfile.gettempdir(), "tool_generalist_eval_tools")
    os.makedirs(temp_root, exist_ok=True)
    rank_selected_json = os.path.join(temp_root, f"tools_selected_rank_{rank}_of_{world_size}.json")
    rank_paths_yaml = os.path.join(temp_root, f"paths_rank_{rank}_of_{world_size}.yaml")

    with open(rank_selected_json, "w", encoding="utf-8") as f:
        json.dump(rank_tool_names, f, ensure_ascii=False, indent=2)

    rank_paths_cfg = copy.deepcopy(paths_cfg)
    rank_paths_cfg["tools"]["tools_selected_json"] = rank_selected_json
    with open(rank_paths_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(rank_paths_cfg, f, sort_keys=False)

    return rank_paths_yaml


parser = argparse.ArgumentParser(description="Evaluate all tools in tools_selected.json with an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate per rank.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate per tool.")
parser.add_argument("--max_episode_steps", type=int, default=300, help="Safety cap on episode length (steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=1_000_000, help="Interval between videos.")
parser.add_argument("--video_width", type=int, default=128, help="Per-env tiled-camera video width.")
parser.add_argument("--video_height", type=int, default=128, help="Per-env tiled-camera video height.")
parser.add_argument("--video_fps", type=int, default=30, help="Output video frames per second.")
parser.add_argument(
    "--video_max_active_tools",
    type=int,
    default=16,
    help="Maximum number of tools encoded concurrently per rank.",
)
parser.add_argument("--video_dir", type=str, default=None, help="Directory to write per-tool MP4 files.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run evaluation across multiple GPUs.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    args_cli.checkpoint = DEFAULT_CHECKPOINT

paths_yaml = os.environ["TOOL_GENERALIST_PATHS_YAML"] if "TOOL_GENERALIST_PATHS_YAML" in os.environ else DEFAULT_PATHS_YAML
paths_yaml = os.path.abspath(os.path.normpath(paths_yaml))
rank, world_size = _distributed_rank_info(args_cli.distributed)
rank_paths_yaml = _prepare_paths_yaml_for_rank(paths_yaml, rank, world_size)
os.environ["TOOL_GENERALIST_PATHS_YAML"] = rank_paths_yaml

if args_cli.video:
    args_cli.enable_cameras = True
    if args_cli.video_width <= 0:
        raise ValueError("--video_width must be positive.")
    if args_cli.video_height <= 0:
        raise ValueError("--video_height must be positive.")
    if args_cli.video_fps <= 0:
        raise ValueError("--video_fps must be positive.")
    if args_cli.video_length <= 0:
        raise ValueError("--video_length must be positive.")
    if args_cli.video_max_active_tools <= 0:
        raise ValueError("--video_max_active_tools must be positive.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.distributed as dist
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import TOOL_DATA


def _tool_names_from_loaded_data() -> list[str]:
    tool_names = [tool_data["name"] for tool_data in TOOL_DATA]
    if len(tool_names) == 0:
        raise ValueError("No tools were loaded into TOOL_DATA.")
    return tool_names


def _init_tool_rows(tool_names: list[str]) -> list[dict]:
    return [{"name": name, "episodes": 0, "successes": 0} for name in tool_names]


def _all_tools_finished(tool_rows: list[dict], episodes_per_tool: int) -> bool:
    for row in tool_rows:
        if int(row["episodes"]) < episodes_per_tool:
            return False
    return True


def _count_finished_episodes(tool_rows: list[dict], episodes_per_tool: int) -> int:
    count = 0
    for row in tool_rows:
        episodes = int(row["episodes"])
        count += min(episodes, episodes_per_tool)
    return count


def _add_success_rates(tool_rows: list[dict], row_rank: int) -> list[dict]:
    rows_with_rates = []
    for row in tool_rows:
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


def _write_per_tool_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "episodes", "successes", "success_rate", "rank"])
        for row in sorted(rows, key=lambda item: item["name"]):
            writer.writerow([row["name"], row["episodes"], row["successes"], row["success_rate"], row["rank"]])


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) == 0:
        raise ValueError(f"Could not create a safe filename from tool name: {name}")
    return safe


def _make_record_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/EvalRecordCamera",
        offset=TiledCameraCfg.OffsetCfg(
            # Diagonal overhead view aimed at the manipulation workspace near x=0.45, y=0.0, z=0.35.
            pos=(1.55, 0.9, 1.35),
            rot=(-0.1973, 0.3801, 0.8020, -0.4164),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
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


def _init_video_state(tool_names: list[str], video_dir: str) -> tuple[list[dict], list[int], list[int], set[int]]:
    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG_PATH}")
    os.makedirs(video_dir, exist_ok=True)
    video_records = []
    for tool_idx, tool_name in enumerate(tool_names):
        video_records.append(
            {
                "tool_idx": tool_idx,
                "tool_name": tool_name,
                "env_id": tool_idx,
                "state": "pending",
                "frames": 0,
                "path": os.path.join(video_dir, f"rank_{rank:03d}_{_safe_filename(tool_name)}.mp4"),
                "writer": None,
            }
        )
    pending = list(range(len(video_records)))
    waiting = []
    recording = set()
    return video_records, pending, waiting, recording


def _activate_video_slots(
    video_records: list[dict],
    pending: list[int],
    waiting: list[int],
    recording: set[int],
    *,
    at_episode_start: bool,
) -> None:
    while len(recording) + len(waiting) < args_cli.video_max_active_tools and len(pending) > 0:
        record_idx = pending.pop(0)
        record = video_records[record_idx]
        if at_episode_start:
            record["writer"] = _start_ffmpeg_writer(record["path"])
            record["state"] = "recording"
            recording.add(record_idx)
        else:
            record["state"] = "waiting_reset"
            waiting.append(record_idx)


def _capture_video_frames(env, video_records: list[dict], recording: set[int]) -> None:
    if len(recording) == 0:
        return
    env.unwrapped.sim.render()
    env.unwrapped.scene["eval_record_camera"].update(dt=0.0, force_recompute=True)
    rgb_all = env.unwrapped.scene["eval_record_camera"].data.output["rgb"]
    for record_idx in list(recording):
        record = video_records[record_idx]
        frame_tensor = rgb_all[int(record["env_id"]), ..., :3].detach().cpu()
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = torch.clamp(frame_tensor * 255.0, 0.0, 255.0).to(torch.uint8)
        frame = frame_tensor.contiguous().numpy()
        record["writer"].stdin.write(frame.tobytes())
        record["frames"] = int(record["frames"]) + 1


def _finish_recording(record: dict, recording: set[int]) -> None:
    _close_ffmpeg_writer(record["writer"])
    record["writer"] = None
    record["state"] = "done"
    recording.remove(int(record["tool_idx"]))


def _update_video_records_on_done(
    ended_env_ids: set[int],
    video_records: list[dict],
    pending: list[int],
    waiting: list[int],
    recording: set[int],
) -> None:
    for record_idx in list(recording):
        record = video_records[record_idx]
        if int(record["frames"]) >= args_cli.video_length or int(record["env_id"]) in ended_env_ids:
            _finish_recording(record, recording)

    for record_idx in list(waiting):
        record = video_records[record_idx]
        if int(record["env_id"]) in ended_env_ids:
            record["writer"] = _start_ffmpeg_writer(record["path"])
            record["state"] = "recording"
            waiting.remove(record_idx)
            recording.add(record_idx)

    _activate_video_slots(video_records, pending, waiting, recording, at_episode_start=False)


def _videos_finished(video_records: list[dict]) -> bool:
    for record in video_records:
        if record["state"] != "done":
            return False
    return True


def _write_summary(log_dir: str, resume_path: str, rows: list[dict]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    summary_path = os.path.join(log_dir, "eval_tools_summary.json")
    per_tool_path = os.path.join(log_dir, "eval_tools_per_tool.csv")
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "source_paths_yaml": paths_yaml,
        "world_size": world_size,
        "num_envs_per_rank": args_cli.num_envs,
        "episodes_per_tool": args_cli.num_episodes,
        "tools": len(rows),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": success_rate,
        "per_tool": sorted(rows, key=lambda item: item["name"]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_per_tool_csv(per_tool_path, rows)

    print("\n========== Multi-Tool Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"World Size: {world_size}")
    print(f"Tools: {len(rows)}")
    print(f"Episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100.0:.2f}%")
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_tool_path}")
    print("==================================================\n")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
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
        else os.path.join(log_dir, "videos", "eval_tools_tiled")
    )

    if args_cli.video:
        env_cfg.scene.eval_record_camera = _make_record_camera_cfg()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO][rank {rank}]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy_obj = ppo_runner.alg.policy

    if not hasattr(env.unwrapped, "episode_success_buf"):
        raise AttributeError("Environment does not have episode_success_buf.")

    tool_names = _tool_names_from_loaded_data()
    tool_rows = _init_tool_rows(tool_names)
    num_tools = len(tool_names)
    num_envs = env.unwrapped.num_envs
    if args_cli.video and num_envs < num_tools:
        raise ValueError(
            f"Per-tool video recording requires num_envs >= tools on this rank, got {num_envs} envs for {num_tools} tools."
        )
    env_to_tool_idx = torch.arange(num_envs, dtype=torch.long) % num_tools

    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    total_required = len(tool_rows) * args_cli.num_episodes
    pbar = tqdm(total=total_required, desc=f"Rank {rank} evaluating tools", unit="episodes", disable=rank != 0)

    video_records = []
    video_pending = []
    video_waiting = []
    video_recording = set()
    if args_cli.video:
        video_records, video_pending, video_waiting, video_recording = _init_video_state(tool_names, video_dir)
        _activate_video_slots(video_records, video_pending, video_waiting, video_recording, at_episode_start=True)
        print(f"[INFO][rank {rank}]: Recording per-tool videos to: {video_dir}")

    while (
        not _all_tools_finished(tool_rows, args_cli.num_episodes)
        or (args_cli.video and not _videos_finished(video_records))
    ) and simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy_obj.act(obs)
            obs, _, dones, _ = env.step(actions)

        if args_cli.video:
            _capture_video_frames(env, video_records, video_recording)

        ended = dones.bool()
        if torch.any(ended):
            ended_ids = torch.where(ended)[0]
            ended_env_ids = set(ended_ids.tolist())
            progress_before = _count_finished_episodes(tool_rows, args_cli.num_episodes)
            for env_id in ended_ids.tolist():
                tool_idx = int(env_to_tool_idx[env_id].item())
                row = tool_rows[tool_idx]
                if int(row["episodes"]) < args_cli.num_episodes:
                    if not hasattr(env.unwrapped, "_episode_success_before_reset"):
                        raise AttributeError("Environment does not have _episode_success_before_reset.")
                    row["episodes"] = int(row["episodes"]) + 1
                    episode_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                    if episode_success:
                        row["successes"] = int(row["successes"]) + 1

            progress_after = _count_finished_episodes(tool_rows, args_cli.num_episodes)
            pbar.update(progress_after - progress_before)
            finished = progress_after
            successes = sum(int(row["successes"]) for row in tool_rows)
            success_rate = float(successes) / float(finished) if finished > 0 else 0.0
            pbar.set_postfix({"Success Rate": f"{success_rate * 100.0:.2f}%", "Episodes": finished})

            if args_cli.video:
                _update_video_records_on_done(
                    ended_env_ids,
                    video_records,
                    video_pending,
                    video_waiting,
                    video_recording,
                )

        elif args_cli.video:
            _update_video_records_on_done(
                set(),
                video_records,
                video_pending,
                video_waiting,
                video_recording,
            )

        if args_cli.real_time and dt is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if args_cli.video:
        for record_idx in list(video_recording):
            _finish_recording(video_records[record_idx], video_recording)

    pbar.close()
    env.close()

    local_rows = _add_success_rates(tool_rows, rank)

    rank_csv_path = os.path.join(log_dir, f"eval_tools_rank_{rank}.csv")
    _write_per_tool_csv(rank_csv_path, local_rows)

    if args_cli.distributed:
        if not dist.is_initialized():
            raise RuntimeError("Distributed evaluation requires OnPolicyRunner to initialize torch.distributed.")
        gathered_rows = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_rows, object_gather_list=gathered_rows, dst=0)
        if rank == 0:
            all_rows = []
            for rank_rows in gathered_rows:
                all_rows.extend(rank_rows)
            _write_summary(log_dir, resume_path, all_rows)
    else:
        _write_summary(log_dir, resume_path, local_rows)


if __name__ == "__main__":
    main()
    if args_cli.distributed and dist.is_initialized():
        dist.destroy_process_group()
    simulation_app.close()

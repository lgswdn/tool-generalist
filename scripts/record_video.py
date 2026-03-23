#!/usr/bin/env python3
"""Minimal script to record video without policy - uses random actions."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video with zero actions.")
parser.add_argument("--task", type=str, default=None, help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--video_length", type=int, default=300, help="Video length in steps")
parser.add_argument("--video_dir", type=str, default="videos", help="Video output directory")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime
from isaaclab.envs import ManagerBasedRLEnvCfg
import IsaacLab_nonPrehensile.tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_kwargs = {
        "video_folder": args_cli.video_dir,
        "name_prefix": f"video_{timestamp}",
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO] Recording {args_cli.video_length} steps")
    obs, _ = env.reset()
    print("[DEBUG] Reset complete")
    for step in range(args_cli.video_length):
        print(f"[DEBUG] Starting step {step}")
        actions = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
        print(f"[DEBUG] About to call env.step() for step {step}")
        obs, _, terminated, truncated, _ = env.step(actions)
        print(f"[DEBUG] env.step() completed for step {step}")
        if terminated.any() or truncated.any():
            obs, _ = env.reset()
        if step % 50 == 0:
            print(f"[INFO] Step {step}/{args_cli.video_length}")

    print("[INFO] Video complete")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

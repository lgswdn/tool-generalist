#!/usr/bin/env python3
"""Minimal test script without video recording."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
import IsaacLab_nonPrehensile.tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    print("[INFO] Testing 30 steps WITH rendering (no video wrapper)")
    obs, _ = env.reset()
    print("[DEBUG] Reset complete")

    for step in range(30):
        print(f"[DEBUG] Step {step}")
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    print("[INFO] Test complete")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

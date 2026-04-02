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

    # Import debug helpers
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
        get_head_area_pos_w,
        get_tool_pointcloud_in_ee_frame,
    )
    from isaaclab.utils.math import matrix_from_quat

    for step in range(args_cli.video_length):
        print(f"[DEBUG] Starting step {step}")
        actions = torch.tensor(env.action_space.sample(), device=env.unwrapped.device)
        print(f"[DEBUG] About to call env.step() for step {step}")
        obs, _, terminated, truncated, _ = env.step(actions)
        print(f"[DEBUG] env.step() completed for step {step}")

        # --- Debug: EE-to-object distance ---
        base_env = env.unwrapped
        ee_pos_w = get_head_area_pos_w(base_env)  # (num_envs, 3)
        obj_pos_w = base_env.scene["object"].data.root_pos_w  # (num_envs, 3)
        ee_obj_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)  # (num_envs,)
        print(f"[DEBUG] Step {step} | EE-obj distance: {ee_obj_dist.cpu().numpy()}")

        # --- Debug: Visualize tool point cloud in world frame ---
        _visualize_tool_pointcloud_world(base_env)

        if terminated.any() or truncated.any():
            obs, _ = env.reset()
        if step % 50 == 0:
            print(f"[INFO] Step {step}/{args_cli.video_length}")

    print("[INFO] Video complete")
    env.close()
    simulation_app.close()


def _visualize_tool_pointcloud_world(env):
    """Transform tool pointcloud from OBJ-local to world frame and visualize as blue spheres.

    Uses the robot articulation body state for the tool body, which gives the
    exact PhysX body pose including center-of-mass offset. The OBJ points are
    scaled and then transformed by the body's world pose.
    """
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
        get_cached_cloud, TOOL_OBJ_PATH, TOOL_SCALE,
    )
    from isaaclab.utils.math import matrix_from_quat
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    robot = env.scene["robot"]

    # Resolve tool body index once
    if not hasattr(env, "_viz_tool_body_idx"):
        from isaaclab.managers import SceneEntityCfg
        cfg = SceneEntityCfg("robot", body_names=["link_coacd_convex_piece_0"])
        cfg.resolve(env.scene)
        env._viz_tool_body_idx = cfg.body_ids[0]
    tool_idx = env._viz_tool_body_idx

    # Get tool body world pose from articulation body state
    # body_state_w: (num_envs, num_bodies, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    body_pos_w = robot.data.body_pos_w[:, tool_idx, :]   # (N, 3)
    body_quat_w = robot.data.body_quat_w[:, tool_idx, :] # (N, 4)

    # Load canonical OBJ points (unit scale)
    tool_cloud = get_cached_cloud(TOOL_OBJ_PATH)
    device = env.device
    base_pts = tool_cloud._get_points_torch(device).float()  # (M, 3)
    pts_scaled = base_pts * TOOL_SCALE  # (M, 3)

    # For VISUALIZATION only: filter to the prong/tine portion (local Z <= 0).
    # The OBJ mesh origin is the weld attachment point. Local Z > 0 is the handle
    # flange that sits inside link7 and is not visually rendered. Showing only Z <= 0
    # gives a cleaner view of the visible part of the fork.
    prong_mask = pts_scaled[:, 2] <= 0.0  # (M,) bool
    pts_viz = pts_scaled[prong_mask] if prong_mask.any() else pts_scaled  # (M', 3)

    num_envs = robot.data.body_pos_w.shape[0]
    num_points_viz = pts_viz.shape[0]

    # Expand to all envs: (N, M', 3)
    pts_local = pts_viz.unsqueeze(0).expand(num_envs, -1, -1)

    # Transform: world_pts = R @ local_pts^T + pos
    R = matrix_from_quat(body_quat_w)  # (N, 3, 3)
    pts_world = torch.bmm(R, pts_local.transpose(1, 2)).transpose(1, 2) + body_pos_w.unsqueeze(1)

    # Debug print (first step only)
    if not hasattr(env, "_tool_pc_debug_printed"):
        env._tool_pc_debug_printed = True
        pc_centroid = pts_world[0].mean(dim=0)
        print(f"[DEBUG] Tool body pos (world): {body_pos_w[0].cpu().numpy()}")
        print(f"[DEBUG] Tool body quat (world): {body_quat_w[0].cpu().numpy()}")
        print(f"[DEBUG] PC centroid (world):     {pc_centroid.cpu().numpy()}")
        print(f"[DEBUG] PC min (world):          {pts_world[0].min(dim=0).values.cpu().numpy()}")
        print(f"[DEBUG] PC max (world):          {pts_world[0].max(dim=0).values.cpu().numpy()}")
        ee_frame = env.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[..., 0, :][0]
        print(f"[DEBUG] EE frame pos (world):    {ee_pos.cpu().numpy()}")
        # Print link7 pos to understand the weld attachment point
        robot = env.scene["robot"]
        link7_names = [n for n in robot.data.body_names if "link7" in n]
        if link7_names:
            link7_idx = list(robot.data.body_names).index(link7_names[0])
            link7_pos = robot.data.body_pos_w[0, link7_idx]
            print(f"[DEBUG] panda_link7 pos (world): {link7_pos.cpu().numpy()}")
            print(f"[DEBUG] Tool-link7 delta (world): {(body_pos_w[0] - link7_pos).cpu().numpy()}")

    # Create/reuse visualizer
    if not hasattr(env, "_tool_pc_visualizer"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ToolPointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
                ),
            },
        )
        env._tool_pc_visualizer = VisualizationMarkers(marker_cfg)

    # Show first env's points
    first_env_pts = pts_world[0]  # (M, 3)
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_points_viz, -1)
    env._tool_pc_visualizer.visualize(translations=first_env_pts, orientations=orientations)


if __name__ == "__main__":
    main()

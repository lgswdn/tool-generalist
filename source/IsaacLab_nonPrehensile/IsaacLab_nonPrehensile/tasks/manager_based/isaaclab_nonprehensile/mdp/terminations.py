# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from .rewards_bimanual import bimanual_link_min_distance
from .step_cache import object_pose_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.01,  # Position threshold in meters
    rotation_threshold: float = 0.1,  # Rotation threshold in radians
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar: bool = False,  # If True, only consider x,y coordinates for position distance
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position and orientation.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object position to reach the goal. Defaults to 0.01.
        rotation_threshold: The threshold for the object orientation in radians. Defaults to 0.1.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        planar: If True, only consider x,y coordinates for position distance (ignore z). Defaults to False.

    """
    object: RigidObject = env.scene[object_cfg.name]
    return object_pose_success_mask(
        env,
        object,
        command_name=command_name,
        threshold=threshold,
        rotation_threshold=rotation_threshold,
        planar=planar,
    )


def object_reached_goal_dwell(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.05,
    rotation_threshold: float = 0.2,
    dwell_steps: int = 10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object stays inside the target-pose window for N steps."""

    object: RigidObject = env.scene[object_cfg.name]
    success_now = object_pose_success_mask(
        env,
        object,
        command_name=command_name,
        threshold=threshold,
        rotation_threshold=rotation_threshold,
    )
    if dwell_steps <= 1:
        return success_now

    counter = getattr(env, "_goal_pose_success_count", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    counter = torch.where(success_now, counter + 1, torch.zeros_like(counter))
    env._goal_pose_success_count = counter
    return counter >= int(dwell_steps)


def object_dropped_off_table(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.02,  # Minimum height above table surface
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object dropping off the table.

    Args:
        env: The environment.
        minimum_height: The minimum height above the table surface. Defaults to 0.02m.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate due to object falling.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # get current object position in world coordinates
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_height = object_pos_w[:, 2]  # z-coordinate
    # terminate if object is below minimum height (fell off table)
    return object_height < minimum_height


def bimanual_links_too_close(
    env: ManagerBasedRLEnv,
    threshold: float = 0.15,
    robot1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Terminate when selected links on the two bimanual arms violate the hard distance threshold."""

    return bimanual_link_min_distance(env, robot1_cfg=robot1_cfg, robot2_cfg=robot2_cfg) < float(threshold)

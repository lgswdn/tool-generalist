# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots with welded tool end-effectors.

The following configurations are available:

* :obj:`FRANKA_PANDA_TOOL_CFG`: Franka Emika Panda robot with a single tool USD
* :obj:`FRANKA_PANDA_TOOL_HIGH_PD_CFG`: Same with stiffer PD control

Helper functions:
* :func:`collect_robot_usd_paths`: Collect all tool-robot USD files from a directory
* :func:`build_multi_tool_robot_cfg`: Create a MultiUsdFileCfg robot config cycling through USDs

Reference: https://github.com/frankaemika/franka_ros
"""

import glob
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Paths
##

# Directory containing all panda_instanceable_<tool>.usd files
FRANKA_TOOL_USD_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..',
    'RobotSmith', 'eef', 'robots_usd',
))

# Default single-tool USD (first fork variant, used as fallback)
FRANKA_TOOL_USD_PATH = os.path.join(
    FRANKA_TOOL_USD_DIR,
    'panda_instanceable_000_robotic_fork_effector_var_000.usd',
)

##
# Single-tool base configuration
##

FRANKA_PANDA_TOOL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_TOOL_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot with a single tool."""


FRANKA_PANDA_TOOL_HIGH_PD_CFG = FRANKA_PANDA_TOOL_CFG.copy()
FRANKA_PANDA_TOOL_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_TOOL_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_TOOL_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_TOOL_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_TOOL_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

##
# Multi-tool helpers
##


def collect_robot_usd_paths(usd_dir: str | None = None) -> list[str]:
    """Collect and sort all tool-robot USD files in a directory.

    Excludes the bare ``panda_instanceable.usd`` (robot without a tool).

    Args:
        usd_dir: Directory to scan.  Defaults to ``FRANKA_TOOL_USD_DIR``.

    Returns:
        Sorted list of absolute USD file paths.
    """
    if usd_dir is None:
        usd_dir = FRANKA_TOOL_USD_DIR
    usd_dir = os.path.abspath(usd_dir)
    if not os.path.isdir(usd_dir):
        raise FileNotFoundError(f"Robot USD directory does not exist: {usd_dir}")

    usd_paths = sorted(
        p
        for p in glob.glob(os.path.join(usd_dir, "*.usd"))
        if os.path.isfile(p) and os.path.basename(p) != "panda_instanceable.usd"
    )
    if len(usd_paths) == 0:
        raise FileNotFoundError(f"No tool-robot USD files found in: {usd_dir}")
    return usd_paths


def build_multi_tool_robot_cfg(
    usd_paths: list[str] | None = None,
    *,
    random_choice: bool = False,
) -> ArticulationCfg:
    """Create a robot config that cycles through multiple tool USD files across envs.

    Args:
        usd_paths: Explicit list of USD paths.  If ``None``, all USDs in
            ``FRANKA_TOOL_USD_DIR`` are collected automatically.
        random_choice: If ``True``, tool assignment is random; otherwise
            deterministic (``env_id % num_tools``).

    Returns:
        An ``ArticulationCfg`` whose spawn uses ``MultiUsdFileCfg``.
    """
    if usd_paths is None:
        usd_paths = collect_robot_usd_paths()

    robot_cfg = FRANKA_PANDA_TOOL_HIGH_PD_CFG.copy()
    base_spawn = robot_cfg.spawn

    robot_cfg.spawn = sim_utils.MultiUsdFileCfg(
        usd_path=usd_paths,
        random_choice=random_choice,
        activate_contact_sensors=base_spawn.activate_contact_sensors,
        rigid_props=base_spawn.rigid_props,
        articulation_props=base_spawn.articulation_props,
        collision_props=base_spawn.collision_props,
        mass_props=base_spawn.mass_props,
        visual_material=base_spawn.visual_material,
        semantic_tags=base_spawn.semantic_tags,
    )
    return robot_cfg

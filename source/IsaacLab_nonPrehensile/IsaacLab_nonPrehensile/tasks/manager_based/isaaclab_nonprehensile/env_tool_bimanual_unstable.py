# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual Franka-tool environment with unstable/random object-pose goals."""

from __future__ import annotations

from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.robots.franka import build_multi_tool_robot_cfg
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.asset_assignment import (
    TOOL_ASSIGNMENT_SALT,
    asset_indices_for_rank,
    sequential_spawn_indices_for_rank,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    JOINT_BOX_MAX,
    JOINT_BOX_MIN,
    OBJECT_ASSET_CFGS_FOR_SPAWN,
    TOOL_ASSET_INDICES_BY_ENV,
    TOOL_DATA,
    TOOL_SCALE,
    TOOL_USD_PATHS,
    TOOL_USD_PATHS_FOR_SPAWN,
    _ASSET_ASSIGNMENT_SEED,
    _GLOBAL_RANK,
    _LOCAL_RANK,
    _NUM_ENVS_PER_RANK,
    _OBSERVATION_DIM,
    _PHYSICS_DIM,
    _PHYSICS_OBSERVATION_FIELDS,
    _RANDOMIZE_TOOL_ASSIGNMENT,
    _RL_CONTRACT,
    _ROBOT_MODE,
    _USE_BARE_FRANKA,
    _WORLD_SIZE,
    _dr_event_enabled,
    _table_bounds_xy,
    _table_top_z,
    _tool_dr_event_enabled,
    _unsupported_event,
    custom_joint_init,
    get_cached_cloud,
)

if _USE_BARE_FRANKA:
    raise RuntimeError("Bimanual unstable tool env requires robot_mode='tool', not bare_franka")

# Side-by-side bimanual setup: both arms face the same direction and are
# separated laterally around the object/goal workspace centerline.
_BIMANUAL_BASE_HALF_SPACING_Y = 0.35
_BIMANUAL_GOAL_XY_OFFSET_RANGE = min(float(_RL_CONTRACT.object_pose_sampling.xy_offset_range), 0.075)
_BIMANUAL_ARM_PROXIMITY_BODY_NAMES = ("panda_link5", "panda_link6", "panda_link7")

ROBOT1_BASE_POS = (0.0, -_BIMANUAL_BASE_HALF_SPACING_Y, 0.0)
ROBOT1_BASE_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT2_BASE_POS = (0.0, _BIMANUAL_BASE_HALF_SPACING_Y, 0.0)
ROBOT2_BASE_ROT = (1.0, 0.0, 0.0, 0.0)


def _offset_indices(indices: list[int], num_assets: int, offset: int) -> list[int]:
    return [int((idx + offset) % num_assets) for idx in indices]


TOOL1_ASSET_INDICES_BY_ENV = list(TOOL_ASSET_INDICES_BY_ENV)
TOOL1_USD_PATHS_FOR_SPAWN = list(TOOL_USD_PATHS_FOR_SPAWN)
TOOL2_ASSET_INDICES_BY_ENV = asset_indices_for_rank(
    _NUM_ENVS_PER_RANK,
    _GLOBAL_RANK,
    len(TOOL_DATA),
    randomize=_RANDOMIZE_TOOL_ASSIGNMENT,
    seed=_ASSET_ASSIGNMENT_SEED,
    salt=TOOL_ASSIGNMENT_SALT + 101,
) if _RANDOMIZE_TOOL_ASSIGNMENT else _offset_indices(TOOL1_ASSET_INDICES_BY_ENV, len(TOOL_DATA), 1)
TOOL2_SPAWN_ASSET_INDICES = (
    TOOL2_ASSET_INDICES_BY_ENV
    if _RANDOMIZE_TOOL_ASSIGNMENT
    else _offset_indices(
        sequential_spawn_indices_for_rank(_NUM_ENVS_PER_RANK, _GLOBAL_RANK, len(TOOL_DATA)),
        len(TOOL_DATA),
        1,
    )
)
TOOL2_USD_PATHS_FOR_SPAWN = [TOOL_USD_PATHS[index] for index in TOOL2_SPAWN_ASSET_INDICES]

print(
    f"[INFO] Bimanual tool assignment rank={_GLOBAL_RANK}/{_WORLD_SIZE} "
    f"local_rank={_LOCAL_RANK} envs={_NUM_ENVS_PER_RANK} "
    f"randomize={_RANDOMIZE_TOOL_ASSIGNMENT}"
)


def _assigned_index_for_env(env_id: int, assignment: list[int], label: str) -> int:
    env_id = int(env_id)
    if env_id < 0 or env_id >= len(assignment):
        raise ValueError(f"env_id {env_id} outside {label} assignment for {len(assignment)} envs")
    return assignment[env_id]


def get_tool1_index_for_env(env_id: int) -> int:
    return _assigned_index_for_env(env_id, TOOL1_ASSET_INDICES_BY_ENV, "tool1")


def get_tool2_index_for_env(env_id: int) -> int:
    return _assigned_index_for_env(env_id, TOOL2_ASSET_INDICES_BY_ENV, "tool2")


def get_tool1_data_for_env(env_id: int) -> dict:
    return TOOL_DATA[get_tool1_index_for_env(env_id)]


def get_tool2_data_for_env(env_id: int) -> dict:
    return TOOL_DATA[get_tool2_index_for_env(env_id)]


def _make_robot_cfg(
    *,
    prim_path: str,
    usd_paths: list[str],
    base_pos: tuple[float, float, float],
    base_rot: tuple[float, float, float, float],
) -> ArticulationCfg:
    robot_cfg = build_multi_tool_robot_cfg(usd_paths, random_choice=False)
    return robot_cfg.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base_pos,
            rot=base_rot,
            joint_pos=custom_joint_init,
        ),
    )


@configclass
class BimanualUnstableSceneCfg(InteractiveSceneCfg):
    """Scene with two independently-tooled Franka arms."""

    replicate_physics: bool = False

    if not _RL_CONTRACT.table.enabled:
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            debug_vis=False,
        )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    if _RL_CONTRACT.table.enabled:
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(_RL_CONTRACT.table.pose_xyz)),
            spawn=sim_utils.CuboidCfg(
                size=tuple(_RL_CONTRACT.table.size_xyz),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=_RL_CONTRACT.table.material.static_friction,
                    dynamic_friction=_RL_CONTRACT.table.material.dynamic_friction,
                    restitution=_RL_CONTRACT.table.material.restitution,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(_RL_CONTRACT.table.color_rgba[:3])),
            ),
        )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=OBJECT_ASSET_CFGS_FOR_SPAWN,
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    robot_1 = _make_robot_cfg(
        prim_path="{ENV_REGEX_NS}/Robot1",
        usd_paths=TOOL1_USD_PATHS_FOR_SPAWN,
        base_pos=ROBOT1_BASE_POS,
        base_rot=ROBOT1_BASE_ROT,
    )
    robot_2 = _make_robot_cfg(
        prim_path="{ENV_REGEX_NS}/Robot2",
        usd_paths=TOOL2_USD_PATHS_FOR_SPAWN,
        base_pos=ROBOT2_BASE_POS,
        base_rot=ROBOT2_BASE_ROT,
    )

    ee_frame_1 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BimanualEndEffectorFrameTransformer1"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot1/tool_mount/link_coacd_convex_piece_0",
                name="ee_tool_1",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    ee_frame_2 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot2/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BimanualEndEffectorFrameTransformer2"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot2/tool_mount/link_coacd_convex_piece_0",
                name="ee_tool_2",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class CommandsCfg:
    """Unstable/random target-pose command."""

    target_object_pose = mdp.RandomPoseCommandCfg(
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        xy_offset_range=_BIMANUAL_GOAL_XY_OFFSET_RANGE,
        initial_position_range=_RL_CONTRACT.object_pose_sampling.initial_position_range,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for arbitrary object target poses."""

    target_pose_stability = CurriculumTerm(
        func=mdp.target_pose_stability_curriculum,
        params={
            "command_name": "target_object_pose",
            "start_step": _RL_CONTRACT.curriculum.start_step,
            "end_step": _RL_CONTRACT.curriculum.end_step,
            "start_stable_probability": _RL_CONTRACT.curriculum.start_stable_pose_probability,
            "end_stable_probability": _RL_CONTRACT.curriculum.end_stable_pose_probability,
        },
    ) if _RL_CONTRACT.curriculum.enabled else None


@configclass
class BimanualRelativeJointPositionActionsCfg:
    """Two 7D relative joint-position action terms, concatenated to 14D."""

    arm1_action = RelativeJointPositionActionCfg(
        asset_name="robot_1",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )
    arm2_action = RelativeJointPositionActionCfg(
        asset_name="robot_2",
        joint_names=["panda_joint.*"],
        scale=_RL_CONTRACT.action.scale,
        use_zero_offset=True,
    )


@configclass
class ObservationsCfg:
    """Bimanual policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        object_cloud = ObsTerm(
            func=mdp.get_object_pointcloud_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        ) if _RL_CONTRACT.observation.include_object_cloud else None

        tool1_cloud = ObsTerm(
            func=mdp.get_tool1_pointcloud_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        ) if _RL_CONTRACT.observation.include_tool_cloud else None

        tool2_cloud = ObsTerm(
            func=mdp.get_tool2_pointcloud_in_env_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        ) if _RL_CONTRACT.observation.include_tool_cloud else None

        object_bbox_center = ObsTerm(func=mdp.get_obj_bbox_center) if _RL_CONTRACT.observation.include_bbox_centers else None
        tool1_bbox_center = ObsTerm(func=mdp.get_tool1_bbox_center) if _RL_CONTRACT.observation.include_bbox_centers else None
        tool2_bbox_center = ObsTerm(func=mdp.get_tool2_bbox_center) if _RL_CONTRACT.observation.include_bbox_centers else None

        hand1_state = ObsTerm(
            func=mdp.hand1_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        hand2_state = ObsTerm(
            func=mdp.hand2_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        robot1_state = ObsTerm(
            func=mdp.robot1_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        robot2_state = ObsTerm(
            func=mdp.robot2_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        previous_action = ObsTerm(func=mdp.last_action)

        relative_goal_pose = ObsTerm(
            func=mdp.rel_pose_goal,
            params={"command_name": "target_object_pose"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        object_velocity = ObsTerm(func=mdp.object_root_velocity)

        phys_params = ObsTerm(
            func=mdp.bimanual_phys_params,
            params={"field_names": _PHYSICS_OBSERVATION_FIELDS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and randomization events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_initial_object_position,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": _RL_CONTRACT.domain_randomization.object.scale.range,
            "asset_cfg": SceneEntityCfg("object"),
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.scale) else None

    randomize_tool1_mass = EventTerm(
        func=mdp.randomize_tool_mass,
        mode="reset",
        params={
            "mass_range": _RL_CONTRACT.domain_randomization.tool.mass.range,
            "robot_cfg": SceneEntityCfg("robot_1"),
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.mass) else None

    randomize_tool2_mass = EventTerm(
        func=mdp.randomize_tool_mass,
        mode="reset",
        params={
            "mass_range": _RL_CONTRACT.domain_randomization.tool.mass.range,
            "robot_cfg": SceneEntityCfg("robot_2"),
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.mass) else None

    randomize_tool1_friction = EventTerm(
        func=mdp.randomize_tool_friction,
        mode="reset",
        params={
            "static_friction_range": _RL_CONTRACT.domain_randomization.tool.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.tool.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.tool.material.restitution_range,
            "robot_cfg": SceneEntityCfg("robot_1"),
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.material) else None

    randomize_tool2_friction = EventTerm(
        func=mdp.randomize_tool_friction,
        mode="reset",
        params={
            "static_friction_range": _RL_CONTRACT.domain_randomization.tool.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.tool.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.tool.material.restitution_range,
            "robot_cfg": SceneEntityCfg("robot_2"),
        },
    ) if _tool_dr_event_enabled(_RL_CONTRACT.domain_randomization.tool.material) else None

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": _RL_CONTRACT.domain_randomization.object.mass.range,
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": _RL_CONTRACT.domain_randomization.object.mass.recompute_inertia,
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.mass) else None

    randomize_object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": _RL_CONTRACT.domain_randomization.object.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.object.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.object.material.restitution_range,
            "num_buckets": _RL_CONTRACT.domain_randomization.object.material.num_buckets,
            "make_consistent": _RL_CONTRACT.domain_randomization.object.material.make_consistent,
        },
    ) if _dr_event_enabled(_RL_CONTRACT.domain_randomization.object.material) else None

    randomize_terrain_material = EventTerm(
        func=mdp.randomize_terrain_material,
        mode="reset",
        params={
            "static_friction_range": _RL_CONTRACT.domain_randomization.ground.material.static_friction_range,
            "dynamic_friction_range": _RL_CONTRACT.domain_randomization.ground.material.dynamic_friction_range,
            "restitution_range": _RL_CONTRACT.domain_randomization.ground.material.restitution_range,
            "num_buckets": _RL_CONTRACT.domain_randomization.ground.material.num_buckets,
        },
    ) if (
        not _RL_CONTRACT.table.enabled
        and _dr_event_enabled(_RL_CONTRACT.domain_randomization.ground.material)
    ) else None

    randomize_table_material = _unsupported_event(
        _RL_CONTRACT.table.enabled
        and _dr_event_enabled(_RL_CONTRACT.domain_randomization.table_surface.material),
        "table_surface material randomization",
    )


@configclass
class RewardsCfg:
    """Unstable rewards with bimanual contact/energy terms."""

    task_success = RewTerm(
        func=mdp.task_success_from_termination,
        params={"term_name": "reached", "base_reward": 1.0},
        weight=_RL_CONTRACT.reward.task_success_term_weight,
    )

    contact_reward = RewTerm(
        func=mdp.bimanual_object_ee_distance_tanh,
        params={"std": _RL_CONTRACT.reward.contact_std},
        weight=_RL_CONTRACT.reward.contact_term_weight,
    )

    object_goal_tracking = RewTerm(
        func=mdp.bimanual_object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_tracking_term_weight,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.bimanual_object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_fine_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_tracking_fine_term_weight,
    )

    object_within_goal_threshold = RewTerm(
        func=mdp.object_within_goal_threshold,
        params={
            "command_name": "target_object_pose",
            "threshold": _RL_CONTRACT.reward.success_threshold,
            "rotation_threshold": _RL_CONTRACT.reward.rotation_threshold,
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_threshold_term_weight,
    )

    arm_proximity_penalty = RewTerm(
        func=mdp.bimanual_link_proximity_penalty,
        params={
            "warning_distance": _RL_CONTRACT.reward.bimanual_arm_proximity_warning_distance,
            "failure_distance": _RL_CONTRACT.reward.bimanual_arm_proximity_failure_distance,
            "robot1_cfg": SceneEntityCfg("robot_1", body_names=list(_BIMANUAL_ARM_PROXIMITY_BODY_NAMES)),
            "robot2_cfg": SceneEntityCfg("robot_2", body_names=list(_BIMANUAL_ARM_PROXIMITY_BODY_NAMES)),
        },
        weight=_RL_CONTRACT.reward.bimanual_arm_proximity_penalty_weight,
    )

    energy_penalty = RewTerm(
        func=mdp.bimanual_joint_power_penalty,
        params={"k_e": 0.0001},
        weight=_RL_CONTRACT.reward.energy_penalty_weight,
    )


@configclass
class TerminationsCfg:
    """Unstable target-pose terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached = DoneTerm(
        func=mdp.object_reached_goal_dwell,
        params={
            "command_name": "target_object_pose",
            "threshold": _RL_CONTRACT.reward.success_threshold,
            "rotation_threshold": _RL_CONTRACT.reward.rotation_threshold,
            "dwell_steps": _RL_CONTRACT.reward.stable_success_dwell_steps,
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    object_dropped = DoneTerm(
        func=mdp.object_dropped_off_table,
        params={"minimum_height": _table_top_z() - 0.15},
    )
    arms_too_close = DoneTerm(
        func=mdp.bimanual_links_too_close,
        params={
            "threshold": _RL_CONTRACT.reward.bimanual_arm_proximity_failure_distance,
            "robot1_cfg": SceneEntityCfg("robot_1", body_names=list(_BIMANUAL_ARM_PROXIMITY_BODY_NAMES)),
            "robot2_cfg": SceneEntityCfg("robot_2", body_names=list(_BIMANUAL_ARM_PROXIMITY_BODY_NAMES)),
        },
    )


@configclass
class BimanualUnstableEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual unstable tool task."""

    scene: BimanualUnstableSceneCfg = BimanualUnstableSceneCfg(
        num_envs=_RL_CONTRACT.env.num_envs,
        env_spacing=_RL_CONTRACT.env.env_spacing,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: BimanualRelativeJointPositionActionsCfg = BimanualRelativeJointPositionActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    normalize_observations: bool = True
    object_cloud_centering: str = _RL_CONTRACT.observation.object_cloud_centering
    tool_cloud_centering: str = _RL_CONTRACT.observation.tool_cloud_centering
    mesh_centering: str = _RL_CONTRACT.observation.mesh_centering
    action_dim: int = 14
    observation_dim: int = _OBSERVATION_DIM
    physics_dim: int = _PHYSICS_DIM
    robot_mode: str = _ROBOT_MODE
    bimanual: bool = True
    physics_observation_fields: tuple[str, ...] = _PHYSICS_OBSERVATION_FIELDS
    table_enabled: bool = _RL_CONTRACT.table.enabled
    table_size_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.size_xyz)
    table_pose_xyz: tuple[float, float, float] = tuple(_RL_CONTRACT.table.pose_xyz)
    table_bounds_xy: tuple[tuple[float, float], tuple[float, float]] = _table_bounds_xy()
    table_placement_margin_xy: float = _RL_CONTRACT.table.placement_margin_xy
    table_placement_max_attempts: int = _RL_CONTRACT.table.placement_max_attempts
    table_material = _RL_CONTRACT.table.material

    visualize_current_object_pose: bool = True
    visualize_object_pointcloud: bool = False
    visualize_tool1_pointcloud: bool = False
    visualize_tool2_pointcloud: bool = False
    visualize_eef_position: bool = False
    visualize_object_velocity_mass: bool = False
    visualize_tool_velocity_mass: bool = False
    use_torch_compile: bool = True
    enforce_joint_limits: bool = False
    disable_obs_noise: bool = False

    def __post_init__(self) -> None:
        if self.disable_obs_noise:
            policy_cfg = getattr(self.observations, "policy", None)
            if policy_cfg is not None:
                for attr_name in dir(policy_cfg):
                    if attr_name.startswith("_"):
                        continue
                    term = getattr(policy_cfg, attr_name, None)
                    noise = getattr(term, "noise", None)
                    if noise is None:
                        continue
                    if hasattr(noise, "mean"):
                        noise.mean = 0.0
                    if hasattr(noise, "std"):
                        noise.std = 0.0

        self.decimation = _RL_CONTRACT.env.decimation
        self.episode_length_s = _RL_CONTRACT.env.episode_length_s
        self.viewer.eye = (2.5, 0.5, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = _RL_CONTRACT.env.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_position_iteration_count = _RL_CONTRACT.env.solver_position_iteration_count
        self.sim.physx.solver_velocity_iteration_count = _RL_CONTRACT.env.solver_velocity_iteration_count


class BimanualUnstableEnv(ManagerBasedRLEnv):
    """Bimanual environment wrapper with success-rate tracking."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._apply_joint_limits("robot_1")
        self._apply_joint_limits("robot_2")
        self._head_area_offsets_1 = torch.zeros(self.num_envs, 3, device=self.device)
        self._head_area_offsets_2 = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.total_episodes = 0
        self.total_successes = 0
        self.recent_success_window = deque(maxlen=100)
        self.recent_success_rate = 0.0
        self._global_step = 0
        self.post_reset()

    def _apply_joint_limits(self, robot_name: str) -> None:
        robot = self.scene[robot_name]
        entity = SceneEntityCfg(robot_name, joint_names=["panda_joint.*"])
        entity.resolve(self.scene)
        joint_ids = entity.joint_ids
        mins = torch.tensor(JOINT_BOX_MIN, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        maxs = torch.tensor(JOINT_BOX_MAX, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        if self.cfg.enforce_joint_limits:
            limits = robot.data.soft_joint_pos_limits
            limits[:, joint_ids, 0] = mins
            limits[:, joint_ids, 1] = maxs
            robot.data.soft_joint_pos_limits[:] = limits

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        success_mask = self.termination_manager.get_term("reached")
        self.episode_success_buf = self.episode_success_buf | success_mask
        episode_ended = terminated | truncated
        if torch.any(episode_ended):
            ended_env_ids = torch.where(episode_ended)[0]
            for env_id in ended_env_ids:
                self.total_episodes += 1
                episode_success = self.episode_success_buf[env_id].item()
                if episode_success:
                    self.total_successes += 1
                self.recent_success_window.append(episode_success)
            self._episode_success_before_reset = self.episode_success_buf.clone()
            self.episode_success_buf[episode_ended] = False
            if self.total_episodes > 0:
                success_rate = self.total_successes / self.total_episodes
                if "log" not in self.extras:
                    self.extras["log"] = dict()
                self.extras["log"]["success_rate"] = success_rate
                self.extras["log"]["total_episodes"] = self.total_episodes
                self.extras["log"]["total_successes"] = self.total_successes
                if len(self.recent_success_window) > 0:
                    self.recent_success_rate = sum(self.recent_success_window) / len(self.recent_success_window)
                    self.extras["log"]["recent_success_rate"] = self.recent_success_rate
        return obs, reward, terminated, truncated, info

    def post_reset(self):
        self._head_area_offsets_1 = mdp.compute_head_area_offsets_for_slot(self, tool_slot=1)
        self._head_area_offsets_2 = mdp.compute_head_area_offsets_for_slot(self, tool_slot=2)

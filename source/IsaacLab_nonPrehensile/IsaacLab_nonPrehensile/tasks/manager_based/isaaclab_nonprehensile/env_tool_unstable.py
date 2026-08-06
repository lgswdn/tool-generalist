"""Random-pose support variant of ``env_tool``.

This task reuses the base scene, actions, events, and environment class from
``env_tool``. It swaps the target command to arbitrary orientations and adds
object root velocity to the policy observation.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env_tool import (
    NonPrehensileEnv,
    NonPrehensileEnvCfg,
    _PHYSICS_OBSERVATION_FIELDS,
    _RL_CONTRACT,
    _USE_BARE_FRANKA,
    _table_top_z,
)


class NonPrehensileUnstableEnv(NonPrehensileEnv):
    """Environment class for arbitrary target-pose support."""


@configclass
class CommandsCfg:
    """Command terms for arbitrary object target poses."""

    target_object_pose = mdp.RandomPoseCommandCfg(
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        xy_offset_range=_RL_CONTRACT.object_pose_sampling.xy_offset_range,
        initial_position_range=_RL_CONTRACT.object_pose_sampling.initial_position_range,
        stable_pose_probability=(
            _RL_CONTRACT.curriculum.start_stable_pose_probability
            if _RL_CONTRACT.curriculum.enabled
            else _RL_CONTRACT.curriculum.end_stable_pose_probability
        ),
        secondary_task=_RL_CONTRACT.object_pose_sampling.secondary_task,
        grasp_lift_height=_RL_CONTRACT.object_pose_sampling.grasp_lift_height,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for arbitrary target-pose support."""

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
class ObservationsCfg:
    """Policy observations for the random-pose support task."""

    @configclass
    class PolicyCfg(ObsGroup):
        object_cloud = ObsTerm(
            func=mdp.get_object_pointcloud_in_env_frame,
            noise=(
                GaussianNoiseCfg(mean=0.0, std=0.005, operation="add")
                if _RL_CONTRACT.observation.point_cloud_noise_enabled
                else None
            ),
        ) if _RL_CONTRACT.observation.include_object_cloud else None

        tool_cloud = ObsTerm(
            func=mdp.get_tool_pointcloud_in_env_frame,
            noise=(
                GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")
                if _RL_CONTRACT.observation.point_cloud_noise_enabled
                else None
            ),
        ) if (_RL_CONTRACT.observation.include_tool_cloud and not _USE_BARE_FRANKA) else None

        kinematic_gripper_clouds = ObsTerm(
            func=mdp.get_generated_gripper_kinematic_state_clouds,
        ) if _RL_CONTRACT.observation.include_kinematic_gripper_clouds else None

        object_bbox_center = ObsTerm(
            func=mdp.get_obj_bbox_center
        ) if _RL_CONTRACT.observation.include_bbox_centers else None

        tool_bbox_center = ObsTerm(
            func=mdp.get_tool_bbox_center
        ) if (
            _RL_CONTRACT.observation.include_bbox_centers
            and _RL_CONTRACT.observation.include_tool_cloud
            and not _USE_BARE_FRANKA
        ) else None

        hand_state = ObsTerm(
            func=mdp.hand_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        robot_state = ObsTerm(
            func=mdp.robot_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        previous_action = ObsTerm(func=mdp.last_action)

        relative_goal_pose = ObsTerm(
            func=mdp.rel_pose_goal,
            params={"command_name": "target_object_pose"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )

        task_embedding = ObsTerm(
            func=mdp.target_pose_task_embedding,
            params={"command_name": "target_object_pose"},
        ) if _RL_CONTRACT.observation.task_embedding_dim > 0 else None

        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"field_names": _PHYSICS_OBSERVATION_FIELDS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for pose tracking and threshold success."""

    task_success = RewTerm(
        func=mdp.task_success_from_termination,
        params={
            "term_name": "reached",
            "base_reward": 1.0,
        },
        weight=_RL_CONTRACT.reward.task_success_term_weight,
    )

    contact_reward = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={"std": _RL_CONTRACT.reward.contact_std},
        weight=_RL_CONTRACT.reward.contact_term_weight,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=_RL_CONTRACT.reward.object_goal_tracking_term_weight,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": _RL_CONTRACT.reward.object_goal_fine_std,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": _RL_CONTRACT.reward.contact_std,
            "rotation_distance_divisor": getattr(_RL_CONTRACT.reward, "rotation_distance_divisor", 5.0),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
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

    energy_penalty = RewTerm(
        func=mdp.joint_power_penalty,
        params={"k_e": 0.0001},
        weight=_RL_CONTRACT.reward.energy_penalty_weight,
    )


@configclass
class TerminationsCfg:
    """Termination terms for arbitrary-pose support."""

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


@configclass
class NonPrehensileUnstableEnvCfg(NonPrehensileEnvCfg):
    """Configuration for the arbitrary target-pose support task."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

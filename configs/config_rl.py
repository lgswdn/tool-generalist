"""RL-stage config for experiment planning.

The concrete Isaac Lab and rsl_rl adapters are deliberately not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RangeCfg:
    low: float
    high: float


@dataclass
class RangeRandomizationCfg:
    enabled: bool = False
    mode: str = "reset"
    distribution: str = "uniform"
    range: tuple[float, float] = (0.0, 1.0)


@dataclass
class MassRandomizationCfg(RangeRandomizationCfg):
    range: tuple[float, float] = (0.05, 0.5)
    recompute_inertia: bool = True
    body_name: Optional[str] = None


@dataclass
class MaterialRandomizationCfg:
    enabled: bool = False
    mode: str = "reset"
    static_friction_range: tuple[float, float] = (0.4, 1.2)
    dynamic_friction_range: tuple[float, float] = (0.4, 1.2)
    restitution_range: tuple[float, float] = (0.0, 0.0)
    num_buckets: int = 256
    make_consistent: bool = True
    body_name: Optional[str] = None
    per_env: bool = False


@dataclass
class ObjectRandomizationCfg:
    scale: RangeRandomizationCfg = field(
        default_factory=lambda: RangeRandomizationCfg(
            enabled=True,
            mode="prestartup",
            range=(0.1, 0.2),
        )
    )
    mass: MassRandomizationCfg = field(
        default_factory=lambda: MassRandomizationCfg(enabled=True, range=(0.1, 0.5))
    )
    material: MaterialRandomizationCfg = field(
        default_factory=lambda: MaterialRandomizationCfg(
            enabled=True,
            static_friction_range=(0.7, 1.0),
            dynamic_friction_range=(0.7, 1.0),
            restitution_range=(0.1, 0.2),
        )
    )


@dataclass
class ToolRandomizationCfg:
    mass: MassRandomizationCfg = field(
        default_factory=lambda: MassRandomizationCfg(enabled=True, range=(0.1, 0.5))
    )
    material: MaterialRandomizationCfg = field(
        default_factory=lambda: MaterialRandomizationCfg(
            enabled=True,
            static_friction_range=(0.8, 1.5),
            dynamic_friction_range=(0.8, 1.5),
        )
    )


@dataclass
class SurfaceRandomizationCfg:
    material: MaterialRandomizationCfg = field(
        default_factory=lambda: MaterialRandomizationCfg(
            static_friction_range=(0.3, 0.8),
            dynamic_friction_range=(0.3, 0.8),
        )
    )


@dataclass
class DomainRandomizationCfg:
    name: str = "physics_dr_default"
    enabled: bool = True
    preset: str = "default"
    seed_offset: int = 0
    apply_on_train: bool = True
    apply_on_eval: bool = False
    object: ObjectRandomizationCfg = field(default_factory=ObjectRandomizationCfg)
    tool: ToolRandomizationCfg = field(default_factory=ToolRandomizationCfg)
    ground: SurfaceRandomizationCfg = field(
        default_factory=lambda: SurfaceRandomizationCfg(
            material=MaterialRandomizationCfg(
                enabled=True,
                static_friction_range=(0.3, 0.8),
                dynamic_friction_range=(0.3, 0.8),
            )
        )
    )
    table_surface: SurfaceRandomizationCfg = field(default_factory=SurfaceRandomizationCfg)

    @property
    def object_mass(self) -> RangeCfg:
        low, high = self.object.mass.range
        return RangeCfg(low, high)

    @object_mass.setter
    def object_mass(self, value: RangeCfg) -> None:
        self.object.mass.range = (value.low, value.high)

    @property
    def tool_mass(self) -> RangeCfg:
        low, high = self.tool.mass.range
        return RangeCfg(low, high)

    @tool_mass.setter
    def tool_mass(self, value: RangeCfg) -> None:
        self.tool.mass.range = (value.low, value.high)

    @property
    def object_friction(self) -> RangeCfg:
        low, high = self.object.material.static_friction_range
        return RangeCfg(low, high)

    @object_friction.setter
    def object_friction(self, value: RangeCfg) -> None:
        self.object.material.static_friction_range = (value.low, value.high)
        self.object.material.dynamic_friction_range = (value.low, value.high)

    @property
    def tool_friction(self) -> RangeCfg:
        low, high = self.tool.material.static_friction_range
        return RangeCfg(low, high)

    @tool_friction.setter
    def tool_friction(self, value: RangeCfg) -> None:
        self.tool.material.static_friction_range = (value.low, value.high)
        self.tool.material.dynamic_friction_range = (value.low, value.high)

    @property
    def ground_friction(self) -> RangeCfg:
        low, high = self.ground.material.static_friction_range
        return RangeCfg(low, high)

    @ground_friction.setter
    def ground_friction(self, value: RangeCfg) -> None:
        self.ground.material.static_friction_range = (value.low, value.high)
        self.ground.material.dynamic_friction_range = (value.low, value.high)


@dataclass
class RewardCfg:
    preset: str = "default"
    task_success_term_weight: float = 2000.0
    contact_term_weight: float = 1.0
    object_goal_tracking_term_weight: float = 4.0
    object_goal_tracking_fine_term_weight: float = 12.0
    energy_penalty_weight: float = -0.5
    success_threshold: float = 0.05
    rotation_threshold: float = 0.1
    object_goal_std: float = 0.5
    object_goal_fine_std: float = 0.3
    contact_std: float = 0.15
    rotation_distance_divisor: float = 5.0


@dataclass
class MaterialCfg:
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0


@dataclass
class TableCfg:
    enabled: bool = False
    per_env_instance: bool = True
    size_xyz: list[float] = field(default_factory=lambda: [1.5, 1.5, 0.04])
    pose_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.02])
    placement_margin_xy: float = 0.02
    placement_max_attempts: int = 64
    color_rgba: list[float] = field(default_factory=lambda: [0.45, 0.45, 0.45, 1.0])
    material_name: str = "default"
    material: MaterialCfg = field(default_factory=MaterialCfg)

    @property
    def size_xy(self) -> list[float]:
        return list(self.size_xyz[:2])

    @size_xy.setter
    def size_xy(self, value: list[float]) -> None:
        self.size_xyz[:2] = value[:2]

    @property
    def thickness(self) -> float:
        return self.size_xyz[2]

    @thickness.setter
    def thickness(self, value: float) -> None:
        self.size_xyz[2] = value

    @property
    def top_z(self) -> float:
        return self.pose_xyz[2] + self.thickness / 2.0

    @top_z.setter
    def top_z(self, value: float) -> None:
        self.pose_xyz[2] = value - self.thickness / 2.0


@dataclass
class ObjectPoseSamplingCfg:
    """Object pose sampling ranges for the original IsaacLab-nonPrehensile XY distributions."""

    initial_position_range: float = 0.15
    """Initial x uses 0.5 +/- range; initial y uses 0.0 +/- 2 * range."""

    xy_offset_range: float = 0.15
    """Target x/y offset magnitude is sampled from [0.5 * range, range]."""


@dataclass
class PPOCfg:
    algorithm: str = "PPO"
    class_name: str = "PPO"
    num_steps_per_env: int = 8
    max_iterations: int = 1000000
    save_interval: int = 200
    learning_rate: float = 5.0e-5
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.3
    entropy_coef: float = 0.005
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.016
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 8
    num_mini_batches: int = 8


@dataclass
class ActionCfg:
    mode: str = "relative_joint_position"
    action_dim: int = 7
    joint_names: list[str] = field(default_factory=lambda: ["panda_joint.*"])
    scale: float | list[float] = 0.1
    clip: tuple[float, float] = (-1.0, 1.0)
    impedance_mode: str = "fixed"
    include_stiffness: bool = False
    include_damping: bool = False
    stiffness_dim: int = 0
    damping_dim: int = 0

    @property
    def resolved_action_dim(self) -> int:
        stiffness_dim = self.stiffness_dim if self.include_stiffness else 0
        damping_dim = self.damping_dim if self.include_damping else 0
        return self.action_dim + stiffness_dim + damping_dim


@dataclass
class ObservationCfg:
    num_points: int = 512
    point_dim: int = 3
    hand_state_dim: int = 9
    robot_state_dim: int = 14
    previous_action_dim: Optional[int] = None
    relative_goal_dim: int = 9
    bbox_center_dim: int = 6
    include_object_cloud: bool = True
    include_tool_cloud: bool = True
    include_bbox_centers: bool = True
    object_cloud_centering: str = "none"
    tool_cloud_centering: str = "none"
    mesh_centering: str = "none"
    model_input_centering: str = "bbox_center"
    tool_cloud_source: str = "adjusted_decomposed_mesh"
    flatten_clouds: bool = True
    layout: list[str] = field(
        default_factory=lambda: [
            "object_cloud_flat",
            "tool_cloud_flat",
            "object_bbox_center",
            "tool_bbox_center",
            "hand_state",
            "robot_state",
            "previous_action",
            "relative_goal_pose",
            "physics",
        ]
    )

    @property
    def physics_dim(self) -> None:
        """Compatibility alias; ``RLCfg.physics_observation_fields`` owns this."""

        return None

    @property
    def cloud_dim(self) -> int:
        cloud_count = int(self.include_object_cloud) + int(self.include_tool_cloud)
        return cloud_count * self.num_points * self.point_dim

    def resolved_previous_action_dim(self, action_dim: int) -> int:
        return self.previous_action_dim if self.previous_action_dim is not None else action_dim

    def resolved_observation_dim(self, action_dim: int, physics_dim: int) -> int:
        bbox_center_dim = self.bbox_center_dim if self.include_bbox_centers else 0
        return (
            self.cloud_dim
            + bbox_center_dim
            + self.hand_state_dim
            + self.robot_state_dim
            + self.resolved_previous_action_dim(action_dim)
            + self.relative_goal_dim
            + physics_dim
        )


@dataclass
class RLEnvCfg:
    # Per-GPU/per-rank environment count.  In distributed multi-GPU RL,
    # total envs = ExpCfg.num_gpus * RLCfg.env.num_envs.
    num_envs: int = 1024
    episode_length_s: float = 30.0
    decimation: int = 8
    # sim_dt * decimation == 0.1
    sim_dt: float = 1.0 / 80.0
    env_spacing: float = 2.0
    solver_position_iteration_count: int = 8
    solver_velocity_iteration_count: int = 1

    @property
    def max_iterations(self) -> None:
        """Compatibility alias; ``PPOCfg.max_iterations`` is canonical."""

        return None

    @property
    def action_dim(self) -> None:
        """Compatibility alias; ``ActionCfg`` owns action dimensions."""

        return None

    @property
    def observation_layout(self) -> tuple[str, ...]:
        """Compatibility alias; ``ObservationCfg.layout`` is canonical."""

        return (
            "object_cloud_flat",
            "tool_cloud_flat",
            "object_bbox_center",
            "tool_bbox_center",
            "hand_state",
            "robot_state",
            "previous_action",
            "relative_goal_pose",
            "physics",
        )


@dataclass
class RLLaunchCfg:
    headless: bool = True
    enable_cameras: bool = False
    disable_fabric: bool = False
    device: Optional[str] = None
    logger: str = "tensorboard"
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    init_at_random_ep_len: bool = True
    distributed: bool = False


@dataclass
class RLCfg:
    name: str = "rl_default"
    enabled: bool = False
    isaac_task_id: Optional[str] = "tool-sdf-v0"
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    rsl_rl_cfg_entry_point: Optional[str] = (
        "IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile."
        "agents.config.rsl_rl_ppo_cfg:TGPPORunnerCfg"
    )
    rsl_rl_train_entrypoint: str = "scripts.train:run_rl_training"
    actor_critic_class: str = "ActorCriticTG"
    algorithm_class: str = "PPO"
    ppo: PPOCfg = field(default_factory=PPOCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    env: RLEnvCfg = field(default_factory=RLEnvCfg)
    launch: RLLaunchCfg = field(default_factory=RLLaunchCfg)
    encoder_checkpoint: Optional[str] = None
    freeze_encoder: bool = True
    separate_actor_critic_fusion: bool = False
    domain_randomization: DomainRandomizationCfg = field(
        default_factory=DomainRandomizationCfg
    )
    reward: RewardCfg = field(default_factory=RewardCfg)
    table: TableCfg = field(default_factory=TableCfg)
    object_pose_sampling: ObjectPoseSamplingCfg = field(default_factory=ObjectPoseSamplingCfg)

    @property
    def effective_action_dim(self) -> int:
        return self.action.resolved_action_dim

    @property
    def effective_physics_dim(self) -> int:
        return len(self.physics_observation_fields)

    @property
    def effective_observation_dim(self) -> int:
        return self.observation.resolved_observation_dim(
            self.effective_action_dim,
            self.effective_physics_dim,
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @num_envs.setter
    def num_envs(self, value: int) -> None:
        self.env.num_envs = value

    @property
    def max_iterations(self) -> int:
        return self.ppo.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self.ppo.max_iterations = value

    @property
    def action_dim(self) -> int:
        return self.effective_action_dim

    @action_dim.setter
    def action_dim(self, value: int) -> None:
        self.action.action_dim = value

    @property
    def physics_dim(self) -> int:
        return self.effective_physics_dim

    @property
    def physics_observation_fields(self) -> tuple[str, ...]:
        """Fields emitted by mdp.phys_params, derived from enabled DR/table config."""

        fields: list[str] = []
        dr = self.domain_randomization
        if dr.enabled:
            if dr.object.mass.enabled:
                fields.append("object_mass")
            if dr.object.material.enabled:
                fields.extend(
                    (
                        "object_static_friction",
                        "object_dynamic_friction",
                        "object_restitution",
                    )
                )
            if dr.tool.mass.enabled:
                fields.append("tool_mass")
            if dr.tool.material.enabled:
                fields.extend(
                    (
                        "tool_static_friction",
                        "tool_dynamic_friction",
                        "tool_restitution",
                    )
                )
            if dr.ground.material.enabled and not self.table.enabled:
                fields.extend(
                    (
                        "ground_static_friction",
                        "ground_dynamic_friction",
                        "ground_restitution",
                    )
                )

        if self.table.enabled:
            fields.extend(
                (
                    "table_static_friction",
                    "table_dynamic_friction",
                    "table_restitution",
                )
            )
        return tuple(fields)

    def resolved_task_id(self) -> Optional[str]:
        return self.isaac_task_id or self.task_id or self.task_name

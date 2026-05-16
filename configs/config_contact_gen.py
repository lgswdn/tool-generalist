"""Contact-generation config owned by the new experiment framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContactPhysicsCfg:
    runner: str = "isaac"
    num_workers: int = 1
    t_stabilize: int = 60
    t_postcontact: int = 5
    unsigned_distance_accept_eps: float = 0.005
    stabilize_linear_velocity_eps: float = 1e-3
    stabilize_angular_velocity_eps: float = 1e-3
    post_delta_seed: Optional[int] = None
    post_delta_translation_min: tuple[float, float, float] = (-0.005, -0.005, -0.005)
    post_delta_translation_max: tuple[float, float, float] = (0.005, 0.005, 0.005)
    post_delta_rotation_max_rad: float = 0.1
    post_tool_reach_translation_eps: float = 2e-3
    post_tool_reach_rotation_eps_rad: float = 5e-2
    post_object_table_z_min: float = 0.0
    post_linear_velocity_eps: float = 1e-2
    post_angular_velocity_eps: float = 1e-2
    object_mass_range: tuple[float, float] = (0.1, 0.5)
    tool_mass_range: tuple[float, float] = (0.1, 0.5)
    object_friction_range: tuple[float, float] = (0.7, 1.0)
    tool_friction_range: tuple[float, float] = (0.8, 1.5)
    ground_friction_range: tuple[float, float] = (0.3, 0.8)


@dataclass
class ContactVisualizationCfg:
    enabled: bool = False
    stabilization_picture: bool = False
    stabilization_picture_num: int = 8
    postcontact_video: bool = False
    postcontact_video_num: int = 8
    video_dir: Optional[str] = "~/tool-generalist/video"
    picture_dir: Optional[str] = "~/tool-generalist/video"
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 30
    camera_pos: tuple[float, float, float] = (0.24, 0.14, 0.18)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.05)
    # Compatibility alias for older configs; prefer postcontact_video_num.
    max_visualized_candidates: int = 1


@dataclass
class ContactGenCfg:
    name: str = "contact_default"
    enabled: bool = False
    regenerate: bool = False
    schema_version: str = "contact_pt_v1"
    num_pairs: int = 300
    num_object_poses: int = 10
    B: int = 2048
    M: int = 4096
    chunk_B: int = 512
    object_scale_range: tuple[float, float] = (0.1, 0.2)
    num_surface_pts: int = 512
    sdf_grid_res: int = 128
    sdf_padding: float = 0.03
    contact_mode_prob: dict[str, float] = field(
        default_factory=lambda: {"head": 0.75, "body": 0.25}
    )
    epsilon: float = 2e-3
    floor_eps: float = 0.0
    upright_threshold: float = 0.0
    max_contacts_per_pair: int = 1
    penetration_eps: float = 5e-4
    rotation_orth_eps: float = 1e-4
    artifact_subdir: str = "contact"
    skip_existing: bool = True
    physics: ContactPhysicsCfg = field(default_factory=ContactPhysicsCfg)
    visualization: ContactVisualizationCfg = field(default_factory=ContactVisualizationCfg)

    @property
    def tool_scale(self) -> float:
        """Compatibility alias; ``GeneralCfg.tool_mount.scale_xyz`` is canonical."""

        return 0.1

    @property
    def stabilize_steps(self) -> int:
        return self.physics.t_stabilize

    @stabilize_steps.setter
    def stabilize_steps(self, value: int) -> None:
        self.physics.t_stabilize = value

    @property
    def postcontact_steps(self) -> int:
        return self.physics.t_postcontact

    @postcontact_steps.setter
    def postcontact_steps(self, value: int) -> None:
        self.physics.t_postcontact = value

    @property
    def object_mass_range(self) -> tuple[float, float]:
        return self.physics.object_mass_range

    @object_mass_range.setter
    def object_mass_range(self, value: tuple[float, float]) -> None:
        self.physics.object_mass_range = value

    @property
    def tool_mass_range(self) -> tuple[float, float]:
        return self.physics.tool_mass_range

    @tool_mass_range.setter
    def tool_mass_range(self, value: tuple[float, float]) -> None:
        self.physics.tool_mass_range = value

    @property
    def object_friction_range(self) -> tuple[float, float]:
        return self.physics.object_friction_range

    @object_friction_range.setter
    def object_friction_range(self, value: tuple[float, float]) -> None:
        self.physics.object_friction_range = value

    @property
    def tool_friction_range(self) -> tuple[float, float]:
        return self.physics.tool_friction_range

    @tool_friction_range.setter
    def tool_friction_range(self, value: tuple[float, float]) -> None:
        self.physics.tool_friction_range = value

    @property
    def ground_friction_range(self) -> tuple[float, float]:
        return self.physics.ground_friction_range

    @ground_friction_range.setter
    def ground_friction_range(self, value: tuple[float, float]) -> None:
        self.physics.ground_friction_range = value

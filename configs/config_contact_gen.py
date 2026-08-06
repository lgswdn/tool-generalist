"""Contact-generation config owned by the new experiment framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


ROTATION_SELECTION_MOST_DOWNWARD = "most_downward_legal_tool_z_axis"
ROTATION_SELECTION_RANDOM_LEGAL = "random_legal"
ROTATION_SELECTION_MOST_CAVITY_CENTERED = "most_cavity_centered_legal"
TOOL_SOURCE_SELECTED_TOOLS = "selected_tools"
TOOL_SOURCE_OBJECTS = "objects"
CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION = "anchor_pair_rejection"
CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST = "bbox_translation_nearest"
CONTACT_GEOMETRY_INTERSECTING_ANCHORS = "intersecting_anchor_pairs"
CONTACT_GEOMETRY_TANGENT_GAUSSIAN = "tangent_gaussian"
PENETRATION_CHECK_TOOL_INTO_OBJECT = "tool_into_object"
PENETRATION_CHECK_BIDIRECTIONAL = "bidirectional_surface_sdf"


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
    object_scale_range: tuple[float, float] = (0.1, 0.4)
    num_surface_pts: int = 512
    sdf_grid_res: int = 128
    sdf_padding: float = 0.03
    epsilon: float = 2e-3
    floor_eps: float = 0.0
    upright_threshold: float = 0.0
    contact_geometry_mode: str = CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION
    # When true, geometry candidates are the terminal dataset: Isaac
    # stabilization and post-contact rollout are intentionally omitted.
    geometry_only: bool = False
    # Require anchors to come from the explicit contact_tip_mesh stored in the
    # adjusted tool catalog. It never falls back to the full tool cloud.
    require_tool_tip_anchor: bool = False
    # Refill rejected anchor pairs until exactly B candidates are produced.
    # Fail rather than silently creating a smaller dataset if this target
    # cannot be reached within rejection_max_rounds.
    rejection_refill: bool = False
    rejection_max_rounds: int = 1
    # Select exactly the same number of object pairs for every selected tool.
    # This is required by controlled representation comparisons whose contact
    # dataset must expose every gripper equally often.
    balanced_tool_pairs: bool = False
    # Treat any missing pair/pose output as a fatal stage error. Controlled
    # comparisons must never silently continue with differently sized data.
    require_complete: bool = False
    # Persist paper-style point-in-any-convex-component labels alongside each
    # geometry candidate instead of recomputing mesh signs every epoch.
    precompute_convex_union_labels: bool = False
    # Persist mutual signed point-to-mesh distances for every candidate and
    # sampled cloud. Pretraining can then apply its contact threshold without
    # repeating expensive mesh queries.
    precompute_mesh_sdf: bool = False
    # Paper-style UniCORN contact generation first brings the two meshes to
    # tangency, then perturbs the moving geometry by a small zero-mean SE(3)
    # Gaussian.  The paper does not publish these two standard deviations, so
    # experiments must record them explicitly.
    tangent_translation_noise_std: float = 0.002
    tangent_rotation_noise_std_rad: float = 0.01
    # Apply the same SE(3) Gaussian after anchor alignment in the rejection
    # sampler, then run floor and bidirectional penetration checks on the
    # perturbed pose. This keeps the final sample nonpenetrating.
    rejection_apply_tangent_gaussian: bool = False
    rotation_selection: str = ROTATION_SELECTION_MOST_DOWNWARD
    tool_source: str = TOOL_SOURCE_SELECTED_TOOLS
    object_tool_manifest: Optional[str] = None
    allow_self_object_tool_pairs: bool = False
    shard_count: int = 1
    shard_index: int = 0
    max_contacts_per_pair: int = 1
    penetration_eps: float = 5e-4
    # The legacy rejection sampler only queried tool surface points against
    # the object SDF. Bidirectional mode additionally rejects object surface
    # points lying inside the tool, which is required for a genuinely
    # non-penetrating geometry control.
    penetration_check_mode: str = PENETRATION_CHECK_TOOL_INTO_OBJECT
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

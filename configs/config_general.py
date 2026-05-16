"""General experiment defaults shared by the new automation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolMountCfg:
    """Tool mount scaling contract for generated robot assets.

    ``scale_xyz`` is anisotropic by schema even when current assets use a
    uniform scale.
    """

    scale_xyz: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    translate: list[float] = field(
        default_factory=lambda: [0.08799998, -4.9709342e-8, 0.926]
    )
    rot_wxyz: list[float] = field(
        default_factory=lambda: [-1.4551854e-11, 0.9238795, 0.38268346, -4.6566123e-10]
    )
    pose_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose_quat_wxyz: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    attach_link_name: str = "panda_link7"
    joint_name: str = "tool_weld_joint"
    local_pos0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.107])
    local_rot0_wxyz: list[float] = field(
        default_factory=lambda: [0.9238795, 0.0, 0.0, -0.38268346]
    )
    local_pos1: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    local_rot1_wxyz: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])


FrankaMountCfg = ToolMountCfg


@dataclass
class WandbCfg:
    enabled: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None
    group: Optional[str] = None
    mode: str = "disabled"
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    metadata_level: str = "summary"
    log_code: bool = False


@dataclass
class GeneralCfg:
    name: str = "default"
    seed: int = 0
    num_points: int = 512
    tools_selected_json: Optional[str] = None
    tools_manifest: Optional[str] = None
    objects_manifest: Optional[str] = None
    randomize_tool_assignment: bool = False
    randomize_object_assignment: bool = False
    deterministic: bool = True
    dtype: str = "float32"
    artifact_root: str = "/mnt/project/world_model/tool_generalist/artifacts"
    wandb: WandbCfg = field(default_factory=WandbCfg)
    tool_mount: ToolMountCfg = field(default_factory=ToolMountCfg)

    @property
    def paths_yaml(self) -> str:
        """Compatibility alias; ``ExpCfg.paths_yaml`` is canonical."""

        return "paths.yaml"

    @property
    def output_root(self) -> str:
        """Compatibility alias; ``artifact_root`` owns experiment outputs."""

        return self.artifact_root

    @output_root.setter
    def output_root(self, value: str) -> None:
        self.artifact_root = value

    @property
    def contact_schema_version(self) -> str:
        """Compatibility alias; ``ContactGenCfg.schema_version`` is canonical."""

        return "contact_pt_v1"

    @property
    def wandb_enabled(self) -> bool:
        return self.wandb.enabled

    @wandb_enabled.setter
    def wandb_enabled(self, value: bool) -> None:
        self.wandb.enabled = value

    @property
    def wandb_entity(self) -> Optional[str]:
        return self.wandb.entity

    @wandb_entity.setter
    def wandb_entity(self, value: Optional[str]) -> None:
        self.wandb.entity = value

    @property
    def wandb_mode(self) -> str:
        return self.wandb.mode

    @wandb_mode.setter
    def wandb_mode(self, value: str) -> None:
        self.wandb.mode = value

    @property
    def wandb_tags(self) -> list[str]:
        return self.wandb.tags

    @wandb_tags.setter
    def wandb_tags(self, value: list[str]) -> None:
        self.wandb.tags = value

    @property
    def wandb_notes(self) -> Optional[str]:
        return self.wandb.notes

    @wandb_notes.setter
    def wandb_notes(self, value: Optional[str]) -> None:
        self.wandb.notes = value

    @property
    def wandb_metadata_level(self) -> str:
        return self.wandb.metadata_level

    @wandb_metadata_level.setter
    def wandb_metadata_level(self, value: str) -> None:
        self.wandb.metadata_level = value

    @property
    def wandb_log_code(self) -> bool:
        return self.wandb.log_code

    @wandb_log_code.setter
    def wandb_log_code(self, value: bool) -> None:
        self.wandb.log_code = value

    @property
    def franka_mount(self) -> ToolMountCfg:
        """Compatibility alias while the project migrates to ``tool_mount``."""

        return self.tool_mount

    @franka_mount.setter
    def franka_mount(self, value: ToolMountCfg) -> None:
        self.tool_mount = value

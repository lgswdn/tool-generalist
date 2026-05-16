"""Post-contact rollout orchestration and contact_pt_env_v1 schema assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from configs.config_contact_gen import ContactGenCfg
from utils.contact.stabilize import (
    PhysicsBatchResult,
    PhysicsRunConfig,
    get_physics_runner,
)
from utils.io import hash_json, write_json

from .gen_contact import GeometryContactConfig
from .gen_contact import candidate_debug_path_for


def _torch():
    import torch

    return torch


def _forbidden_object_frame_keys() -> set[str]:
    suffix = "O"
    return {
        f"tool_translation_{suffix}",
        f"tool_rotation_{suffix}",
        f"contact_point_{suffix}",
        f"post_tool_delta_pose9d_{suffix}",
        f"post_object_delta_pose9d_{suffix}",
    }


@dataclass
class ContactPairConfig:
    object_mesh_path: str
    tool_mesh_path: str
    output_path: str
    tools_json_path: str
    object_id: str
    tool_id: str
    config_name: str = "contact_generation"
    config_hash: str = ""
    device: str = "cuda:0"
    seed: int = 0
    B: int = 2
    M: int = 3
    K: int = 512
    sdf_grid_res: int = 128
    sdf_padding: float = 0.05
    chunk_B: int = 1
    tool_scale: float = 0.1
    tool_scale_xyz: tuple[float, float, float] = (0.1, 0.1, 0.1)
    object_scale_range: tuple[float, float] = (0.1, 0.2)
    num_tool_surface_pts: int = 512
    upright_threshold: float = 0.0
    epsilon: float = 2e-3
    floor_eps: float = 0.0
    penetration_eps: float = 5e-4
    contact_mode_prob: Mapping[str, float] | float = None
    physics_runner: str = "isaac"
    t_stabilize: int = 120
    t_postcontact: int = 120
    object_mass_range: tuple[float, float] = (0.05, 0.5)
    tool_mass_range: tuple[float, float] = (0.05, 0.5)
    object_friction_range: tuple[float, float] = (0.4, 1.2)
    tool_friction_range: tuple[float, float] = (0.4, 1.2)
    ground_friction_range: tuple[float, float] = (0.4, 1.2)
    post_delta_seed: Optional[int] = None
    post_delta_translation_min: tuple[float, float, float] = (-0.02, -0.02, 0.0)
    post_delta_translation_max: tuple[float, float, float] = (0.02, 0.02, 0.04)
    post_delta_rotation_max_rad: float = 0.25
    post_tool_reach_translation_eps: float = 2e-3
    post_tool_reach_rotation_eps_rad: float = 5e-2
    post_object_table_z_min: float = 0.0
    env_spacing: float = 2.0
    unsigned_distance_accept_eps: float = 0.005
    visualization_enabled: bool = False
    visualization_stabilization_picture: bool = False
    visualization_stabilization_picture_num: int = 8
    visualization_postcontact_video: bool = False
    visualization_postcontact_video_num: int = 8
    visualization_video_dir: str = ""
    visualization_picture_dir: str = ""
    visualization_video_width: int = 640
    visualization_video_height: int = 480
    visualization_video_fps: int = 30
    visualization_camera_pos: tuple[float, float, float] = (0.24, 0.14, 0.18)
    visualization_camera_target: tuple[float, float, float] = (0.0, 0.0, 0.05)
    visualization_max_candidates: int = 1
    debug_dir: str = ""
    headless: bool = True
    close_after_run: bool = False

    @classmethod
    def from_contact_cfg(
        cls,
        *,
        contact_cfg: ContactGenCfg,
        object_mesh_path: str,
        tool_mesh_path: str,
        output_path: str,
        tools_json_path: str,
        object_id: str,
        tool_id: str,
        config_name: str,
        config_hash: str,
        device: str,
        seed: int,
        tool_scale_xyz: tuple[float, float, float],
        debug_dir: str = "",
    ) -> "ContactPairConfig":
        physics = contact_cfg.physics
        return cls(
            object_mesh_path=object_mesh_path,
            tool_mesh_path=tool_mesh_path,
            output_path=output_path,
            tools_json_path=tools_json_path,
            object_id=object_id,
            tool_id=tool_id,
            config_name=config_name,
            config_hash=config_hash,
            device=device,
            seed=seed,
            B=contact_cfg.B,
            M=contact_cfg.M,
            K=contact_cfg.num_surface_pts,
            sdf_grid_res=contact_cfg.sdf_grid_res,
            sdf_padding=contact_cfg.sdf_padding,
            chunk_B=contact_cfg.chunk_B,
            tool_scale=float(tool_scale_xyz[0]),
            tool_scale_xyz=tool_scale_xyz,
            object_scale_range=tuple(contact_cfg.object_scale_range),
            num_tool_surface_pts=contact_cfg.num_surface_pts,
            upright_threshold=contact_cfg.upright_threshold,
            epsilon=contact_cfg.epsilon,
            floor_eps=contact_cfg.floor_eps,
            contact_mode_prob=dict(contact_cfg.contact_mode_prob),
            physics_runner=physics.runner,
            t_stabilize=physics.t_stabilize,
            t_postcontact=physics.t_postcontact,
            object_mass_range=tuple(physics.object_mass_range),
            tool_mass_range=tuple(physics.tool_mass_range),
            object_friction_range=tuple(physics.object_friction_range),
            tool_friction_range=tuple(physics.tool_friction_range),
            ground_friction_range=tuple(physics.ground_friction_range),
            post_delta_seed=physics.post_delta_seed,
            post_delta_translation_min=tuple(physics.post_delta_translation_min),
            post_delta_translation_max=tuple(physics.post_delta_translation_max),
            post_delta_rotation_max_rad=physics.post_delta_rotation_max_rad,
            post_tool_reach_translation_eps=physics.post_tool_reach_translation_eps,
            post_tool_reach_rotation_eps_rad=physics.post_tool_reach_rotation_eps_rad,
            post_object_table_z_min=physics.post_object_table_z_min,
            unsigned_distance_accept_eps=float(physics.unsigned_distance_accept_eps),
            visualization_enabled=bool(contact_cfg.visualization.enabled),
            visualization_stabilization_picture=bool(
                contact_cfg.visualization.enabled and contact_cfg.visualization.stabilization_picture
            ),
            visualization_stabilization_picture_num=int(contact_cfg.visualization.stabilization_picture_num),
            visualization_postcontact_video=bool(contact_cfg.visualization.enabled and contact_cfg.visualization.postcontact_video),
            visualization_postcontact_video_num=int(contact_cfg.visualization.postcontact_video_num),
            visualization_video_dir=contact_cfg.visualization.video_dir or "",
            visualization_picture_dir=contact_cfg.visualization.picture_dir or "",
            visualization_video_width=int(contact_cfg.visualization.video_width),
            visualization_video_height=int(contact_cfg.visualization.video_height),
            visualization_video_fps=int(contact_cfg.visualization.video_fps),
            visualization_camera_pos=tuple(contact_cfg.visualization.camera_pos),
            visualization_camera_target=tuple(contact_cfg.visualization.camera_target),
            visualization_max_candidates=int(contact_cfg.visualization.max_visualized_candidates),
            debug_dir=debug_dir,
        )

    def geometry_config(self) -> GeometryContactConfig:
        contact_mode_prob = self.contact_mode_prob
        if isinstance(contact_mode_prob, (float, int)):
            contact_mode_prob = {"head": float(contact_mode_prob), "body": 1.0 - float(contact_mode_prob)}
        return GeometryContactConfig(
            object_mesh_path=self.object_mesh_path,
            tool_mesh_path=self.tool_mesh_path,
            tools_json_path=self.tools_json_path,
            object_id=self.object_id,
            tool_id=self.tool_id,
            config_name=self.config_name,
            config_hash=self.config_hash or hash_json({"config_name": self.config_name}),
            output_path=self.output_path,
            device=self.device,
            seed=self.seed,
            B=self.B,
            M=self.M,
            K=self.K,
            sdf_grid_res=self.sdf_grid_res,
            sdf_padding=self.sdf_padding,
            chunk_B=self.chunk_B,
            tool_scale_xyz=tuple(float(x) for x in self.tool_scale_xyz),
            object_scale_range=tuple(float(x) for x in self.object_scale_range),
            contact_mode_prob=dict(contact_mode_prob or {"head": 0.7, "body": 0.3}),
            upright_threshold=self.upright_threshold,
            epsilon=self.epsilon,
            floor_eps=self.floor_eps,
            penetration_eps=self.penetration_eps,
            visualization_enabled=bool(self.visualization_enabled),
        )

    def physics_config(self, object_scale: float) -> PhysicsRunConfig:
        return PhysicsRunConfig(
            t_stabilize=self.t_stabilize,
            t_postcontact=self.t_postcontact,
            runner=self.physics_runner,
            post_delta_seed=int(self.seed if self.post_delta_seed is None else self.post_delta_seed),
            post_delta_translation_min=self.post_delta_translation_min,
            post_delta_translation_max=self.post_delta_translation_max,
            post_delta_rotation_max_rad=self.post_delta_rotation_max_rad,
            post_tool_reach_translation_eps=self.post_tool_reach_translation_eps,
            post_tool_reach_rotation_eps_rad=self.post_tool_reach_rotation_eps_rad,
            post_object_table_z_min=self.post_object_table_z_min,
            unsigned_distance_accept_eps=float(self.unsigned_distance_accept_eps),
            env_spacing=float(self.env_spacing),
            visualization_enabled=bool(self.visualization_enabled),
            visualization_stabilization_picture=bool(self.visualization_stabilization_picture),
            visualization_stabilization_picture_num=int(self.visualization_stabilization_picture_num),
            visualization_postcontact_video=bool(self.visualization_postcontact_video),
            visualization_postcontact_video_num=int(self.visualization_postcontact_video_num),
            visualization_video_dir=self.visualization_video_dir or None,
            visualization_picture_dir=self.visualization_picture_dir or None,
            visualization_video_width=int(self.visualization_video_width),
            visualization_video_height=int(self.visualization_video_height),
            visualization_video_fps=int(self.visualization_video_fps),
            visualization_camera_pos=tuple(float(x) for x in self.visualization_camera_pos),
            visualization_camera_target=tuple(float(x) for x in self.visualization_camera_target),
            visualization_max_candidates=int(self.visualization_max_candidates),
            object_mesh_path=self.object_mesh_path,
            tool_mesh_path=self.tool_mesh_path,
            object_scale=object_scale,
            tool_scale_xyz=tuple(float(x) for x in self.tool_scale_xyz),
            debug_dir=self.debug_dir or None,
            headless=True if self.visualization_enabled else self.headless,
            close_after_run=self.close_after_run,
        )


def manifest_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def write_manifest(
    output_path: str | Path,
    *,
    status: str,
    physics_runner: str,
    num_candidates: int,
    num_contacts: int,
    debug_artifact_path: Optional[str | Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    manifest_path = manifest_path_for(output_path)
    payload: dict[str, Any] = {
        "schema_version": "contact_manifest_v1",
        "status": status,
        "physics_runner": physics_runner,
        "num_candidates": int(num_candidates),
        "num_contacts": int(num_contacts),
        "output_path": str(Path(output_path)),
    }
    if debug_artifact_path is not None:
        payload["debug_artifact_path"] = str(Path(debug_artifact_path))
    if extra:
        payload.update(extra)
    write_json(manifest_path, payload)
    return manifest_path


def candidate_debug_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".candidate.pt")


def physics_debug_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".physics_debug.pt")


def stabilized_success_artifact_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".stabilized_success.pt")


def stabilized_success_manifest_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".stabilized_success.manifest.json")


def _select(value: Any, mask: Any):
    return value[mask].detach().cpu()


def _tensor_payload(values: Mapping[str, Any], mask: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        tensor = value.detach().cpu()
        payload[key] = tensor[mask] if mask is not None else tensor
    return payload


def _select_optional_per_candidate(value: Any, mask: Any, expected_n: int) -> Any:
    torch = _torch()
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if int(tensor.shape[0]) == int(expected_n):
            return tensor[mask]
        return tensor
    if isinstance(value, list) and len(value) == int(expected_n):
        mask_list = mask.detach().cpu().bool().tolist()
        return [item for item, keep in zip(value, mask_list) if keep]
    return value


def _stabilized_physics_payload(physics: PhysicsBatchResult, mask: Any) -> dict[str, Any]:
    return {
        "stabilized_in_contact": _select(physics.stabilized_in_contact, mask),
        "stabilized_contact_count": _select(physics.stabilized_contact_count, mask),
        "stabilized_contact_impulse_norm": _select(physics.stabilized_contact_impulse_norm, mask),
        "stabilized_unsigned_distance_min": _select(physics.stabilized_unsigned_distance_min, mask),
        "stabilize_steps": _select(physics.stabilize_steps, mask),
        "status": physics.status,
        "runner": physics.runner,
        "is_real_physics": bool(physics.is_real_physics),
        "video_paths": [str(path) for path in physics.video_paths],
        "video_metadata": [dict(item) for item in physics.video_metadata],
        "visualization_timeline": [dict(item) for item in physics.visualization_timeline],
        "snapshot_paths": [str(path) for path in physics.snapshot_paths],
        "debug_paths": {str(key): str(value) for key, value in physics.debug_paths.items()},
    }


def save_stabilized_success_artifact(
    output_path: str | Path,
    *,
    candidate_payload: Mapping[str, Any],
    physical_props: Mapping[str, Any],
    physics: PhysicsBatchResult,
) -> Path:
    torch = _torch()
    output = stabilized_success_artifact_path_for(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    legacy_keys = _forbidden_object_frame_keys()
    candidate_keys = set(candidate_payload.get("candidates", {}).keys())
    physics_keys = set((physics.stabilized_candidates or {}).keys())
    found_legacy = sorted((candidate_keys | physics_keys) & legacy_keys)
    if found_legacy:
        raise RuntimeError(f"Legacy object-frame contact fields are forbidden: {found_legacy}")
    mask = physics.success_mask.detach().cpu().bool()
    success_count = int(mask.sum().item())
    original_n = int(physics.success_mask.numel())
    source_candidates = physics.stabilized_candidates
    if not source_candidates:
        raise RuntimeError("Stabilization did not return stabilized candidate poses; refusing to write success artifact.")
    payload = {
        "schema_version": "contact_stabilized_success_v1",
        "source_candidate_artifact_path": str(candidate_debug_path_for(output_path)),
        "status": "complete" if success_count > 0 and physics.is_real_physics else physics.status,
        "num_candidates": success_count,
        "num_source_candidates": original_n,
        "object_id": candidate_payload["object_id"],
        "tool_id": candidate_payload["tool_id"],
        "object_mesh_path": candidate_payload["object_mesh_path"],
        "tool_mesh_path": candidate_payload["tool_mesh_path"],
        "object_scale": float(candidate_payload["object_scale"]),
        "tool_scale_xyz": candidate_payload["tool_scale_xyz"],
        "tool_head_area_aabb_norm": candidate_payload["tool_head_area_aabb_norm"],
        "object_bbox_center_M": candidate_payload["object_bbox_center_M"],
        "object_bbox_extent_M": candidate_payload["object_bbox_extent_M"],
        "tool_bbox_center_M": candidate_payload["tool_bbox_center_M"],
        "tool_bbox_extent_M": candidate_payload["tool_bbox_extent_M"],
        "object_points_O": candidate_payload.get("object_points_O"),
        "tool_points_T": candidate_payload.get("tool_points_T"),
        "contact_normal_E": _select_optional_per_candidate(
            candidate_payload.get("contact_normal_E"), mask, original_n
        ),
        "object_point_sample_seed": int(candidate_payload.get("object_point_sample_seed", 0)),
        "tool_point_sample_seed": int(candidate_payload.get("tool_point_sample_seed", 0)),
        "config_name": candidate_payload.get("config_name", ""),
        "config_hash": candidate_payload.get("config_hash", ""),
        "candidate_artifact_path": candidate_payload.get("candidate_artifact_path", ""),
        "source_candidate_index": _select_optional_per_candidate(
            candidate_payload.get("source_candidate_index"), mask, original_n
        ),
        "debug_metrics": dict(candidate_payload.get("debug_metrics", {})),
        "candidates": _tensor_payload(source_candidates, mask),
        "physical_props": _tensor_payload(physical_props, mask),
        "physics": _stabilized_physics_payload(physics, mask),
    }
    torch.save(payload, output)
    write_json(
        stabilized_success_manifest_path_for(output_path),
        {
            "schema_version": "contact_stabilized_success_manifest_v1",
            "status": payload["status"],
            "stabilized_success_artifact_path": str(output),
            "source_candidate_artifact_path": str(candidate_debug_path_for(output_path)),
            "num_source_candidates": original_n,
            "num_stabilized": success_count,
            "physics_runner": physics.runner,
        },
    )
    return output


def load_stabilized_success_artifact(output_path: str | Path) -> dict[str, Any]:
    torch = _torch()
    path = stabilized_success_artifact_path_for(output_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Success-only stabilized contact artifact does not exist: {path}. "
            "Run the contact stabilization phase after geometry generation."
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != "contact_stabilized_success_v1":
        raise ValueError(f"Invalid success-only stabilized contact artifact: {path}")
    n = int(payload.get("num_candidates", 0))
    if n <= 0:
        raise ValueError(f"Success-only stabilized contact artifact is empty: {path}")
    candidates = payload.get("candidates")
    physical_props = payload.get("physical_props")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError(f"Success-only stabilized artifact has no candidate tensors: {path}")
    if not isinstance(physical_props, Mapping) or not physical_props:
        raise ValueError(f"Success-only stabilized artifact has no physical property tensors: {path}")
    legacy_keys = _forbidden_object_frame_keys()
    found_legacy = sorted((set(candidates.keys()) | set(payload.keys())) & legacy_keys)
    if found_legacy:
        raise ValueError(f"Success-only stabilized artifact contains forbidden legacy fields: {found_legacy}")
    return payload


def _physics_debug_extra(physics: PhysicsBatchResult) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "is_real_physics": bool(physics.is_real_physics),
        "stabilized_in_contact": physics.stabilized_in_contact.detach().cpu().tolist(),
        "stabilized_contact_count": physics.stabilized_contact_count.detach().cpu().tolist(),
        "stabilized_contact_impulse_norm": physics.stabilized_contact_impulse_norm.detach().cpu().tolist(),
        "stabilized_unsigned_distance_min": physics.stabilized_unsigned_distance_min.detach().cpu().tolist(),
    }
    if physics.video_paths:
        extra["video_paths"] = [str(path) for path in physics.video_paths]
    if physics.video_metadata:
        extra["video_metadata"] = [dict(item) for item in physics.video_metadata]
    if physics.visualization_timeline:
        extra["visualization_timeline"] = [dict(item) for item in physics.visualization_timeline]
    if physics.failure_reasons is not None:
        extra["failure_reasons"] = list(physics.failure_reasons)
    if physics.stage_usd_path:
        extra["stage_usd_path"] = physics.stage_usd_path
    if physics.debug_json_path:
        extra["debug_json_path"] = physics.debug_json_path
    if physics.snapshot_paths:
        extra["snapshot_paths"] = [str(path) for path in physics.snapshot_paths]
    if physics.debug_paths:
        extra["debug_paths"] = {str(key): str(value) for key, value in physics.debug_paths.items()}
    return extra


def assemble_contact_pt_env_v1(
    *,
    created_at: str | None = None,
    generator: str = "contact_generation.gen_postcontact",
    config_name: str = "contact_gen_default",
    config_hash: str | None = None,
    object_id: str,
    tool_id: str,
    object_mesh_path: str,
    tool_mesh_path: str,
    object_scale: float,
    tool_scale_xyz: Any,
    tool_head_area_aabb_norm: Any,
    object_bbox_center_M: Any,
    object_bbox_extent_M: Any,
    tool_bbox_center_M: Any,
    tool_bbox_extent_M: Any,
    object_point_sample_seed: int,
    tool_point_sample_seed: int,
    candidates: Mapping[str, Any],
    physical_props: Mapping[str, Any],
    physics: PhysicsBatchResult,
    object_points_O: Optional[Any] = None,
    tool_points_T: Optional[Any] = None,
    contact_normal_E: Optional[Any] = None,
    debug_metrics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    torch = _torch()
    mask = physics.success_mask.detach().cpu().bool()
    n = int(mask.sum().item())
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if config_hash is None:
        config_hash = hash_json(
            {
                "generator": generator,
                "config_name": config_name,
                "object_id": object_id,
                "tool_id": tool_id,
                "object_mesh_path": str(object_mesh_path),
                "tool_mesh_path": str(tool_mesh_path),
                "object_scale": float(object_scale),
                "tool_scale_xyz": torch.as_tensor(tool_scale_xyz, dtype=torch.float32).tolist(),
                "object_point_sample_seed": int(object_point_sample_seed),
                "tool_point_sample_seed": int(tool_point_sample_seed),
            }
        )

    generation_status = "complete" if physics.is_real_physics and n > 0 else physics.status
    payload: dict[str, Any] = {
        "schema_version": "contact_pt_env_v1",
        "created_at": created_at,
        "generator": generator,
        "config_name": config_name,
        "config_hash": config_hash,
        "generation_status": generation_status,
        "physics_runner": physics.runner,
        "object_id": object_id,
        "tool_id": tool_id,
        "object_mesh_path": str(Path(object_mesh_path).resolve()),
        "tool_mesh_path": str(Path(tool_mesh_path).resolve()),
        "object_scale": float(object_scale),
        "tool_scale_xyz": torch.as_tensor(tool_scale_xyz, dtype=torch.float32).cpu(),
        "tool_head_area_aabb_norm": torch.as_tensor(tool_head_area_aabb_norm, dtype=torch.float32).cpu(),
        "object_bbox_center_M": torch.as_tensor(object_bbox_center_M, dtype=torch.float32).cpu(),
        "object_bbox_extent_M": torch.as_tensor(object_bbox_extent_M, dtype=torch.float32).cpu(),
        "tool_bbox_center_M": torch.as_tensor(tool_bbox_center_M, dtype=torch.float32).cpu(),
        "tool_bbox_extent_M": torch.as_tensor(tool_bbox_extent_M, dtype=torch.float32).cpu(),
        "num_contacts": n,
        "object_point_sample_seed": int(object_point_sample_seed),
        "tool_point_sample_seed": int(tool_point_sample_seed),
    }

    pose_source = physics.stabilized_candidates
    if not pose_source:
        raise RuntimeError("Postcontact physics did not return candidate poses; refusing to assemble contact_pt_env_v1.")
    for key in (
        "object_rotation_E",
        "object_bbox_center_E",
        "tool_translation_E",
        "tool_rotation_E",
        "contact_point_E",
    ):
        payload[key] = _select(pose_source[key].detach().cpu(), mask)

    for key, value in physical_props.items():
        payload[key] = _select(value.detach().cpu(), mask)

    payload.update(
        {
            "is_real_physics": bool(physics.is_real_physics),
            "stabilized_in_contact": _select(physics.stabilized_in_contact, mask),
            "stabilized_contact_count": _select(physics.stabilized_contact_count, mask),
            "stabilized_contact_impulse_norm": _select(physics.stabilized_contact_impulse_norm, mask),
            "stabilize_steps": _select(physics.stabilize_steps, mask),
            "post_tool_delta_pose9d_E": _select(physics.post_tool_delta_pose9d_E, mask),
            "post_tool_achieved_delta_pose9d_E": _select(physics.post_tool_achieved_delta_pose9d_E, mask),
            "post_object_delta_pose9d_E": _select(physics.post_object_delta_pose9d_E, mask),
            "postcontact_steps": _select(physics.postcontact_steps, mask),
        }
    )

    if object_points_O is not None:
        payload["object_points_O"] = object_points_O.detach().cpu()
    if tool_points_T is not None:
        payload["tool_points_T"] = tool_points_T.detach().cpu()
    if contact_normal_E is not None:
        payload["contact_normal_E"] = _select(contact_normal_E.detach().cpu(), mask)
    if debug_metrics is not None:
        payload["debug_metrics"] = dict(debug_metrics)
    debug_extra = _physics_debug_extra(physics)
    if debug_extra:
        payload.setdefault("debug_metrics", {}).update(debug_extra)

    return payload


def save_final_or_debug(
    output_path: str | Path,
    payload: dict[str, Any],
    *,
    candidates: Mapping[str, Any],
    physics: PhysicsBatchResult,
    debug_extra: Optional[Mapping[str, Any]] = None,
    manifest_extra: Optional[Mapping[str, Any]] = None,
) -> int:
    torch = _torch()
    n_contacts = int(payload["num_contacts"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if n_contacts > 0:
        torch.save(payload, output)
        manifest_status = "complete" if physics.is_real_physics else payload["generation_status"]
        debug_path = None
        if n_contacts < int(physics.success_mask.numel()):
            failed = ~physics.success_mask.detach().cpu().bool()
            debug_path = physics_debug_path_for(output)
            torch.save(
                {
                    "schema_version": "contact_candidate_debug_v1",
                    "generation_status": "partial_failures",
                    "physics_runner": physics.runner,
                    "is_real_physics": bool(physics.is_real_physics),
                    "num_candidates": int(physics.success_mask.numel()),
                    "num_failed": int(failed.sum().item()),
                    "candidates": {key: value.detach().cpu()[failed] for key, value in candidates.items()},
                    **_physics_debug_extra(physics),
                },
                debug_path,
            )
        write_manifest(
            output,
            status=str(manifest_status),
            physics_runner=physics.runner,
            num_candidates=int(physics.success_mask.numel()),
            num_contacts=n_contacts,
            debug_artifact_path=debug_path,
            extra={**_physics_debug_extra(physics), **({} if manifest_extra is None else dict(manifest_extra))},
        )
        return n_contacts

    debug_path = physics_debug_path_for(output)
    torch.save(
        {
            "schema_version": "contact_candidate_debug_v1",
            "generation_status": physics.status,
            "physics_runner": physics.runner,
            "is_real_physics": bool(physics.is_real_physics),
            "num_candidates": int(physics.success_mask.numel()),
            "candidates": {key: value.detach().cpu() for key, value in candidates.items()},
            **_physics_debug_extra(physics),
            **({} if debug_extra is None else dict(debug_extra)),
        },
        debug_path,
    )
    write_manifest(
        output,
        status="incomplete",
        physics_runner=physics.runner,
        num_candidates=int(physics.success_mask.numel()),
        num_contacts=0,
        debug_artifact_path=debug_path,
        extra={**_physics_debug_extra(physics), **({} if manifest_extra is None else dict(manifest_extra))},
    )
    return 0


def run_contact_pair(cfg: ContactPairConfig, physics_runner: Any = None) -> int:
    stabilized_path = stabilized_success_artifact_path_for(cfg.output_path)
    _log(
        "[POSTCONTACT-START] "
        f"tool={cfg.tool_id} object={cfg.object_id} stabilized_success={stabilized_path}"
    )
    stabilized_payload = load_stabilized_success_artifact(cfg.output_path)
    candidates = stabilized_payload["candidates"]
    physical_props = stabilized_payload["physical_props"]
    n = int(stabilized_payload["num_candidates"])
    _log(
        "[POSTCONTACT-INPUT] "
        f"tool={cfg.tool_id} object={cfg.object_id} stabilized_success_candidates={n}"
    )
    _log(
        "[POSTCONTACT-PHYSICS] "
        f"runner={cfg.physics_runner} stabilize_steps=0 "
        f"postcontact_steps={cfg.t_postcontact} candidates={n}"
    )
    runner = physics_runner if physics_runner is not None else get_physics_runner(cfg.physics_runner)
    physics_cfg = replace(
        cfg.physics_config(float(stabilized_payload["object_scale"])),
        t_stabilize=0,
        run_postcontact=True,
        require_stabilized_contact=False,
        visualization_stabilization_picture=False,
    )
    physics = runner.run(
        candidates,
        physical_props,
        physics_cfg,
    )
    success_count = int(physics.success_mask.detach().cpu().bool().sum().item())
    _log(
        "[POSTCONTACT-DONE] "
        f"runner={physics.runner} status={physics.status} successes={success_count}/{n}"
    )
    _log("[SCHEMA] assembling contact_pt_env_v1 payload")
    payload = assemble_contact_pt_env_v1(
        generator="contact_generation.gen_postcontact",
        config_name=cfg.config_name,
        config_hash=str(stabilized_payload["config_hash"]),
        object_id=cfg.object_id,
        tool_id=cfg.tool_id,
        object_mesh_path=cfg.object_mesh_path,
        tool_mesh_path=cfg.tool_mesh_path,
        object_scale=float(stabilized_payload["object_scale"]),
        tool_scale_xyz=stabilized_payload["tool_scale_xyz"],
        tool_head_area_aabb_norm=stabilized_payload["tool_head_area_aabb_norm"],
        object_bbox_center_M=stabilized_payload["object_bbox_center_M"],
        object_bbox_extent_M=stabilized_payload["object_bbox_extent_M"],
        tool_bbox_center_M=stabilized_payload["tool_bbox_center_M"],
        tool_bbox_extent_M=stabilized_payload["tool_bbox_extent_M"],
        object_point_sample_seed=int(stabilized_payload.get("object_point_sample_seed", cfg.seed)),
        tool_point_sample_seed=int(stabilized_payload.get("tool_point_sample_seed", cfg.seed)),
        candidates=candidates,
        physical_props=physical_props,
        physics=physics,
        object_points_O=stabilized_payload.get("object_points_O"),
        tool_points_T=stabilized_payload.get("tool_points_T"),
        contact_normal_E=stabilized_payload.get("contact_normal_E"),
        debug_metrics={
            **dict(stabilized_payload.get("debug_metrics", {})),
            "source_candidate_index": stabilized_payload.get("source_candidate_index"),
            "stabilized_success_artifact_path": str(stabilized_path),
        },
    )
    _log(f"[SAVE] writing final/debug contact artifact output={cfg.output_path}")
    return save_final_or_debug(
        cfg.output_path,
        payload,
        candidates=candidates,
        physics=physics,
        debug_extra={
            "source_candidate_index": stabilized_payload.get("source_candidate_index"),
            "stabilized_success_artifact_path": str(stabilized_path),
        },
        manifest_extra={
            "schema_version": "contact_manifest_v1",
            "config_name": cfg.config_name,
            "config_hash": str(stabilized_payload["config_hash"]),
            "object_id": cfg.object_id,
            "tool_id": cfg.tool_id,
            "candidate_artifact_path": str(stabilized_payload.get("candidate_artifact_path", "")),
            "stabilized_success_artifact_path": str(stabilized_path),
            "tool_mesh_source": "adjusted_decomposed_mesh",
            "contact_schema_version": "contact_pt_env_v1",
        },
    )


def _log(message: str) -> None:
    print(f"[contact_generation.postcontact] {message}", flush=True)

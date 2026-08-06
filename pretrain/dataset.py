"""contact_pt_env_v1 dataset for geometry pretraining."""

from __future__ import annotations

import random
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset

from utils.contact.schema import (
    CONTACT_SCHEMA_VERSION,
    ContactSchemaError,
    load_and_validate_contact_pt,
)
from utils.geometry.mesh_io import load_mesh_vertices_faces, load_scaled_sampled_surface_points, scale_vertices
from utils.geometry.generated_gripper_kinematics import (
    CachedGripperPointKinematics,
    point_kinematics_from_candidate,
)
from utils.pretrain.noise_utils import build_precontact_trajectory
from utils.assets.tool_assets import load_tool_kinematic_cloud


NUM_TOOL_PTS = 512
NUM_OBJ_PTS = 512
CANDIDATE_SCHEMA_VERSION = "contact_candidate_v1"

def _as_float_tensor(value: Any, shape: tuple[int, ...] | None = None, key: str = "value") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if shape is not None and tuple(tensor.shape) != shape:
        raise ContactSchemaError(f"{key} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _cached_gripper_points(
    data: Mapping[str, Any],
    *,
    num_points: int,
) -> torch.Tensor | None:
    tool_id = str(data.get("tool_id", ""))
    if not tool_id.startswith(("generated_gripper_", "one_dof_gripper_")):
        return None
    if num_points != 512:
        raise ContactSchemaError(
            f"Canonical gripper clouds require num_points=512, got {num_points}"
        )
    mesh_path = Path(str(data["tool_mesh_path"])).expanduser().resolve()
    tools_json = mesh_path.parents[3] / "tools_adjusted.json"
    _, cloud = load_tool_kinematic_cloud(tools_json, tool_id)
    center = _as_float_tensor(
        data["tool_bbox_center_M"], (3,), "tool_bbox_center_M"
    )
    return (cloud - center).contiguous()


def _sample_surface_points(
    mesh_path: str | Path,
    *,
    scale: Any,
    bbox_center: torch.Tensor,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    """Sample mesh surface points, then subtract the explicit schema bbox center."""

    points = load_scaled_sampled_surface_points(
        mesh_path,
        scale=scale,
        num_points=num_points,
        seed=int(seed),
        process=False,
    )
    centered = torch.as_tensor(points, dtype=torch.float32) - bbox_center.to(dtype=torch.float32)
    return centered.contiguous()


def _load_centered_mesh_tensors(
    mesh_path: str | Path,
    *,
    scale: Any,
    bbox_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = load_mesh_vertices_faces(mesh_path, process=False)
    scaled = scale_vertices(vertices, scale)
    centered = torch.as_tensor(scaled, dtype=torch.float32) - bbox_center.to(dtype=torch.float32)
    return centered.contiguous(), torch.as_tensor(faces, dtype=torch.long).contiguous()


class NewPretrainDataset(Dataset):
    """Dataset indexed by individual contact cases in contact_pt_env_v1 files."""

    def __init__(
        self,
        pt_files: Iterable[str | Path],
        augment: bool = True,
        require_movement: bool = False,
        *,
        num_points: int = NUM_TOOL_PTS,
        num_precontact_steps: int = 4,
        allow_mock_physics: bool = False,
        noise_max_trans: float = 0.1,
        noise_max_rot_deg: float = 30.0,
        noise_max_retries: int = 10,
        floor_eps: float = 0.0,
        penetration_eps: float | None = None,
        validation_seed: int = 12345,
        denoise_target_mode: str = "one_step",
        tool_mesh_contract: str = "adjusted_decomposed_mesh",
        include_meshes: bool = True,
        max_contacts_per_file: int = 0,
        surface_jitter_std: float = 1e-3,
        kinematic_conditioning: bool = False,
        kinematic_delta_std: float = 0.15,
        use_saved_contact_clouds: bool = False,
    ):
        if int(num_precontact_steps) < 0:
            raise ValueError("num_precontact_steps must be non-negative")

        self.augment = bool(augment)
        self.require_movement = bool(require_movement)
        self.num_points = int(num_points)
        self.num_precontact_steps = int(num_precontact_steps)
        self.allow_mock_physics = bool(allow_mock_physics)
        self.noise_max_trans = float(noise_max_trans)
        self.noise_max_rot_deg = float(noise_max_rot_deg)
        self.noise_max_retries = int(noise_max_retries)
        self.floor_eps = float(floor_eps)
        self.penetration_eps = penetration_eps
        self.validation_seed = int(validation_seed)
        self.denoise_target_mode = str(denoise_target_mode)
        self.tool_mesh_contract = str(tool_mesh_contract)
        self.include_meshes = bool(include_meshes)
        self.max_contacts_per_file = int(max_contacts_per_file)
        self.surface_jitter_std = float(surface_jitter_std)
        self.kinematic_conditioning = bool(kinematic_conditioning)
        self.kinematic_delta_std = float(kinematic_delta_std)
        self.use_saved_contact_clouds = bool(use_saved_contact_clouds)
        if self.kinematic_conditioning and self.kinematic_delta_std <= 0.0:
            raise ValueError("kinematic_delta_std must be > 0")
        if self.max_contacts_per_file < 0:
            raise ValueError("max_contacts_per_file must be non-negative")

        self._index: list[tuple[str, int]] = []
        self._pt_cache: dict[str, Mapping[str, Any]] = {}
        self._cloud_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._mesh_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._kinematics_cache: dict[
            tuple[str, str], CachedGripperPointKinematics
        ] = {}
        self._path_kinematics: dict[
            str, CachedGripperPointKinematics
        ] = {}
        self._source_paths: list[str] = []

        for raw_path in pt_files:
            path = str(Path(raw_path))
            if path.endswith(".candidate.pt"):
                data = torch.load(path, map_location="cpu")
                if not isinstance(data, Mapping) or data.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
                    raise ContactSchemaError(f"Invalid geometry candidate artifact: {path}")
                if int(data.get("num_candidates", 0)) <= 0:
                    raise ContactSchemaError(f"Geometry candidate artifact has no candidates: {path}")
            else:
                data = load_and_validate_contact_pt(
                    path,
                    allow_mock=self.allow_mock_physics,
                    require_real_physics=False,
                    require_complete=True,
                    strict_mesh=not self.use_saved_contact_clouds,
                    tool_mesh_contract=self.tool_mesh_contract,
                )
            self._pt_cache[path] = data
            self._cloud_cache[path] = self._reconstruct_clouds(data)
            if self.kinematic_conditioning:
                if data.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
                    raise ContactSchemaError(
                        "Kinematic conditioning requires geometry candidate artifacts"
                    )
                tool_id = str(data["tool_id"])
                tool_points = self._cloud_cache[path]["tool_points_T"]
                point_hash = hashlib.sha256(
                    tool_points.contiguous().numpy().tobytes()
                ).hexdigest()
                cache_key = (tool_id, point_hash)
                kinematics = self._kinematics_cache.get(cache_key)
                if kinematics is None:
                    kinematics = point_kinematics_from_candidate(
                        data,
                        tool_points,
                    )
                    self._kinematics_cache[cache_key] = kinematics
                self._path_kinematics[path] = kinematics
            if self.include_meshes:
                self._mesh_cache[path] = self._reconstruct_meshes(data)
            self._source_paths.append(path)

            is_candidate = data.get("schema_version") == CANDIDATE_SCHEMA_VERSION
            movement_valid = None if is_candidate else data.get("movement_delta_valid")
            n = int(data["num_candidates"] if is_candidate else data["num_contacts"])
            if self.max_contacts_per_file > 0:
                # This is a cap, not a minimum. Older rejection-sampled files
                # contain variable candidate counts, so retain all cases from
                # shorter files rather than rejecting the entire file.
                n = min(n, self.max_contacts_per_file)
            for contact_i in range(n):
                if (
                    self.require_movement
                    and movement_valid is not None
                    and not bool(torch.as_tensor(movement_valid)[contact_i])
                ):
                    continue
                self._index.append((path, contact_i))

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(self._source_paths)

    @property
    def schema_version(self) -> str:
        versions = {str(data.get("schema_version", "")) for data in self._pt_cache.values()}
        return versions.pop() if len(versions) == 1 else "mixed_contact_schemas"

    def _reconstruct_clouds(self, data: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.use_saved_contact_clouds:
            if data.get("schema_version") != CONTACT_SCHEMA_VERSION:
                raise ContactSchemaError(
                    "Saved contact clouds require completed contact_pt_env_v1 files"
                )
            missing = [
                key
                for key in ("object_points_O", "tool_points_T")
                if key not in data
            ]
            if missing:
                raise ContactSchemaError(
                    f"Saved contact clouds are missing required fields: {missing}"
                )
            object_points = _as_float_tensor(
                data["object_points_O"], key="object_points_O"
            )
            tool_points = _as_float_tensor(
                data["tool_points_T"], key="tool_points_T"
            )
            expected = (self.num_points, 3)
            if tuple(object_points.shape) != expected or tuple(tool_points.shape) != expected:
                raise ContactSchemaError(
                    "Saved contact clouds must match configured num_points: "
                    f"object={tuple(object_points.shape)} "
                    f"tool={tuple(tool_points.shape)} expected={expected}"
                )
            if not torch.isfinite(object_points).all() or not torch.isfinite(tool_points).all():
                raise ContactSchemaError("Saved contact clouds contain non-finite values")
            return {
                "object_points_O": object_points.contiguous(),
                "tool_points_T": tool_points.contiguous(),
            }
        cached_gripper_points = _cached_gripper_points(
            data, num_points=self.num_points
        )
        if data.get("schema_version") == CANDIDATE_SCHEMA_VERSION:
            object_points = _as_float_tensor(data["object_points_O"], key="object_points_O")
            saved_tool_points = _as_float_tensor(
                data["tool_points_T"], key="tool_points_T"
            )
            tool_points = (
                saved_tool_points
                if cached_gripper_points is None
                else cached_gripper_points
            )
            if object_points.shape != (self.num_points, 3) or tool_points.shape != (self.num_points, 3):
                raise ContactSchemaError(
                    "Geometry candidate point clouds must match configured num_points: "
                    f"object={tuple(object_points.shape)} tool={tuple(tool_points.shape)} "
                    f"expected=({self.num_points}, 3)"
                )
            if (
                cached_gripper_points is not None
                and not torch.equal(saved_tool_points, cached_gripper_points)
            ):
                raise ContactSchemaError(
                    f"{data['tool_id']!r} candidate was not generated from "
                    "its canonical 128-bin cloud cache"
                )
            return {
                "object_points_O": object_points.contiguous(),
                "tool_points_T": tool_points.contiguous(),
            }
        object_center = _as_float_tensor(data["object_bbox_center_M"], (3,), "object_bbox_center_M")
        tool_center = _as_float_tensor(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M")
        object_points = _sample_surface_points(
            data["object_mesh_path"],
            scale=float(data["object_scale"]),
            bbox_center=object_center,
            num_points=self.num_points,
            seed=int(data["object_point_sample_seed"]),
        )
        tool_points = cached_gripper_points
        if tool_points is None:
            tool_points = _sample_surface_points(
                data["tool_mesh_path"],
                scale=_as_float_tensor(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
                bbox_center=tool_center,
                num_points=self.num_points,
                seed=int(data["tool_point_sample_seed"]),
            )
        return {
            "object_points_O": object_points,
            "tool_points_T": tool_points,
        }

    def _reconstruct_meshes(self, data: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        object_center = _as_float_tensor(data["object_bbox_center_M"], (3,), "object_bbox_center_M")
        tool_center = _as_float_tensor(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M")
        object_vertices, object_faces = _load_centered_mesh_tensors(
            data["object_mesh_path"],
            scale=float(data["object_scale"]),
            bbox_center=object_center,
        )
        tool_vertices, tool_faces = _load_centered_mesh_tensors(
            data["tool_mesh_path"],
            scale=_as_float_tensor(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
            bbox_center=tool_center,
        )
        return {
            "object_mesh_vertices": object_vertices,
            "object_mesh_faces": object_faces,
            "tool_mesh_vertices": tool_vertices,
            "tool_mesh_faces": tool_faces,
        }

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pt_path, contact_i = self._index[idx]
        data = self._pt_cache[pt_path]
        clouds = self._cloud_cache[pt_path]
        meshes = self._mesh_cache.get(pt_path)

        if data.get("schema_version") == CANDIDATE_SCHEMA_VERSION:
            return self._candidate_sample(
                pt_path, contact_i, data, clouds, meshes
            )

        object_points_O = clouds["object_points_O"]
        tool_points_T = clouds["tool_points_T"]
        if self.augment and self.surface_jitter_std > 0.0:
            object_points_O = (
                object_points_O
                + torch.randn_like(object_points_O) * self.surface_jitter_std
            )
            tool_points_T = (
                tool_points_T
                + torch.randn_like(tool_points_T) * self.surface_jitter_std
            )

        object_R_E = _as_float_tensor(data["object_rotation_E"][contact_i], (3, 3), "object_rotation_E")
        object_t_E = _as_float_tensor(data["object_bbox_center_E"][contact_i], (3,), "object_bbox_center_E")
        contact_tool_t_E = _as_float_tensor(data["tool_translation_E"][contact_i], (3,), "tool_translation_E")
        contact_tool_R_E = _as_float_tensor(data["tool_rotation_E"][contact_i], (3, 3), "tool_rotation_E")

        min_separation = self.penetration_eps
        if min_separation is None:
            min_separation = 0.0

        traj = build_precontact_trajectory(
            tool_points_T=tool_points_T,
            object_points_O=object_points_O,
            object_rotation_E=object_R_E,
            object_bbox_center_E=object_t_E,
            contact_tool_rotation_E=contact_tool_R_E,
            contact_tool_translation_E=contact_tool_t_E,
            num_precontact_steps=self.num_precontact_steps,
            noise_max_trans=self.noise_max_trans,
            noise_max_rot_deg=self.noise_max_rot_deg,
            max_retries=self.noise_max_retries,
            floor_eps=self.floor_eps,
            min_separation=float(min_separation),
            seed=self.validation_seed + int(idx),
            target_mode=self.denoise_target_mode,
        )

        physics = torch.stack(
            [
                torch.as_tensor(data["object_mass"][contact_i], dtype=torch.float32),
                torch.as_tensor(data["tool_mass"][contact_i], dtype=torch.float32),
                torch.as_tensor(data["object_friction"][contact_i], dtype=torch.float32),
                torch.as_tensor(data["tool_friction"][contact_i], dtype=torch.float32),
                torch.as_tensor(data["ground_friction"][contact_i], dtype=torch.float32),
            ]
        )

        sample = {
            "schema_version": CONTACT_SCHEMA_VERSION,
            "pt_path": pt_path,
            "contact_index": torch.tensor(contact_i, dtype=torch.long),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "tool_points_T": tool_points_T.float(),
            "object_points_O": object_points_O.float(),
            "tool_points_E_k": traj["tool_points_E_k"].float(),
            "object_points_E_k": traj["object_points_E_k"].float(),
            "rel_tool_object_t_k": traj["rel_tool_object_t_k"].float(),
            "tool_rotation_E_k": traj["tool_rotation_E_k"].float(),
            "tool_translation_E_k": traj["tool_translation_E_k"].float(),
            "object_rotation_E": object_R_E.float(),
            "object_bbox_center_E": object_t_E.float(),
            "contact_tool_rotation_E": contact_tool_R_E.float(),
            "contact_tool_translation_E": contact_tool_t_E.float(),
            "target_tool_denoise_pose9d_k": traj["target_tool_denoise_pose9d_k"].float(),
            "target_tool_denoise_mode": self.denoise_target_mode,
            "target_object_post_delta9d": _as_float_tensor(
                data["post_object_delta_pose9d_E"][contact_i], (9,), "post_object_delta_pose9d_E"
            ),
            "cond_tool_post_delta9d": _as_float_tensor(
                data["post_tool_delta_pose9d_E"][contact_i], (9,), "post_tool_delta_pose9d_E"
            ),
            "cond_object_post_delta9d": _as_float_tensor(
                data["post_object_delta_pose9d_E"][contact_i], (9,), "post_object_delta_pose9d_E"
            ),
            "physics": physics.float(),
        }
        if meshes is not None:
            sample.update(meshes)
        if data.get("tool_point_inside_object") is not None:
            sample["tool_point_inside_object"] = torch.as_tensor(
                data["tool_point_inside_object"][contact_i], dtype=torch.bool
            )
        if data.get("object_point_inside_tool") is not None:
            sample["object_point_inside_tool"] = torch.as_tensor(
                data["object_point_inside_tool"][contact_i], dtype=torch.bool
            )
        if data.get("tool_point_object_signed_sdf") is not None:
            sample["tool_point_object_signed_sdf"] = torch.as_tensor(
                data["tool_point_object_signed_sdf"][contact_i],
                dtype=torch.float32,
            )
        if data.get("object_point_tool_signed_sdf") is not None:
            sample["object_point_tool_signed_sdf"] = torch.as_tensor(
                data["object_point_tool_signed_sdf"][contact_i],
                dtype=torch.float32,
            )
        return sample

    def _candidate_sample(
        self,
        pt_path: str,
        contact_i: int,
        data: Mapping[str, Any],
        clouds: Mapping[str, torch.Tensor],
        meshes: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        """Materialize a geometry-only candidate for the contact objective."""

        candidates = data["candidates"]
        object_points_O = clouds["object_points_O"]
        tool_points_T = clouds["tool_points_T"]
        openness_delta = None
        kinematic_tool_clouds = None
        if self.kinematic_conditioning:
            kinematics = self._path_kinematics[pt_path]
            openness_delta = self._sample_openness_delta(
                kinematics.opening_fraction,
                pt_path=pt_path,
                contact_i=contact_i,
            )
            kinematic_tool_clouds = kinematics.static_state_clouds(tool_points_T)
            tool_points_T = kinematics.cloud_at_fraction(
                tool_points_T,
                kinematics.opening_fraction + openness_delta,
                canonical_local=False,
            )
        if self.augment and self.surface_jitter_std > 0.0:
            object_points_O = (
                object_points_O
                + torch.randn_like(object_points_O) * self.surface_jitter_std
            )
            tool_points_T = (
                tool_points_T
                + torch.randn_like(tool_points_T) * self.surface_jitter_std
            )

        object_R_E = _as_float_tensor(
            candidates["object_rotation_E"][contact_i], (3, 3), "object_rotation_E"
        )
        object_t_E = _as_float_tensor(
            candidates["object_bbox_center_E"][contact_i], (3,), "object_bbox_center_E"
        )
        tool_R_E = _as_float_tensor(
            candidates["tool_rotation_E"][contact_i], (3, 3), "tool_rotation_E"
        )
        tool_t_E = _as_float_tensor(
            candidates["tool_translation_E"][contact_i], (3,), "tool_translation_E"
        )
        # Match build_precontact_trajectory's representation: point tensors
        # carry rotation but remain bbox-centered; translation is supplied
        # separately through rel_tool_object_t_k and the pose tensors.
        object_points_E = torch.einsum("ij,nj->ni", object_R_E, object_points_O)
        tool_points_E = torch.einsum("ij,nj->ni", tool_R_E, tool_points_T)
        zeros9 = torch.zeros(9, dtype=torch.float32)
        sample = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "pt_path": pt_path,
            "contact_index": torch.tensor(contact_i, dtype=torch.long),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "tool_points_T": tool_points_T.float(),
            "object_points_O": object_points_O.float(),
            "tool_points_E_k": tool_points_E.unsqueeze(0).float(),
            "object_points_E_k": object_points_E.unsqueeze(0).float(),
            "rel_tool_object_t_k": (tool_t_E - object_t_E).unsqueeze(0).float(),
            "tool_rotation_E_k": tool_R_E.unsqueeze(0).float(),
            "tool_translation_E_k": tool_t_E.unsqueeze(0).float(),
            "object_rotation_E": object_R_E.float(),
            "object_bbox_center_E": object_t_E.float(),
            "contact_tool_rotation_E": tool_R_E.float(),
            "contact_tool_translation_E": tool_t_E.float(),
            # These compatibility tensors are ignored when only the contact
            # head is enabled; no physical/post-contact labels are fabricated.
            "target_tool_denoise_pose9d_k": torch.zeros(0, 9, dtype=torch.float32),
            "target_object_post_delta9d": zeros9,
            "cond_tool_post_delta9d": zeros9,
            "cond_object_post_delta9d": zeros9,
            "physics": torch.zeros(5, dtype=torch.float32),
        }
        if self.kinematic_conditioning:
            sample["kinematic_tool_clouds"] = kinematic_tool_clouds.float()
            sample["openness_delta"] = torch.tensor(
                openness_delta, dtype=torch.float32
            )
        if meshes is not None:
            sample.update(meshes)
        if data.get("tool_point_inside_object") is not None:
            sample["tool_point_inside_object"] = torch.as_tensor(
                data["tool_point_inside_object"][contact_i], dtype=torch.bool
            )
        if data.get("object_point_inside_tool") is not None:
            sample["object_point_inside_tool"] = torch.as_tensor(
                data["object_point_inside_tool"][contact_i], dtype=torch.bool
            )
        if data.get("tool_point_object_signed_sdf") is not None:
            sample["tool_point_object_signed_sdf"] = torch.as_tensor(
                data["tool_point_object_signed_sdf"][contact_i],
                dtype=torch.float32,
            )
        if data.get("object_point_tool_signed_sdf") is not None:
            sample["object_point_tool_signed_sdf"] = torch.as_tensor(
                data["object_point_tool_signed_sdf"][contact_i],
                dtype=torch.float32,
            )
        return sample

    def _sample_openness_delta(
        self,
        opening_fraction: float,
        *,
        pt_path: str,
        contact_i: int,
    ) -> float:
        if self.augment:
            gaussian = random.gauss
        else:
            digest = hashlib.sha256(
                f"{self.validation_seed}:{pt_path}:{contact_i}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "little"))
            gaussian = rng.gauss
        while True:
            delta = float(gaussian(0.0, self.kinematic_delta_std))
            if 0.0 <= opening_fraction + delta <= 1.0:
                return delta


def collect_pt_files(
    data_dir: str | Path, *, use_geometry_candidates: bool = False
) -> list[str]:
    """Recursively find contact .pt files under ``data_dir``."""

    blocked_suffixes = (".candidate.pt", ".physics_debug.pt", ".stabilized_success.pt", ".stabilized.pt")
    if use_geometry_candidates:
        candidates = sorted(str(p) for p in Path(data_dir).rglob("*.candidate.pt"))
        if not candidates:
            raise RuntimeError(f"No geometry candidate .pt files found under {data_dir}")
        return candidates
    complete = sorted(
        str(p)
        for p in Path(data_dir).rglob("*.pt")
        if not any(str(p).endswith(suffix) for suffix in blocked_suffixes)
    )
    if complete:
        return complete
    # Geometry-only contact experiments intentionally stop at candidate files.
    return sorted(str(p) for p in Path(data_dir).rglob("*.candidate.pt"))


def make_split(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    max_files: int = 0,
    require_movement: bool = False,
    *,
    num_points: int = NUM_TOOL_PTS,
    num_precontact_steps: int = 4,
    allow_mock_physics: bool = False,
    noise_max_trans: float = 0.1,
    noise_max_rot_deg: float = 30.0,
    noise_max_retries: int = 10,
    floor_eps: float = 0.0,
    validation_seed: int = 12345,
    denoise_target_mode: str = "one_step",
    tool_mesh_contract: str = "adjusted_decomposed_mesh",
    include_meshes: bool = True,
    use_geometry_candidates: bool = False,
    max_contacts_per_file: int = 0,
    surface_jitter_std: float = 1e-3,
    kinematic_conditioning: bool = False,
    kinematic_delta_std: float = 0.15,
    use_saved_contact_clouds: bool = False,
) -> tuple[NewPretrainDataset, NewPretrainDataset]:
    """Return ``(train_dataset, val_dataset)`` split by file."""

    files = collect_pt_files(
        data_dir, use_geometry_candidates=use_geometry_candidates
    )
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files > 0:
        files = files[:max_files]
    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:] or val_files

    common = dict(
        require_movement=require_movement,
        num_points=num_points,
        num_precontact_steps=num_precontact_steps,
        allow_mock_physics=allow_mock_physics,
        noise_max_trans=noise_max_trans,
        noise_max_rot_deg=noise_max_rot_deg,
        noise_max_retries=noise_max_retries,
        floor_eps=floor_eps,
        validation_seed=validation_seed,
        denoise_target_mode=denoise_target_mode,
        tool_mesh_contract=tool_mesh_contract,
        include_meshes=include_meshes,
        max_contacts_per_file=max_contacts_per_file,
        surface_jitter_std=surface_jitter_std,
        kinematic_conditioning=kinematic_conditioning,
        kinematic_delta_std=kinematic_delta_std,
        use_saved_contact_clouds=use_saved_contact_clouds,
    )
    return (
        NewPretrainDataset(train_files, augment=augment, **common),
        NewPretrainDataset(val_files, augment=False, **common),
    )


PretrainDataset = NewPretrainDataset

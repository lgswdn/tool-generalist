"""contact_pt_env_v1 dataset for geometry pretraining.

The loader treats the v1 schema as the only source of truth.  Mesh paths,
scales, explicit mesh bbox centers, per-contact poses, and point-sampling seeds
drive reconstruction; legacy point-cloud and pose fields are ignored.
"""

from __future__ import annotations

import random
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
from utils.pretrain.noise_utils import build_precontact_trajectory


NUM_TOOL_PTS = 512
NUM_OBJ_PTS = 512

def _as_float_tensor(value: Any, shape: tuple[int, ...] | None = None, key: str = "value") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if shape is not None and tuple(tensor.shape) != shape:
        raise ContactSchemaError(f"{key} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


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

        self._index: list[tuple[str, int]] = []
        self._pt_cache: dict[str, Mapping[str, Any]] = {}
        self._cloud_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._mesh_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._source_paths: list[str] = []

        for raw_path in pt_files:
            path = str(Path(raw_path))
            data = load_and_validate_contact_pt(
                path,
                allow_mock=self.allow_mock_physics,
                require_real_physics=False,
                require_complete=True,
            )
            self._pt_cache[path] = data
            self._cloud_cache[path] = self._reconstruct_clouds(data)
            self._mesh_cache[path] = self._reconstruct_meshes(data)
            self._source_paths.append(path)

            movement_valid = data.get("movement_delta_valid")
            n = int(data["num_contacts"])
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
        return CONTACT_SCHEMA_VERSION

    def _reconstruct_clouds(self, data: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        object_center = _as_float_tensor(data["object_bbox_center_M"], (3,), "object_bbox_center_M")
        tool_center = _as_float_tensor(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M")
        object_points = _sample_surface_points(
            data["object_mesh_path"],
            scale=float(data["object_scale"]),
            bbox_center=object_center,
            num_points=self.num_points,
            seed=int(data["object_point_sample_seed"]),
        )
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
        meshes = self._mesh_cache[pt_path]

        object_points_O = clouds["object_points_O"]
        tool_points_T = clouds["tool_points_T"]
        if self.augment:
            object_points_O = object_points_O + torch.randn_like(object_points_O) * 1e-3
            tool_points_T = tool_points_T + torch.randn_like(tool_points_T) * 1e-3

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

        return {
            "schema_version": CONTACT_SCHEMA_VERSION,
            "pt_path": pt_path,
            "contact_index": torch.tensor(contact_i, dtype=torch.long),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "tool_points_T": tool_points_T.float(),
            "object_points_O": object_points_O.float(),
            "object_mesh_vertices": meshes["object_mesh_vertices"],
            "object_mesh_faces": meshes["object_mesh_faces"],
            "tool_mesh_vertices": meshes["tool_mesh_vertices"],
            "tool_mesh_faces": meshes["tool_mesh_faces"],
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


def collect_pt_files(data_dir: str | Path) -> list[str]:
    """Recursively find contact .pt files under ``data_dir``."""

    blocked_suffixes = (".candidate.pt", ".physics_debug.pt", ".stabilized_success.pt", ".stabilized.pt")
    return sorted(
        str(p)
        for p in Path(data_dir).rglob("*.pt")
        if not any(str(p).endswith(suffix) for suffix in blocked_suffixes)
    )


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
) -> tuple[NewPretrainDataset, NewPretrainDataset]:
    """Return ``(train_dataset, val_dataset)`` split by file."""

    files = collect_pt_files(data_dir)
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
    )
    return (
        NewPretrainDataset(train_files, augment=augment, **common),
        NewPretrainDataset(val_files, augment=False, **common),
    )


PretrainDataset = NewPretrainDataset

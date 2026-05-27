"""UniCORN contact-patch dataset built from contact_pt_env_v1 artifacts."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset

from pretrain.dataset import collect_pt_files
from utils.contact.schema import CONTACT_SCHEMA_VERSION, ContactSchemaError, load_and_validate_contact_pt
from utils.geometry.mesh_io import load_mesh_vertices_faces, load_scaled_sampled_surface_points, scale_vertices


def _as_float_tensor(value: Any, shape: tuple[int, ...] | None = None, key: str = "value") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if shape is not None and tuple(tensor.shape) != shape:
        raise ContactSchemaError(f"{key} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _sample_centered_surface_points(
    mesh_path: str | Path,
    *,
    scale: Any,
    bbox_center: torch.Tensor,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    points = load_scaled_sampled_surface_points(
        mesh_path,
        scale=scale,
        num_points=num_points,
        seed=int(seed),
        process=False,
    )
    return (torch.as_tensor(points, dtype=torch.float32) - bbox_center).contiguous()


def _load_centered_mesh(
    mesh_path: str | Path,
    *,
    scale: Any,
    bbox_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = load_mesh_vertices_faces(mesh_path, process=False)
    vertices = scale_vertices(vertices, scale)
    vertices = torch.as_tensor(vertices, dtype=torch.float32) - bbox_center
    return vertices.contiguous(), torch.as_tensor(faces, dtype=torch.long).contiguous()


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    angles = (torch.rand(3, dtype=dtype) * 2.0 - 1.0) * math.pi
    sx, sy, sz = torch.sin(angles)
    cx, cy, cz = torch.cos(angles)
    one = torch.ones((), dtype=dtype)
    zero = torch.zeros((), dtype=dtype)
    rx = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, cx, -sx)),
            torch.stack((zero, sx, cx)),
        )
    )
    ry = torch.stack(
        (
            torch.stack((cy, zero, sy)),
            torch.stack((zero, one, zero)),
            torch.stack((-sy, zero, cy)),
        )
    )
    rz = torch.stack(
        (
            torch.stack((cz, -sz, zero)),
            torch.stack((sz, cz, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    return rz @ ry @ rx


def _augment_pair(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    *,
    translation_range: tuple[float, float],
    log_scale_range: tuple[float, float],
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = points_a.dtype
    rot = _random_rotation(dtype)
    log_lo, log_hi = float(log_scale_range[0]), float(log_scale_range[1])
    scale = torch.exp(torch.empty((), dtype=dtype).uniform_(log_lo, log_hi))
    trans_lo, trans_hi = float(translation_range[0]), float(translation_range[1])
    trans = torch.empty(3, dtype=dtype).uniform_(trans_lo, trans_hi)
    out_a = scale * (points_a @ rot.T) + trans
    out_b = scale * (points_b @ rot.T) + trans
    if float(noise_std) > 0:
        out_a = out_a + torch.randn_like(out_a) * float(noise_std)
        out_b = out_b + torch.randn_like(out_b) * float(noise_std)
    return out_a.contiguous(), out_b.contiguous()


class UnicornContactPairDataset(Dataset):
    """Contact-patch samples derived from existing in-contact cases."""

    def __init__(
        self,
        pt_files: Iterable[str | Path],
        *,
        num_points: int = 512,
        augment: bool = True,
        allow_mock_physics: bool = False,
        contact_eps: float = 0.002,
        label_backend: str = "kaolin",
        label_chunk_size: int = 8192,
        translation_range: tuple[float, float] = (-0.1, 0.1),
        log_scale_range: tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.01,
    ):
        self.num_points = int(num_points)
        self.augment = bool(augment)
        self.allow_mock_physics = bool(allow_mock_physics)
        self.contact_eps = float(contact_eps)
        self.label_backend = str(label_backend)
        self.label_chunk_size = int(label_chunk_size)
        self.translation_range = tuple(float(v) for v in translation_range)
        self.log_scale_range = tuple(float(v) for v in log_scale_range)
        self.noise_std = float(noise_std)

        self._index: list[tuple[str, int]] = []
        self._pt_cache: dict[str, Mapping[str, Any]] = {}
        self._geom_cache: dict[str, dict[str, torch.Tensor]] = {}
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
            self._geom_cache[path] = self._reconstruct_geometry(data)
            self._source_paths.append(path)
            for contact_i in range(int(data["num_contacts"])):
                self._index.append((path, contact_i))

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(self._source_paths)

    @property
    def schema_version(self) -> str:
        return CONTACT_SCHEMA_VERSION

    def _reconstruct_geometry(self, data: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        object_center = _as_float_tensor(data["object_bbox_center_M"], (3,), "object_bbox_center_M")
        tool_center = _as_float_tensor(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M")
        object_points = _sample_centered_surface_points(
            data["object_mesh_path"],
            scale=float(data["object_scale"]),
            bbox_center=object_center,
            num_points=self.num_points,
            seed=int(data["object_point_sample_seed"]),
        )
        tool_points = _sample_centered_surface_points(
            data["tool_mesh_path"],
            scale=_as_float_tensor(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
            bbox_center=tool_center,
            num_points=self.num_points,
            seed=int(data["tool_point_sample_seed"]),
        )
        object_vertices, object_faces = _load_centered_mesh(
            data["object_mesh_path"],
            scale=float(data["object_scale"]),
            bbox_center=object_center,
        )
        tool_vertices, tool_faces = _load_centered_mesh(
            data["tool_mesh_path"],
            scale=_as_float_tensor(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
            bbox_center=tool_center,
        )
        return {
            "object_points_O": object_points,
            "tool_points_T": tool_points,
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
        geom = self._geom_cache[pt_path]

        object_R_E = _as_float_tensor(data["object_rotation_E"][contact_i], (3, 3), "object_rotation_E")
        object_t_E = _as_float_tensor(data["object_bbox_center_E"][contact_i], (3,), "object_bbox_center_E")
        tool_R_E = _as_float_tensor(data["tool_rotation_E"][contact_i], (3, 3), "tool_rotation_E")
        tool_t_E = _as_float_tensor(data["tool_translation_E"][contact_i], (3,), "tool_translation_E")

        points_a = geom["tool_points_T"] @ tool_R_E.T + tool_t_E
        points_b = geom["object_points_O"] @ object_R_E.T + object_t_E
        label_points_a = points_a
        label_points_b = points_b

        if self.augment:
            points_a, points_b = _augment_pair(
                points_a,
                points_b,
                translation_range=self.translation_range,
                log_scale_range=self.log_scale_range,
                noise_std=self.noise_std,
            )

        return {
            "schema_version": CONTACT_SCHEMA_VERSION,
            "pt_path": pt_path,
            "contact_index": torch.tensor(contact_i, dtype=torch.long),
            "object_id": str(data["object_id"]),
            "tool_id": str(data["tool_id"]),
            "points_A": points_a.float(),
            "points_B": points_b.float(),
            "label_points_A_E": label_points_a.float(),
            "label_points_B_E": label_points_b.float(),
            "object_mesh_vertices": geom["object_mesh_vertices"],
            "object_mesh_faces": geom["object_mesh_faces"],
            "tool_mesh_vertices": geom["tool_mesh_vertices"],
            "tool_mesh_faces": geom["tool_mesh_faces"],
            "object_rotation_E": object_R_E.float(),
            "object_bbox_center_E": object_t_E.float(),
            "tool_rotation_E_k": tool_R_E.reshape(1, 3, 3).float(),
            "tool_translation_E_k": tool_t_E.reshape(1, 3).float(),
            "label_backend": self.label_backend,
        }


def make_unicorn_split(
    data_dir: str | Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    max_files: int = 0,
    num_points: int = 512,
    allow_mock_physics: bool = False,
    contact_eps: float = 0.002,
    label_backend: str = "kaolin",
    label_chunk_size: int = 8192,
    translation_range: tuple[float, float] = (-0.1, 0.1),
    log_scale_range: tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.01,
) -> tuple[UnicornContactPairDataset, UnicornContactPairDataset]:
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
        num_points=num_points,
        allow_mock_physics=allow_mock_physics,
        contact_eps=contact_eps,
        label_backend=label_backend,
        label_chunk_size=label_chunk_size,
        translation_range=translation_range,
        log_scale_range=log_scale_range,
        noise_std=noise_std,
    )
    return (
        UnicornContactPairDataset(train_files, augment=augment, **common),
        UnicornContactPairDataset(val_files, augment=False, **common),
    )

"""Mesh loading and surface sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .bbox import bbox_center_mesh


def load_mesh_vertices_faces(path: str | Path, process: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load a mesh file and return ``(vertices, faces)`` as numpy arrays."""

    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=process)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh {path} did not load vertices with shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Mesh {path} did not load triangular faces")
    return vertices, faces


def scale_vertices(vertices: np.ndarray, scale: float | Sequence[float]) -> np.ndarray:
    scale_arr = np.asarray(scale, dtype=np.float64)
    if scale_arr.ndim == 0:
        return vertices * float(scale_arr)
    if scale_arr.shape == (3,):
        return vertices * scale_arr.reshape(1, 3)
    raise ValueError(f"Scale must be scalar or shape (3,), got {scale_arr.shape}")


def sample_surface_points_numpy(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Area-weighted surface sampling with vertex fallback for degenerate meshes."""

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    rng = np.random.default_rng(seed)
    if faces.shape[0] == 0:
        choice = rng.integers(0, vertices.shape[0], size=num_points)
        return vertices[choice]

    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        choice = rng.integers(0, vertices.shape[0], size=num_points)
        return vertices[choice]

    face_idx = rng.choice(faces.shape[0], size=num_points, replace=True, p=areas / areas.sum())
    selected = tri[face_idx]
    u = rng.random(num_points)
    v = rng.random(num_points)
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    return (
        selected[:, 0] * w0[:, None]
        + selected[:, 1] * w1[:, None]
        + selected[:, 2] * w2[:, None]
    )


def load_scaled_sampled_surface_points(
    path: str | Path,
    *,
    scale: float | Sequence[float],
    num_points: int,
    seed: int | None = None,
    process: bool = False,
) -> np.ndarray:
    vertices, faces = load_mesh_vertices_faces(path, process=process)
    return sample_surface_points_numpy(
        scale_vertices(vertices, scale),
        faces,
        num_points,
        seed=seed,
    )


def load_mesh_tensors(path: str | Path, device: str, process: bool = False):
    """Load a mesh file as torch tensors. Imports torch/trimesh only when called."""

    import torch

    vertices, faces = load_mesh_vertices_faces(path, process=process)
    return (
        torch.as_tensor(vertices, dtype=torch.float32, device=device),
        torch.as_tensor(faces, dtype=torch.int64, device=device),
    )


def sample_surface_points_torch(verts, faces, num_points: int):
    """Area-weighted torch surface sampling."""

    import torch

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    device = verts.device
    if faces.shape[0] == 0:
        return verts[torch.randint(verts.shape[0], (num_points,), device=device)]
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    if not torch.isfinite(areas).all() or float(areas.sum().detach().cpu()) <= 0.0:
        return verts[torch.randint(verts.shape[0], (num_points,), device=device)]
    probs = areas / areas.sum()
    face_idx = torch.multinomial(probs, num_points, replacement=True)
    r1 = torch.sqrt(torch.rand(num_points, device=device))
    r2 = torch.rand(num_points, device=device)
    return (
        (1 - r1).unsqueeze(-1) * v0[face_idx]
        + (r1 * (1 - r2)).unsqueeze(-1) * v1[face_idx]
        + (r1 * r2).unsqueeze(-1) * v2[face_idx]
    )


def scaled_mesh_bbox(
    path: str | Path,
    scale: float | Sequence[float],
    process: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bbox ``(center, extent)`` for a mesh after scale is applied."""

    vertices, _ = load_mesh_vertices_faces(path, process=process)
    scaled = scale_vertices(vertices, scale)
    center, extent = bbox_center_mesh(scaled)
    return np.asarray(center, dtype=np.float64), np.asarray(extent, dtype=np.float64)

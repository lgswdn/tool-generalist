"""Axis-aligned bounding-box helpers."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:  # torch is an existing project dependency, but keep import-time failure lazy.
    import torch
except Exception:  # pragma: no cover - exercised only in torch-less environments.
    torch = None  # type: ignore[assignment]


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _points_from_mesh(mesh_or_points: Any) -> Any:
    if hasattr(mesh_or_points, "vertices"):
        return mesh_or_points.vertices
    if isinstance(mesh_or_points, (tuple, list)) and mesh_or_points:
        first = mesh_or_points[0]
        if hasattr(first, "shape"):
            return first
    return mesh_or_points


def _validate_points(points: Any) -> None:
    if len(points.shape) != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {tuple(points.shape)}")
    if points.shape[0] == 0:
        raise ValueError("Cannot compute a bbox for an empty point set")


def bbox_center_mesh(mesh_or_points: Any) -> Tuple[Any, Any]:
    """Return ``(center, extent)`` for the mesh/point-cloud AABB.

    The center is always exactly ``(bbox_min + bbox_max) / 2`` and the extent is
    always ``bbox_max - bbox_min``.  The input can be a Trimesh-like object with
    ``.vertices``, a points array, a torch tensor, or ``(vertices, faces)``.
    """

    points = _points_from_mesh(mesh_or_points)
    _validate_points(points)

    if _is_torch_tensor(points):
        bbox_min = points.min(dim=0).values
        bbox_max = points.max(dim=0).values
    else:
        pts = np.asarray(points)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)

    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    return center, extent


def _validate_bbox_metadata(points: Any, center: Any, extent: Any) -> Tuple[Any, Any]:
    if _is_torch_tensor(points):
        center_value = torch.as_tensor(center, dtype=points.dtype, device=points.device)
        extent_value = torch.as_tensor(extent, dtype=points.dtype, device=points.device)
        center_shape = tuple(center_value.shape)
        extent_shape = tuple(extent_value.shape)
    else:
        pts = np.asarray(points)
        center_value = np.asarray(center, dtype=pts.dtype)
        extent_value = np.asarray(extent, dtype=pts.dtype)
        center_shape = center_value.shape
        extent_shape = extent_value.shape

    if center_shape != (3,):
        raise ValueError(f"bbox_center must have shape (3,), got {center_shape}")
    if extent_shape != (3,):
        raise ValueError(f"bbox_extent must have shape (3,), got {extent_shape}")
    return center_value, extent_value


def centralize_points_by_bbox(
    points: Any,
    bbox_center: Any = None,
    bbox_extent: Any = None,
    *,
    infer_from_points: bool = False,
) -> Tuple[Any, Any, Any]:
    """Center points by subtracting an explicit AABB center.

    Returns ``(centered_points, bbox_center, bbox_extent)``.  ``bbox_center`` and
    ``bbox_extent`` must describe the full scaled mesh AABB when ``points`` is a
    sampled point cloud.  A sampled point cloud cannot recover the original mesh
    bbox center unless it contains the full extrema, so callers must pass mesh
    bbox metadata by default.
    """

    _validate_points(points)
    if bbox_center is None and bbox_extent is None:
        if not infer_from_points:
            raise ValueError(
                "centralize_points_by_bbox requires bbox_center and bbox_extent; "
                "a sampled point cloud cannot recover the original mesh bbox center"
            )
        center, extent = bbox_center_mesh(points)
    elif bbox_center is None or bbox_extent is None:
        raise ValueError("bbox_center and bbox_extent must be provided together")
    else:
        center, extent = _validate_bbox_metadata(points, bbox_center, bbox_extent)
    return points - center, center, extent


def centralize_points_by_own_bbox(points: Any) -> Tuple[Any, Any, Any]:
    """Center a complete vertex set or complete point set by its own AABB.

    Do not use this for sampled mesh point clouds unless the sample is known to
    include the full AABB extrema.
    """

    return centralize_points_by_bbox(points, infer_from_points=True)

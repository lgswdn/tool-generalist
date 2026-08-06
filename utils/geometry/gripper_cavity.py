"""Exact generated-gripper finger cavity geometry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def _named_obj_vertices(path: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[tuple[float, float, float]]] = {}
    active: str | None = None
    obj_path = Path(path).expanduser().resolve()
    for line_number, raw in enumerate(
        obj_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if line.startswith("o "):
            active = line[2:].strip()
            if not active or active in groups:
                raise ValueError(
                    f"Invalid or duplicate OBJ object name at {obj_path}:{line_number}"
                )
            groups[active] = []
        elif line.startswith("v "):
            if active is None:
                raise ValueError(
                    f"OBJ vertex precedes object name at {obj_path}:{line_number}"
                )
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed OBJ vertex at {obj_path}:{line_number}")
            groups[active].append(
                (float(fields[1]), float(fields[2]), float(fields[3]))
            )
    return {
        name: np.asarray(vertices, dtype=np.float64)
        for name, vertices in groups.items()
        if vertices
    }


@lru_cache(maxsize=512)
def _cached_finger_hull_halfspaces(
    mesh_path: str,
    scale_xyz: tuple[float, float, float],
    bbox_center: tuple[float, float, float],
) -> np.ndarray:
    from scipy.spatial import ConvexHull

    groups = _named_obj_vertices(mesh_path)
    left = [
        vertices
        for name, vertices in groups.items()
        if name.startswith("left_")
    ]
    right = [
        vertices
        for name, vertices in groups.items()
        if name.startswith("right_")
    ]
    if not left or not right:
        raise ValueError(
            "Generated gripper OBJ must contain named left/right moving "
            f"parts: {mesh_path}"
        )
    scale = np.asarray(scale_xyz, dtype=np.float64)
    center = np.asarray(bbox_center, dtype=np.float64)
    vertices = np.concatenate(left + right, axis=0) * scale - center
    equations = np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
    if equations.ndim != 2 or equations.shape[1] != 4:
        raise ValueError(
            f"Finger hull equations must have shape (N, 4), got {equations.shape}"
        )
    equations.setflags(write=False)
    return equations


def finger_hull_halfspaces(
    mesh_path: str | Path,
    *,
    scale_xyz: tuple[float, float, float],
    bbox_center: tuple[float, float, float],
) -> np.ndarray:
    """Return ``normal @ point + offset <= 0`` finger-hull halfspaces."""

    return _cached_finger_hull_halfspaces(
        str(Path(mesh_path).expanduser().resolve()),
        tuple(float(value) for value in scale_xyz),
        tuple(float(value) for value in bbox_center),
    )

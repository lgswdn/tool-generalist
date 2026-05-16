"""Render success-only stabilized contact poses as a tiled PNG.

This script is intentionally Isaac-free.  It loads the ``*.stabilized_success.pt``
artifact written by ``stabilize_contact.py``, applies the persisted env-frame
object/tool poses to the raw meshes, and renders mesh triangles directly with
matplotlib.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.geometry.mesh_io import load_mesh_vertices_faces, scale_vertices


def main() -> int:
    args = _parse_args()
    artifact = Path(args.stabilized_artifact).expanduser()
    payload = _load_stabilized_success(artifact)
    output = Path(args.output).expanduser() if args.output else artifact.with_suffix(".png")
    indices = _select_indices(int(payload["num_candidates"]), args.num, args.indices)

    object_vertices, object_faces = load_mesh_vertices_faces(payload["object_mesh_path"], process=False)
    tool_vertices, tool_faces = load_mesh_vertices_faces(payload["tool_mesh_path"], process=False)
    object_vertices = scale_vertices(object_vertices, float(payload["object_scale"]))
    tool_vertices = scale_vertices(tool_vertices, _np(payload["tool_scale_xyz"], shape=(3,)))

    object_center_M = _np(payload["object_bbox_center_M"], shape=(3,))
    tool_center_M = _np(payload["tool_bbox_center_M"], shape=(3,))
    object_local = object_vertices - object_center_M.reshape(1, 3)
    tool_local = tool_vertices - tool_center_M.reshape(1, 3)

    candidates = payload["candidates"]
    object_faces = _subsample_faces(object_faces, args.max_faces_per_mesh)
    tool_faces = _subsample_faces(tool_faces, args.max_faces_per_mesh)

    _render_grid(
        output_path=output,
        payload=payload,
        candidates=candidates,
        indices=indices,
        object_local=object_local,
        object_faces=object_faces,
        tool_local=tool_local,
        tool_faces=tool_faces,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
    )
    print(f"[contact_generation.visualize] wrote {output}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stabilized contact poses from a success-only .stabilized_success.pt artifact."
    )
    parser.add_argument(
        "stabilized_artifact",
        help="Path to a contact .stabilized_success.pt file written by contact_generation/stabilize_contact.py.",
    )
    parser.add_argument("--output", default="", help="Output PNG path. Defaults to <artifact>.png.")
    parser.add_argument("--num", type=int, default=8, help="Number of stabilized cases to render into one PNG.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated candidate indices. If omitted, indices are evenly sampled.",
    )
    parser.add_argument(
        "--max-faces-per-mesh",
        type=int,
        default=8000,
        help="Deterministically subsample faces per mesh for rendering speed. Use 0 for all faces.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    return parser.parse_args()


def _load_stabilized_success(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"stabilized artifact does not exist: {path}")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"stabilized artifact must contain a dict: {path}")
    if payload.get("schema_version") != "contact_stabilized_success_v1":
        raise ValueError(
            f"expected schema_version='contact_stabilized_success_v1', got {payload.get('schema_version')!r}"
        )
    if int(payload.get("num_candidates", 0)) <= 0:
        raise ValueError(f"stabilized artifact has no successful candidates: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError(f"stabilized artifact is missing candidates dict: {path}")
    for key in (
        "object_rotation_E",
        "object_bbox_center_E",
        "tool_rotation_E",
        "tool_translation_E",
        "contact_point_E",
    ):
        if key not in candidates:
            raise ValueError(f"stabilized artifact missing env-frame candidate field: {key}")
    for forbidden in _legacy_object_frame_keys():
        if forbidden in payload or forbidden in candidates:
            raise ValueError(f"legacy object-frame field is forbidden in stabilized artifact: {forbidden}")
    return payload


def _legacy_object_frame_keys() -> set[str]:
    suffix = "O"
    return {
        f"tool_translation_{suffix}",
        f"tool_rotation_{suffix}",
        f"contact_point_{suffix}",
        f"post_tool_delta_pose9d_{suffix}",
        f"post_object_delta_pose9d_{suffix}",
    }


def _select_indices(n: int, num: int, indices_arg: str) -> list[int]:
    if n <= 0:
        raise ValueError("num_candidates must be positive")
    if indices_arg.strip():
        indices = [int(item.strip()) for item in indices_arg.split(",") if item.strip()]
        if not indices:
            raise ValueError("--indices did not contain any indices")
    else:
        k = min(max(int(num), 1), n)
        indices = np.linspace(0, n - 1, num=k, dtype=np.int64).tolist()
    bad = [idx for idx in indices if idx < 0 or idx >= n]
    if bad:
        raise IndexError(f"indices out of range for {n} candidates: {bad}")
    return indices


def _np(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64)
    if shape is not None and tuple(arr.shape) != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def _candidate_np(candidates: dict[str, Any], key: str, index: int, shape: tuple[int, ...]) -> np.ndarray:
    arr = _np(candidates[key])
    item = arr[index]
    if tuple(item.shape) != shape:
        raise ValueError(f"{key}[{index}] expected shape {shape}, got {item.shape}")
    return np.asarray(item, dtype=np.float64)


def _subsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    keep = np.linspace(0, faces.shape[0] - 1, num=max_faces, dtype=np.int64)
    return faces[keep]


def _pose_vertices(local_vertices: np.ndarray, rotation_E: np.ndarray, translation_E: np.ndarray) -> np.ndarray:
    return local_vertices @ rotation_E.T + translation_E.reshape(1, 3)


def _render_grid(
    *,
    output_path: Path,
    payload: dict[str, Any],
    candidates: dict[str, Any],
    indices: Sequence[int],
    object_local: np.ndarray,
    object_faces: np.ndarray,
    tool_local: np.ndarray,
    tool_faces: np.ndarray,
    dpi: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cols = min(4, len(indices))
    rows = int(math.ceil(len(indices) / cols))
    fig = plt.figure(figsize=(4.0 * cols, 3.6 * rows), dpi=dpi)
    fig.patch.set_facecolor("white")

    for panel, index in enumerate(indices, start=1):
        ax = fig.add_subplot(rows, cols, panel, projection="3d")
        object_R = _candidate_np(candidates, "object_rotation_E", index, (3, 3))
        object_t = _candidate_np(candidates, "object_bbox_center_E", index, (3,))
        tool_R = _candidate_np(candidates, "tool_rotation_E", index, (3, 3))
        tool_t = _candidate_np(candidates, "tool_translation_E", index, (3,))
        contact = _candidate_np(candidates, "contact_point_E", index, (3,))

        object_world = _pose_vertices(object_local, object_R, object_t)
        tool_world = _pose_vertices(tool_local, tool_R, tool_t)

        _add_mesh(ax, object_world, object_faces, color=(0.62, 0.70, 0.82), alpha=0.62)
        _add_mesh(ax, tool_world, tool_faces, color=(0.95, 0.48, 0.20), alpha=0.82)
        ax.scatter(
            [contact[0]],
            [contact[1]],
            [contact[2]],
            s=26,
            c=[(0.95, 0.02, 0.02)],
            depthshade=False,
        )
        ax.set_title(f"idx={index}", fontsize=9, pad=2)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        _set_equal_axes(ax, np.vstack([object_world, tool_world, contact.reshape(1, 3)]))
        ax.set_axis_off()

    title = f"{payload.get('tool_id', '')} / {payload.get('object_id', '')}"
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _add_mesh(ax: Any, vertices: np.ndarray, faces: np.ndarray, *, color: tuple[float, float, float], alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = vertices[faces]
    collection = Poly3DCollection(
        triangles,
        facecolors=[(*color, alpha)],
        edgecolors=[(0.08, 0.08, 0.08, 0.08)],
        linewidths=0.05,
    )
    ax.add_collection3d(collection)


def _set_equal_axes(ax: Any, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        raise ValueError("cannot render non-finite vertices")
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.58)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    raise SystemExit(main())

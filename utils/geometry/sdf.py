"""Signed mesh SDF helpers for pretrain supervision.

The contact dataset does not persist SDF labels.  Pretraining computes them
from mesh vertices/faces and env-frame poses on demand.  This module deliberately
does not provide an unsigned-distance fallback: if the signed backend is missing,
the caller gets a clear error instead of silently training on wrong labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _torch():
    import torch

    return torch


def _kaolin_backend():
    try:
        import kaolin
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Signed mesh SDF supervision requires kaolin. Install/enable kaolin "
            "or disable the SDF pretrain head; unsigned distance fallback is forbidden."
        ) from exc
    missing = []
    if not hasattr(getattr(kaolin, "metrics", None), "trianglemesh"):
        missing.append("kaolin.metrics.trianglemesh")
    if not hasattr(getattr(kaolin, "ops", None), "mesh"):
        missing.append("kaolin.ops.mesh")
    if missing:
        raise RuntimeError(
            "Signed mesh SDF supervision requires kaolin trianglemesh/check_sign APIs, "
            f"missing: {missing}"
        )
    return kaolin


def signed_distance_points_to_mesh(
    points: Any,
    verts: Any,
    faces: Any,
    *,
    chunk_size: int,
    backend: str = "kaolin",
) -> Any:
    """Return signed distances from query points to a closed triangle mesh.

    Negative values mean the query point is inside the mesh according to the
    backend sign test.  Only signed backends are accepted.
    """

    if backend != "kaolin":
        raise RuntimeError(f"Unsupported signed SDF backend {backend!r}; expected 'kaolin'")
    torch = _torch()
    kaolin = _kaolin_backend()
    pts = torch.as_tensor(points)
    dev = pts.device
    dtype = pts.dtype if torch.is_floating_point(pts) else torch.float32
    pts = pts.to(device=dev, dtype=dtype).reshape(-1, 3).contiguous()
    mesh_v = torch.as_tensor(verts, device=dev, dtype=dtype).reshape(-1, 3).contiguous()
    mesh_f = torch.as_tensor(faces, device=dev, dtype=torch.long).reshape(-1, 3).contiguous()
    if pts.numel() == 0:
        return torch.empty(0, dtype=dtype, device=dev)
    if mesh_v.numel() == 0 or mesh_f.numel() == 0:
        raise RuntimeError("Signed mesh SDF requires non-empty mesh vertices and faces")
    if not bool(torch.isfinite(pts).all()) or not bool(torch.isfinite(mesh_v).all()):
        raise RuntimeError("Signed mesh SDF received non-finite points or vertices")

    face_vertices = mesh_v[mesh_f].unsqueeze(0).contiguous()
    out = []
    step = max(1, int(chunk_size))
    for start in range(0, pts.shape[0], step):
        chunk = pts[start : start + step].contiguous()
        dist_raw, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            chunk.unsqueeze(0),
            face_vertices,
        )
        dist = torch.sqrt(torch.clamp(dist_raw.squeeze(0).to(dtype=dtype), min=0.0))
        try:
            inside = kaolin.ops.mesh.check_sign(
                mesh_v.unsqueeze(0),
                mesh_f,
                chunk.unsqueeze(0),
            ).squeeze(0)
        except Exception as exc:  # pragma: no cover - backend/version dependent.
            raise RuntimeError(
                "Signed mesh SDF backend failed during inside/outside sign check; "
                "unsigned fallback is forbidden."
            ) from exc
        out.append(torch.where(inside.to(device=dev), -dist, dist))
    return torch.cat(out, dim=0)


def _signed_distance_points_to_prepared_mesh(
    points: Any,
    *,
    mesh_v: Any,
    mesh_f: Any,
    face_vertices: Any,
    chunk_size: int,
    backend: str,
) -> Any:
    """Return signed distances using already-device-local mesh tensors."""

    if backend != "kaolin":
        raise RuntimeError(f"Unsupported signed SDF backend {backend!r}; expected 'kaolin'")
    torch = _torch()
    kaolin = _kaolin_backend()
    pts = torch.as_tensor(points, device=mesh_v.device, dtype=mesh_v.dtype).reshape(-1, 3).contiguous()
    if pts.numel() == 0:
        return torch.empty(0, dtype=mesh_v.dtype, device=mesh_v.device)
    if not bool(torch.isfinite(pts).all()) or not bool(torch.isfinite(mesh_v).all()):
        raise RuntimeError("Signed mesh SDF received non-finite points or vertices")

    out = []
    step = max(1, int(chunk_size))
    for start in range(0, pts.shape[0], step):
        chunk = pts[start : start + step].contiguous()
        dist_raw, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            chunk.unsqueeze(0),
            face_vertices,
        )
        dist = torch.sqrt(torch.clamp(dist_raw.squeeze(0).to(dtype=mesh_v.dtype), min=0.0))
        try:
            inside = kaolin.ops.mesh.check_sign(
                mesh_v.unsqueeze(0),
                mesh_f,
                chunk.unsqueeze(0),
            ).squeeze(0)
        except Exception as exc:  # pragma: no cover - backend/version dependent.
            raise RuntimeError(
                "Signed mesh SDF backend failed during inside/outside sign check; "
                "unsigned fallback is forbidden."
            ) from exc
        out.append(torch.where(inside.to(device=mesh_v.device), -dist, dist))
    return torch.cat(out, dim=0)


def mutual_signed_sdf_labels_env_frame(
    *,
    tool_query_points_E: Any,
    object_query_points_E: Any,
    object_mesh_vertices: Sequence[Any] | Any,
    object_mesh_faces: Sequence[Any] | Any,
    tool_mesh_vertices: Sequence[Any] | Any,
    tool_mesh_faces: Sequence[Any] | Any,
    object_rotation_E: Any,
    object_bbox_center_E: Any,
    tool_rotation_E_k: Any,
    tool_translation_E_k: Any,
    chunk_size: int,
    backend: str = "kaolin",
) -> tuple[Any, Any]:
    """Compute mutual signed SDF labels for env-frame pretrain timesteps.

    Inputs use row-vector convention:
    ``x_E = x_centered @ R_E.T + bbox_center_E``.  Query points are env-frame.
    Internally, points are transformed into the corresponding centered mesh
    coordinates for SDF evaluation.
    """

    torch = _torch()
    tool_pts_E = torch.as_tensor(tool_query_points_E)
    obj_pts_E = torch.as_tensor(object_query_points_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    obj_R = torch.as_tensor(object_rotation_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    obj_t = torch.as_tensor(object_bbox_center_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_R = torch.as_tensor(tool_rotation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_t = torch.as_tensor(tool_translation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    if tool_pts_E.ndim != 4 or obj_pts_E.ndim != 4:
        raise RuntimeError("SDF query points must have shape (B, T, N, 3)")
    B, T, tool_n, _ = tool_pts_E.shape
    if obj_pts_E.shape[:2] != (B, T) or obj_R.shape != (B, 3, 3) or obj_t.shape != (B, 3):
        raise RuntimeError("SDF object pose/query shapes do not match batch dimensions")
    if tool_R.shape != (B, T, 3, 3) or tool_t.shape != (B, T, 3):
        raise RuntimeError("SDF tool pose shapes do not match batch/timestep dimensions")

    obj_v_list = _as_list(object_mesh_vertices, B)
    obj_f_list = _as_list(object_mesh_faces, B)
    tool_v_list = _as_list(tool_mesh_vertices, B)
    tool_f_list = _as_list(tool_mesh_faces, B)

    tool_sdf = torch.empty(B, T, tool_n, dtype=tool_pts_E.dtype, device=tool_pts_E.device)
    obj_sdf = torch.empty(B, T, obj_pts_E.shape[2], dtype=tool_pts_E.dtype, device=tool_pts_E.device)
    for b in range(B):
        obj_v = torch.as_tensor(obj_v_list[b], device=tool_pts_E.device, dtype=tool_pts_E.dtype)
        obj_f = torch.as_tensor(obj_f_list[b], device=tool_pts_E.device, dtype=torch.long)
        tool_v = torch.as_tensor(tool_v_list[b], device=tool_pts_E.device, dtype=tool_pts_E.dtype)
        tool_f = torch.as_tensor(tool_f_list[b], device=tool_pts_E.device, dtype=torch.long)
        obj_face_vertices = obj_v[obj_f].unsqueeze(0).contiguous()
        tool_face_vertices = tool_v[tool_f].unsqueeze(0).contiguous()

        # Object pose and mesh are fixed across all pre-contact timesteps for a
        # contact case, so query every timestep's tool points in one signed-SDF
        # call.  Env-frame query -> object centered mesh coordinates.
        q_tool_obj = (tool_pts_E[b].reshape(T * tool_n, 3) - obj_t[b].reshape(1, 3)) @ obj_R[b]
        tool_sdf[b] = _signed_distance_points_to_prepared_mesh(
            q_tool_obj,
            mesh_v=obj_v,
            mesh_f=obj_f,
            face_vertices=obj_face_vertices,
            chunk_size=chunk_size,
            backend=backend,
        ).reshape(T, tool_n)

        # The tool mesh is also fixed; only its env-frame pose changes across
        # timesteps.  Transform all object query points into each timestep's
        # tool-centered coordinates, then query the fixed tool mesh once.
        q_obj_tool = torch.matmul(
            obj_pts_E[b] - tool_t[b].reshape(T, 1, 3),
            tool_R[b],
        ).reshape(T * obj_pts_E.shape[2], 3)
        obj_sdf[b] = _signed_distance_points_to_prepared_mesh(
            q_obj_tool,
            mesh_v=tool_v,
            mesh_f=tool_f,
            face_vertices=tool_face_vertices,
            chunk_size=chunk_size,
            backend=backend,
        ).reshape(T, obj_pts_E.shape[2])
    return tool_sdf, obj_sdf


def _as_list(value: Sequence[Any] | Any, expected: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        try:
            if hasattr(value, "shape") and int(value.shape[0]) == expected:
                items = [value[i] for i in range(expected)]
            else:
                items = [value for _ in range(expected)]
        except Exception:
            items = [value for _ in range(expected)]
    if len(items) != expected:
        raise RuntimeError(f"Expected {expected} mesh entries for SDF batch, got {len(items)}")
    return items

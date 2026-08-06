"""Signed mesh SDF helpers for generation and pretrain supervision.

Controlled contact-quality datasets persist these distances during generation;
other datasets compute them from mesh vertices/faces and poses on demand. This
module deliberately has no unsigned-distance fallback.
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


def _closest_point_and_face_normal(
    points: Any,
    *,
    face_vertices: Any,
    face_indices: Any,
    distance_types: Any,
) -> tuple[Any, Any]:
    """Reconstruct Kaolin's closest point and selected triangle normal."""

    torch = _torch()
    selected = face_vertices[0].index_select(0, face_indices.to(dtype=torch.long))
    vertex0, vertex1, vertex2 = selected.unbind(dim=1)
    edge01 = vertex1 - vertex0
    edge12 = vertex2 - vertex1
    edge20 = vertex0 - vertex2
    normal = torch.cross(edge01, vertex2 - vertex0, dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-12)

    plane_offset = ((points - vertex0) * normal).sum(dim=-1, keepdim=True)
    closest = points - plane_offset * normal

    def edge_point(origin: Any, edge: Any) -> Any:
        parameter = ((points - origin) * edge).sum(dim=-1, keepdim=True)
        parameter = parameter / edge.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return origin + parameter.clamp(0.0, 1.0) * edge

    candidates = (
        closest,
        vertex0,
        vertex1,
        vertex2,
        edge_point(vertex0, edge01),
        edge_point(vertex1, edge12),
        edge_point(vertex2, edge20),
    )
    for distance_type, candidate in enumerate(candidates):
        mask = distance_types == distance_type
        closest = torch.where(mask.unsqueeze(-1), candidate, closest)
    return closest, normal


def _signed_distance_geometry_points_to_prepared_mesh(
    points: Any,
    *,
    mesh_v: Any,
    mesh_f: Any,
    face_vertices: Any,
    chunk_size: int,
    backend: str,
) -> tuple[Any, Any, Any]:
    """Return signed distance, query-to-closest displacement, and face normal."""

    if backend != "kaolin":
        raise RuntimeError(f"Unsupported signed SDF backend {backend!r}; expected 'kaolin'")
    torch = _torch()
    kaolin = _kaolin_backend()
    pts = torch.as_tensor(points, device=mesh_v.device, dtype=mesh_v.dtype).reshape(-1, 3).contiguous()
    signed_parts = []
    displacement_parts = []
    normal_parts = []
    step = max(1, int(chunk_size))
    for start in range(0, pts.shape[0], step):
        chunk = pts[start : start + step].contiguous()
        dist_raw, face_indices, distance_types = (
            kaolin.metrics.trianglemesh.point_to_mesh_distance(
                chunk.unsqueeze(0),
                face_vertices,
            )
        )
        face_indices = face_indices.squeeze(0)
        distance_types = distance_types.squeeze(0)
        closest, normal = _closest_point_and_face_normal(
            chunk,
            face_vertices=face_vertices,
            face_indices=face_indices,
            distance_types=distance_types,
        )
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
        distance = torch.sqrt(torch.clamp(dist_raw.squeeze(0), min=0.0))
        signed_parts.append(torch.where(inside, -distance, distance))
        displacement_parts.append(closest - chunk)
        normal_parts.append(normal)
    return (
        torch.cat(signed_parts, dim=0),
        torch.cat(displacement_parts, dim=0),
        torch.cat(normal_parts, dim=0),
    )


def _unsigned_distance_points_to_prepared_mesh(
    points: Any,
    *,
    mesh_v: Any,
    mesh_f: Any,
    face_vertices: Any,
    chunk_size: int,
    backend: str,
) -> Any:
    """Return exact point-to-triangle distance without an inside/outside query."""

    if backend != "kaolin":
        raise RuntimeError(f"Unsupported unsigned mesh-distance backend {backend!r}; expected 'kaolin'")
    torch = _torch()
    kaolin = _kaolin_backend()
    pts = torch.as_tensor(points, device=mesh_v.device, dtype=mesh_v.dtype).reshape(-1, 3).contiguous()
    if pts.numel() == 0:
        return torch.empty(0, dtype=mesh_v.dtype, device=mesh_v.device)
    out = []
    step = max(1, int(chunk_size))
    for start in range(0, pts.shape[0], step):
        dist_raw, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            pts[start : start + step].contiguous().unsqueeze(0),
            face_vertices,
        )
        out.append(torch.sqrt(torch.clamp(dist_raw.squeeze(0), min=0.0)))
    return torch.cat(out, dim=0).to(dtype=mesh_v.dtype)


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


def mutual_unsigned_mesh_distance_env_frame(
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
    """Compute mutual exact unsigned point-to-triangle distances in env frame."""

    torch = _torch()
    tool_pts_E = torch.as_tensor(tool_query_points_E)
    obj_pts_E = torch.as_tensor(
        object_query_points_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype
    )
    obj_R = torch.as_tensor(object_rotation_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    obj_t = torch.as_tensor(object_bbox_center_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_R = torch.as_tensor(tool_rotation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_t = torch.as_tensor(tool_translation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    if tool_pts_E.ndim != 4 or obj_pts_E.ndim != 4:
        raise RuntimeError("mesh-distance query points must have shape (B, T, N, 3)")
    batch_size, timesteps, tool_n, _ = tool_pts_E.shape
    if obj_pts_E.shape[:2] != (batch_size, timesteps):
        raise RuntimeError("unsigned mesh-distance query batch dimensions do not match")

    obj_v_list = _as_list(object_mesh_vertices, batch_size)
    obj_f_list = _as_list(object_mesh_faces, batch_size)
    tool_v_list = _as_list(tool_mesh_vertices, batch_size)
    tool_f_list = _as_list(tool_mesh_faces, batch_size)
    tool_distance = torch.empty(
        batch_size, timesteps, tool_n, dtype=tool_pts_E.dtype, device=tool_pts_E.device
    )
    object_n = obj_pts_E.shape[2]
    object_distance = torch.empty(
        batch_size, timesteps, object_n, dtype=tool_pts_E.dtype, device=tool_pts_E.device
    )
    for batch_index in range(batch_size):
        obj_v = torch.as_tensor(
            obj_v_list[batch_index], device=tool_pts_E.device, dtype=tool_pts_E.dtype
        )
        obj_f = torch.as_tensor(
            obj_f_list[batch_index], device=tool_pts_E.device, dtype=torch.long
        )
        tool_v = torch.as_tensor(
            tool_v_list[batch_index], device=tool_pts_E.device, dtype=tool_pts_E.dtype
        )
        tool_f = torch.as_tensor(
            tool_f_list[batch_index], device=tool_pts_E.device, dtype=torch.long
        )
        obj_faces = obj_v[obj_f].unsqueeze(0).contiguous()
        tool_faces = tool_v[tool_f].unsqueeze(0).contiguous()
        query_tool_in_object = (
            tool_pts_E[batch_index].reshape(timesteps * tool_n, 3)
            - obj_t[batch_index].reshape(1, 3)
        ) @ obj_R[batch_index]
        tool_distance[batch_index] = _unsigned_distance_points_to_prepared_mesh(
            query_tool_in_object,
            mesh_v=obj_v,
            mesh_f=obj_f,
            face_vertices=obj_faces,
            chunk_size=chunk_size,
            backend=backend,
        ).reshape(timesteps, tool_n)
        query_object_in_tool = torch.matmul(
            obj_pts_E[batch_index] - tool_t[batch_index].reshape(timesteps, 1, 3),
            tool_R[batch_index],
        ).reshape(timesteps * object_n, 3)
        object_distance[batch_index] = _unsigned_distance_points_to_prepared_mesh(
            query_object_in_tool,
            mesh_v=tool_v,
            mesh_f=tool_f,
            face_vertices=tool_faces,
            chunk_size=chunk_size,
            backend=backend,
        ).reshape(timesteps, object_n)
    return tool_distance, object_distance


def mutual_signed_sdf_geometry_env_frame(
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Mutual signed SDF plus exact closest displacement/normal in env axes.

    The outputs are tool SDF/displacement/normal followed by object
    SDF/displacement/normal.  Displacements and normals are expressed in the
    same shared axes as the input point clouds; translations do not affect
    these vectors.
    """

    torch = _torch()
    tool_pts_E = torch.as_tensor(tool_query_points_E)
    obj_pts_E = torch.as_tensor(
        object_query_points_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype
    )
    obj_R = torch.as_tensor(object_rotation_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    obj_t = torch.as_tensor(object_bbox_center_E, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_R = torch.as_tensor(tool_rotation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_t = torch.as_tensor(tool_translation_E_k, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    if tool_pts_E.ndim != 4 or obj_pts_E.ndim != 4:
        raise RuntimeError("SDF geometry query points must have shape (B, T, N, 3)")
    batch_size, timesteps, tool_n, _ = tool_pts_E.shape
    object_n = obj_pts_E.shape[2]
    obj_v_list = _as_list(object_mesh_vertices, batch_size)
    obj_f_list = _as_list(object_mesh_faces, batch_size)
    tool_v_list = _as_list(tool_mesh_vertices, batch_size)
    tool_f_list = _as_list(tool_mesh_faces, batch_size)

    tool_sdf = torch.empty(batch_size, timesteps, tool_n, device=tool_pts_E.device, dtype=tool_pts_E.dtype)
    tool_displacement = torch.empty(
        batch_size, timesteps, tool_n, 3, device=tool_pts_E.device, dtype=tool_pts_E.dtype
    )
    tool_normal = torch.empty_like(tool_displacement)
    object_sdf = torch.empty(
        batch_size, timesteps, object_n, device=tool_pts_E.device, dtype=tool_pts_E.dtype
    )
    object_displacement = torch.empty(
        batch_size, timesteps, object_n, 3, device=tool_pts_E.device, dtype=tool_pts_E.dtype
    )
    object_normal = torch.empty_like(object_displacement)

    for batch_index in range(batch_size):
        obj_v = torch.as_tensor(
            obj_v_list[batch_index], device=tool_pts_E.device, dtype=tool_pts_E.dtype
        )
        obj_f = torch.as_tensor(
            obj_f_list[batch_index], device=tool_pts_E.device, dtype=torch.long
        )
        tool_v = torch.as_tensor(
            tool_v_list[batch_index], device=tool_pts_E.device, dtype=tool_pts_E.dtype
        )
        tool_f = torch.as_tensor(
            tool_f_list[batch_index], device=tool_pts_E.device, dtype=torch.long
        )
        obj_faces = obj_v[obj_f].unsqueeze(0).contiguous()
        tool_faces = tool_v[tool_f].unsqueeze(0).contiguous()

        query_tool_in_object = (
            tool_pts_E[batch_index].reshape(timesteps * tool_n, 3)
            - obj_t[batch_index].reshape(1, 3)
        ) @ obj_R[batch_index]
        signed, displacement, normal = _signed_distance_geometry_points_to_prepared_mesh(
            query_tool_in_object,
            mesh_v=obj_v,
            mesh_f=obj_f,
            face_vertices=obj_faces,
            chunk_size=chunk_size,
            backend=backend,
        )
        tool_sdf[batch_index] = signed.reshape(timesteps, tool_n)
        tool_displacement[batch_index] = (
            displacement.reshape(timesteps, tool_n, 3) @ obj_R[batch_index].T
        )
        tool_normal[batch_index] = normal.reshape(timesteps, tool_n, 3) @ obj_R[batch_index].T

        query_object_in_tool = torch.matmul(
            obj_pts_E[batch_index] - tool_t[batch_index].reshape(timesteps, 1, 3),
            tool_R[batch_index],
        ).reshape(timesteps * object_n, 3)
        signed, displacement, normal = _signed_distance_geometry_points_to_prepared_mesh(
            query_object_in_tool,
            mesh_v=tool_v,
            mesh_f=tool_f,
            face_vertices=tool_faces,
            chunk_size=chunk_size,
            backend=backend,
        )
        object_sdf[batch_index] = signed.reshape(timesteps, object_n)
        object_displacement[batch_index] = torch.matmul(
            displacement.reshape(timesteps, object_n, 3),
            tool_R[batch_index].transpose(-1, -2),
        )
        object_normal[batch_index] = torch.matmul(
            normal.reshape(timesteps, object_n, 3),
            tool_R[batch_index].transpose(-1, -2),
        )
    return (
        tool_sdf,
        tool_displacement,
        tool_normal,
        object_sdf,
        object_displacement,
        object_normal,
    )


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

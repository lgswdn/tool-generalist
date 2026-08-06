"""Fast signed point-to-triangle-mesh queries for Isaac RL observations."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp


_WARP_MESH_CACHE: dict[tuple[int, int, str, bool], Any] = {}


@wp.kernel
def _signed_mesh_distance_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, point, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest) * query.sign
    else:
        distances[index] = max_dist


@wp.kernel
def _unsigned_mesh_distance_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest)
    else:
        distances[index] = max_dist


def _cached_warp_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    support_winding_number: bool,
):
    vertices = vertices.contiguous()
    key = (
        int(vertices.data_ptr()),
        int(faces.data_ptr()),
        str(vertices.device),
        bool(support_winding_number),
    )
    cached = _WARP_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    faces_i32 = faces.reshape(-1).to(dtype=torch.int32).contiguous()
    points_wp = wp.from_torch(vertices, dtype=wp.vec3)
    indices_wp = wp.from_torch(faces_i32, dtype=wp.int32)
    mesh = wp.Mesh(
        points=points_wp,
        indices=indices_wp,
        support_winding_number=bool(support_winding_number),
        bvh_constructor="lbvh",
    )
    # Keep the converted index tensor alive alongside Warp's mesh buffers.
    cached = (mesh, vertices, faces_i32)
    _WARP_MESH_CACHE[key] = cached
    return cached


def signed_distance_points_to_prepared_mesh_warp(
    points: torch.Tensor,
    *,
    mesh_v: torch.Tensor,
    mesh_f: torch.Tensor,
    max_dist: float = 1.0e6,
) -> torch.Tensor:
    """Return signed distance in mesh units using a cached Warp BVH.

    Distance is the exact closest-triangle Euclidean distance. Sign is obtained
    from Warp's robust winding-number query; negative means inside.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Warp mesh SDF points must have shape (N, 3), got {tuple(points.shape)}")
    if points.device != mesh_v.device or mesh_v.device != mesh_f.device:
        raise RuntimeError("Warp mesh SDF tensors must share one device")
    if points.dtype != torch.float32 or mesh_v.dtype != torch.float32:
        raise RuntimeError("Warp mesh SDF currently requires float32 points and vertices")
    if points.numel() == 0:
        return torch.empty((0,), device=points.device, dtype=points.dtype)

    mesh, _, _ = _cached_warp_mesh(mesh_v, mesh_f, support_winding_number=True)
    query_points = points.contiguous()
    output = torch.empty((query_points.shape[0],), device=points.device, dtype=torch.float32)
    warp_device = wp.device_from_torch(points.device)
    warp_stream = wp.stream_from_torch(points.device) if points.is_cuda else None
    wp.launch(
        kernel=_signed_mesh_distance_kernel,
        dim=query_points.shape[0],
        inputs=[mesh.id, wp.from_torch(query_points, dtype=wp.vec3), float(max_dist)],
        outputs=[wp.from_torch(output, dtype=wp.float32)],
        device=warp_device,
        stream=warp_stream,
    )
    return output


def unsigned_distance_points_to_prepared_mesh_warp(
    points: torch.Tensor,
    *,
    mesh_v: torch.Tensor,
    mesh_f: torch.Tensor,
    max_dist: float = 1.0e6,
) -> torch.Tensor:
    """Return exact unsigned closest-triangle distance using a cached Warp BVH."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Warp mesh-distance points must have shape (N, 3), got {tuple(points.shape)}")
    if points.device != mesh_v.device or mesh_v.device != mesh_f.device:
        raise RuntimeError("Warp mesh-distance tensors must share one device")
    if points.dtype != torch.float32 or mesh_v.dtype != torch.float32:
        raise RuntimeError("Warp mesh distance currently requires float32")
    if points.numel() == 0:
        return torch.empty((0,), device=points.device, dtype=points.dtype)
    mesh, _, _ = _cached_warp_mesh(mesh_v, mesh_f, support_winding_number=False)
    query_points = points.contiguous()
    output = torch.empty((query_points.shape[0],), device=points.device, dtype=torch.float32)
    warp_device = wp.device_from_torch(points.device)
    warp_stream = wp.stream_from_torch(points.device) if points.is_cuda else None
    wp.launch(
        kernel=_unsigned_mesh_distance_kernel,
        dim=query_points.shape[0],
        inputs=[mesh.id, wp.from_torch(query_points, dtype=wp.vec3), float(max_dist)],
        outputs=[wp.from_torch(output, dtype=wp.float32)],
        device=warp_device,
        stream=warp_stream,
    )
    return output

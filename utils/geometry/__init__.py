"""Geometry helpers shared by contact generation and validation."""

from .bbox import bbox_center_mesh, centralize_points_by_bbox, centralize_points_by_own_bbox
from .mesh_io import (
    load_mesh_tensors,
    load_mesh_vertices_faces,
    load_scaled_sampled_surface_points,
    sample_surface_points_numpy,
    sample_surface_points_torch,
    scale_vertices,
    scaled_mesh_bbox,
)
from .pose import (
    apply_pose9d_delta,
    apply_pose_about_bbox_center,
    pose9d_from_rt,
    pose9d_from_transform_np,
    rotation_from_pose9d,
    rotation_from_pose9d_np,
)
from .sdf import (
    mutual_signed_sdf_geometry_env_frame,
    mutual_signed_sdf_labels_env_frame,
    mutual_unsigned_mesh_distance_env_frame,
    signed_distance_points_to_mesh,
)

__all__ = [
    "apply_pose_about_bbox_center",
    "bbox_center_mesh",
    "centralize_points_by_bbox",
    "centralize_points_by_own_bbox",
    "apply_pose9d_delta",
    "load_mesh_vertices_faces",
    "load_mesh_tensors",
    "load_scaled_sampled_surface_points",
    "pose9d_from_rt",
    "pose9d_from_transform_np",
    "rotation_from_pose9d",
    "rotation_from_pose9d_np",
    "sample_surface_points_numpy",
    "sample_surface_points_torch",
    "scale_vertices",
    "scaled_mesh_bbox",
    "mutual_signed_sdf_labels_env_frame",
    "mutual_signed_sdf_geometry_env_frame",
    "mutual_unsigned_mesh_distance_env_frame",
    "signed_distance_points_to_mesh",
]

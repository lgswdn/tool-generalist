import math

import pytest
import numpy as np
import torch

from utils.geometry import (
    apply_pose_about_bbox_center,
    bbox_center_mesh,
    centralize_points_by_bbox,
    centralize_points_by_own_bbox,
)


def test_bbox_center_and_extent_are_aabb_midpoint():
    points = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 5.0], [2.0, 4.0, 4.0]])

    center, extent = bbox_center_mesh(points)

    assert torch.allclose(center, torch.tensor([2.0, 4.0, 4.0]))
    assert torch.allclose(extent, torch.tensor([2.0, 4.0, 2.0]))


def test_centralize_points_by_bbox_uses_bbox_center_not_mean():
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 10.0, 0.0]])

    centered, center, extent = centralize_points_by_own_bbox(points)

    assert torch.allclose(center, torch.tensor([1.0, 5.0, 0.0]))
    assert torch.allclose(extent, torch.tensor([2.0, 10.0, 0.0]))
    centered_center, _ = bbox_center_mesh(centered)
    assert torch.allclose(centered_center, torch.zeros(3))


def test_apply_pose_about_bbox_center_rotates_around_aabb_center():
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rotation = torch.tensor(
        [
            [math.cos(math.pi), -math.sin(math.pi), 0.0],
            [math.sin(math.pi), math.cos(math.pi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([10.0, 0.0, 0.0])

    posed = apply_pose_about_bbox_center(points, rotation, translation)

    assert torch.allclose(posed, torch.tensor([[12.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), atol=1e-6)
    center, _ = bbox_center_mesh(posed)
    assert torch.allclose(center, torch.tensor([11.0, 0.0, 0.0]), atol=1e-6)


def test_numpy_points_are_supported():
    points = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32)

    centered, center, extent = centralize_points_by_bbox(
        points,
        bbox_center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bbox_extent=np.array([2.0, 2.0, 2.0], dtype=np.float32),
    )

    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(extent, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(centered, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    assert centered.dtype == points.dtype
    assert center.dtype == points.dtype
    assert extent.dtype == points.dtype


def test_sampled_cloud_uses_mesh_bbox_metadata_when_extrema_are_missing():
    mesh_vertices = torch.tensor(
        [[1.0, 2.0, 3.0], [5.0, 10.0, 15.0], [3.0, 4.0, 5.0]],
        dtype=torch.float64,
    )
    sampled_points = torch.tensor([[3.0, 4.0, 5.0], [3.5, 4.5, 5.5]], dtype=torch.float64)
    mesh_center, mesh_extent = bbox_center_mesh(mesh_vertices)

    centered, center, extent = centralize_points_by_bbox(
        sampled_points,
        bbox_center=mesh_center,
        bbox_extent=mesh_extent,
    )

    assert centered.dtype == sampled_points.dtype
    assert centered.device == sampled_points.device
    assert torch.allclose(center, torch.tensor([3.0, 6.0, 9.0], dtype=torch.float64))
    assert torch.allclose(extent, torch.tensor([4.0, 8.0, 12.0], dtype=torch.float64))
    assert torch.allclose(centered, sampled_points - mesh_center)
    own_center, _ = bbox_center_mesh(centered)
    assert not torch.allclose(own_center, torch.zeros(3, dtype=torch.float64))


def test_sampled_cloud_without_bbox_metadata_is_rejected():
    sampled_points = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])

    with pytest.raises(ValueError, match="sampled point cloud cannot recover the original mesh bbox center"):
        centralize_points_by_bbox(sampled_points)


def test_infer_from_points_is_explicit_for_complete_point_sets():
    points = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])

    centered, center, extent = centralize_points_by_bbox(points, infer_from_points=True)

    assert torch.allclose(center, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(extent, torch.tensor([4.0, 4.0, 4.0]))
    assert torch.allclose(centered, torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]))


def test_bbox_metadata_shape_is_validated():
    points = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="bbox_center must have shape"):
        centralize_points_by_bbox(points, bbox_center=torch.zeros(1, 3), bbox_extent=torch.ones(3))

    with pytest.raises(ValueError, match="bbox_extent must have shape"):
        centralize_points_by_bbox(points, bbox_center=torch.zeros(3), bbox_extent=torch.ones(1, 3))

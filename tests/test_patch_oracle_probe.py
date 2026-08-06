import torch

from pretrain.patch_oracle_probe import (
    PATCH_ORACLE_FEATURE_NAMES,
    PatchOracleToRankToken,
    build_patch_oracle_features,
)
from utils.geometry.sdf import _closest_point_and_face_normal


def _inputs():
    generator = torch.Generator().manual_seed(7)
    points = torch.randn(2, 6, 32, 3, generator=generator) * 0.02
    centers = points[:, :, 0].clone()
    sdf = torch.randn(2, 6, 32, generator=generator) * 0.01
    displacement = torch.randn(2, 6, 32, 3, generator=generator) * 0.01
    normal = torch.nn.functional.normalize(
        torch.randn(2, 6, 32, 3, generator=generator), dim=-1
    )
    patch_is_tool = torch.zeros(2, 6, dtype=torch.bool)
    patch_is_tool[:, :3] = True
    return points, centers, sdf, displacement, normal, patch_is_tool


def _features(points, centers, sdf, displacement, normal, patch_is_tool):
    return build_patch_oracle_features(
        patch_points=points,
        patch_centers=centers,
        signed_sdf=sdf,
        closest_displacement=displacement,
        closest_normal=normal,
        patch_is_tool=patch_is_tool,
    )


def test_patch_oracle_feature_contract_is_strictly_patch_equivariant():
    points, centers, sdf, displacement, normal, patch_is_tool = _inputs()
    baseline = _features(points, centers, sdf, displacement, normal, patch_is_tool)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted = _features(
        points[:, permutation],
        centers[:, permutation],
        sdf[:, permutation],
        displacement[:, permutation],
        normal[:, permutation],
        patch_is_tool[:, permutation],
    )

    assert baseline.shape == (2, 6, len(PATCH_ORACLE_FEATURE_NAMES))
    torch.testing.assert_close(permuted, baseline[:, permutation], atol=1e-6, rtol=1e-6)


def test_patch_oracle_features_are_invariant_to_point_order_within_patch():
    points, centers, sdf, displacement, normal, patch_is_tool = _inputs()
    baseline = _features(points, centers, sdf, displacement, normal, patch_is_tool)
    point_permutation = torch.randperm(32, generator=torch.Generator().manual_seed(11))
    permuted = _features(
        points[:, :, point_permutation],
        centers,
        sdf[:, :, point_permutation],
        displacement[:, :, point_permutation],
        normal[:, :, point_permutation],
        patch_is_tool,
    )

    torch.testing.assert_close(permuted, baseline, atol=2e-6, rtol=2e-6)


def test_modifying_one_patch_cannot_change_another_patch_features_or_probe_output():
    points, centers, sdf, displacement, normal, patch_is_tool = _inputs()
    baseline = _features(points, centers, sdf, displacement, normal, patch_is_tool)
    changed_points = points.clone()
    changed_sdf = sdf.clone()
    changed_displacement = displacement.clone()
    changed_normal = normal.clone()
    changed_points[:, 2] += 0.3
    changed_sdf[:, 2] -= 0.2
    changed_displacement[:, 2] += 0.1
    changed_normal[:, 2] *= -1
    changed = _features(
        changed_points,
        centers,
        changed_sdf,
        changed_displacement,
        changed_normal,
        patch_is_tool,
    )

    untouched = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(changed[:, untouched], baseline[:, untouched])
    probe = PatchOracleToRankToken()
    torch.testing.assert_close(
        probe(changed)[:, untouched],
        probe(baseline)[:, untouched],
    )


def test_closest_triangle_geometry_reconstructs_face_vertex_and_edge_cases():
    face_vertices = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]
    )
    points = torch.tensor(
        [
            [0.25, 0.25, 2.0],
            [-1.0, -1.0, 0.0],
            [0.50, -1.0, 0.0],
        ]
    )
    closest, normal = _closest_point_and_face_normal(
        points,
        face_vertices=face_vertices,
        face_indices=torch.zeros(3, dtype=torch.long),
        distance_types=torch.tensor([0, 1, 4], dtype=torch.int32),
    )

    torch.testing.assert_close(
        closest,
        torch.tensor([[0.25, 0.25, 0.0], [0.0, 0.0, 0.0], [0.50, 0.0, 0.0]]),
    )
    torch.testing.assert_close(normal, torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1))


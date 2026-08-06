"""The canonical 128-bin, corresponding-point cloud representation for grippers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from utils.assets.one_dof_gripper_assets import OneDofGripperAsset
from utils.assets.generated_gripper_assets import (
    GeneratedGripperAsset,
    PrismaticJointSpec,
    RigidTransformSpec,
)
from utils.geometry.one_dof_gripper_kinematics import (
    one_dof_body_poses,
    transform_points,
)
from utils.geometry.gripper_cloud_contract import (
    CACHE_SCHEMA_VERSION,
    NUM_BINS,
    NUM_POINTS,
)


DEFAULT_NUM_BINS = NUM_BINS
DEFAULT_NUM_POINTS = NUM_POINTS


def cache_path_for_asset(
    asset: OneDofGripperAsset | GeneratedGripperAsset,
    cache_dir: str | Path | None = None,
) -> Path:
    if cache_dir is None:
        if isinstance(asset, GeneratedGripperAsset):
            raise ValueError(
                "Generated parallel grippers require an explicit cloud-cache directory"
            )
        root = asset.manifest_path.parent / "kinematic_cloud_cache"
    else:
        root = Path(cache_dir)
    return root.expanduser().resolve() / f"{asset.gripper_id}.pt"


def _counts(total: int, components: int) -> list[int]:
    if total < components:
        raise ValueError(
            f"Cannot distribute {total} points across {components} components"
        )
    base, remainder = divmod(total, components)
    return [base + int(index < remainder) for index in range(components)]


def _rng(asset_id: str, part_index: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{asset_id}:{part_index}:canonical-cloud-v1".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _sample_box(
    size: tuple[float, float, float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size_a = np.asarray(size, dtype=np.float64)
    points = (rng.random((count, 3)) - 0.5) * size_a
    weights = np.asarray(
        (size_a[1] * size_a[2], size_a[0] * size_a[2], size_a[0] * size_a[1])
    )
    axes = rng.choice(3, size=count, p=weights / weights.sum())
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    points[np.arange(count), axes] = signs * size_a[axes] * 0.5
    return points


def _sample_cylinder(
    radius: float,
    length: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    angle = rng.random(count) * (2.0 * math.pi)
    side = rng.random(count) < length / (length + radius)
    radial = np.where(side, radius, radius * np.sqrt(rng.random(count)))
    z = np.where(
        side,
        length * (rng.random(count) - 0.5),
        np.where(rng.random(count) < 0.5, -0.5, 0.5) * length,
    )
    return np.stack(
        (radial * np.cos(angle), radial * np.sin(angle), z), axis=1
    )


def _sample_mesh(
    path: Path,
    scale: tuple[float, float, float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if path.suffix.lower() == ".stl":
        data = path.read_bytes()
        if len(data) < 84:
            raise ValueError(f"Binary STL is truncated: {path}")
        triangle_count = int.from_bytes(data[80:84], "little")
        if len(data) != 84 + 50 * triangle_count:
            raise ValueError(f"Expected a binary STL with no trailing data: {path}")
        records = np.frombuffer(
            data,
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            ),
            count=triangle_count,
            offset=84,
        )
        vertices = records["vertices"].reshape(-1, 3).astype(np.float64)
        faces = np.arange(3 * triangle_count, dtype=np.int64).reshape(-1, 3)
    elif path.suffix.lower() == ".obj":
        raw_vertices: list[list[float]] = []
        raw_faces: list[list[int]] = []
        for raw_line in path.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines():
            fields = raw_line.strip().split()
            if not fields:
                continue
            if fields[0] == "v":
                raw_vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                polygon = [
                    int(token.split("/", 1)[0]) - 1 for token in fields[1:]
                ]
                if any(index < 0 for index in polygon):
                    raise ValueError(
                        f"Only positive OBJ face indices are supported: {path}"
                    )
                for offset in range(1, len(polygon) - 1):
                    raw_faces.append(
                        [polygon[0], polygon[offset], polygon[offset + 1]]
                    )
        vertices = np.asarray(raw_vertices, dtype=np.float64)
        faces = np.asarray(raw_faces, dtype=np.int64)
    else:
        raise ValueError(f"Cloud cache supports only OBJ/STL meshes: {path}")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError(f"Mesh has no triangular surface: {path}")
    vertices = vertices * np.asarray(scale, dtype=np.float64)
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        raise ValueError(f"Mesh has no finite positive-area surface: {path}")
    face_ids = rng.choice(
        len(faces), size=count, replace=True, p=areas / areas.sum()
    )
    selected = triangles[face_ids]
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    return (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )


def _quat_matrix(quat_wxyz: tuple[float, float, float, float]) -> torch.Tensor:
    w, x, y, z = quat_wxyz
    return torch.tensor(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=torch.float64,
    )


def _transform_spec_points(
    points: torch.Tensor,
    transform: RigidTransformSpec,
) -> torch.Tensor:
    rotation = _quat_matrix(transform.quat_wxyz)
    translation = torch.tensor(transform.translation, dtype=torch.float64)
    return points @ rotation.T + translation


def _rpy_matrix(rpy: tuple[float, float, float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=torch.float64,
    )


def _parallel_joint_points(
    points_child: torch.Tensor,
    joint: PrismaticJointSpec,
    position: float,
) -> torch.Tensor:
    rotation = _rpy_matrix(joint.origin_rpy)
    origin = torch.tensor(joint.origin_xyz, dtype=torch.float64)
    axis = torch.tensor(joint.axis_xyz, dtype=torch.float64)
    translation = origin + rotation @ (axis * float(position))
    return points_child @ rotation.T + translation


def _sample_generated_component(
    asset: GeneratedGripperAsset,
    *,
    component: str,
    count: int,
    part_index: int,
) -> torch.Tensor:
    if component == "plank":
        path = asset.plank_mesh
    elif component == "finger":
        path = asset.finger_mesh
    elif component == "finger_tip" and asset.finger_tip_mesh is not None:
        path = asset.finger_tip_mesh
    else:
        raise ValueError(
            f"Invalid generated-gripper cloud component {component!r}"
        )
    sampled = _sample_mesh(
        path,
        (1.0, 1.0, 1.0),
        count,
        _rng(f"generated_parallel:{asset.gripper_id}", part_index),
    )
    points = _transform_spec_points(
        torch.as_tensor(sampled, dtype=torch.float64),
        asset.mesh_to_body_frame[component],
    )
    if component == "finger_tip":
        if asset.finger_tip_to_finger_frame is None:
            raise ValueError(
                f"{asset.gripper_id} has a tip mesh but no tip-to-finger transform"
            )
        points = _transform_spec_points(
            points, asset.finger_tip_to_finger_frame
        )
    return points


def _build_one_dof_cache_payload(
    asset: OneDofGripperAsset,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict:
    if num_bins < 2:
        raise ValueError("Kinematic cloud cache requires at least two bins")
    body_names = tuple(dict.fromkeys(part.body_name for part in asset.cloud_parts))
    body_to_index = {name: index for index, name in enumerate(body_names)}
    points_body = []
    point_body_index = []
    for part_index, (part, count) in enumerate(
        zip(asset.cloud_parts, _counts(num_points, len(asset.cloud_parts)))
    ):
        rng = _rng(asset.gripper_id, part_index)
        if part.geometry_type == "box":
            assert part.box_size is not None
            sampled = _sample_box(part.box_size, count, rng)
        elif part.geometry_type == "cylinder":
            assert part.cylinder_radius is not None
            assert part.cylinder_length is not None
            sampled = _sample_cylinder(
                part.cylinder_radius, part.cylinder_length, count, rng
            )
        elif part.geometry_type == "mesh":
            assert part.mesh_path is not None
            sampled = _sample_mesh(
                part.mesh_path, part.mesh_scale, count, rng
            )
        else:
            raise ValueError(
                f"Unsupported cloud geometry {part.geometry_type!r}"
            )
        points = torch.as_tensor(sampled, dtype=torch.float64)
        rotation = _quat_matrix(part.geometry_to_body.quat_wxyz)
        translation = torch.tensor(
            part.geometry_to_body.translation, dtype=torch.float64
        )
        points_body.append(points @ rotation.T + translation)
        point_body_index.extend([body_to_index[part.body_name]] * count)
    points_body_t = torch.cat(points_body, dim=0)
    body_index_t = torch.tensor(point_body_index, dtype=torch.long)

    fractions = torch.linspace(0.0, 1.0, num_bins, dtype=torch.float64)
    clouds = []
    for fraction in fractions.tolist():
        poses = one_dof_body_poses(
            asset, fraction, device="cpu", dtype=torch.float64
        )
        cloud = torch.empty_like(points_body_t)
        for body_index, body_name in enumerate(body_names):
            rows = body_index_t == body_index
            cloud[rows] = transform_points(points_body_t[rows], poses[body_name])
        clouds.append(cloud)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "gripper_id": asset.gripper_id,
        "source_manifest": str(asset.manifest_path),
        "source_asset_root": str(asset.root_dir),
        "num_bins": int(num_bins),
        "num_points": int(num_points),
        "opening_fractions": fractions.float(),
        "body_names": body_names,
        "point_body_index": body_index_t,
        "points_body": points_body_t.float(),
        "state_clouds_palm": torch.stack(clouds).float(),
    }


def _build_generated_parallel_cache_payload(
    asset: GeneratedGripperAsset,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict:
    if num_bins != DEFAULT_NUM_BINS or num_points != DEFAULT_NUM_POINTS:
        raise ValueError(
            "The gripper cloud contract is fixed at 128 bins and 512 points"
        )
    components = (
        ("plank", 0),
        ("finger", 1),
        ("finger", 2),
        *((("finger_tip", 1), ("finger_tip", 2)) if asset.has_tip else ()),
    )
    counts = _counts(num_points, len(components))
    body_names = (
        asset.palm_body_name,
        asset.finger_body_names[0],
        asset.finger_body_names[1],
    )
    points_body_parts: list[torch.Tensor] = []
    point_body_indices: list[int] = []
    for part_index, ((component, body_index), count) in enumerate(
        zip(components, counts)
    ):
        points_body_parts.append(
            _sample_generated_component(
                asset,
                component=component,
                count=count,
                part_index=part_index,
            )
        )
        point_body_indices.extend([body_index] * count)
    points_body = torch.cat(points_body_parts, dim=0)
    point_body_index = torch.tensor(point_body_indices, dtype=torch.long)

    fractions = torch.linspace(0.0, 1.0, num_bins, dtype=torch.float64)
    clouds = []
    for fraction in fractions.tolist():
        opening = float(fraction) * asset.open_joint_pos
        cloud = torch.empty_like(points_body)
        palm_rows = point_body_index == 0
        cloud[palm_rows] = points_body[palm_rows]
        for body_index, joint in enumerate(
            asset.finger_joint_local_poses, start=1
        ):
            rows = point_body_index == body_index
            cloud[rows] = _parallel_joint_points(
                points_body[rows], joint, opening
            )
        clouds.append(cloud)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "gripper_id": asset.gripper_id,
        "source_manifest": "",
        "source_asset_root": str(asset.root_dir),
        "num_bins": num_bins,
        "num_points": num_points,
        "opening_fractions": fractions.float(),
        "body_names": body_names,
        "point_body_index": point_body_index,
        "points_body": points_body.float(),
        "state_clouds_palm": torch.stack(clouds).float(),
    }


def build_cache_payload(
    asset: OneDofGripperAsset | GeneratedGripperAsset,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict:
    """Build the sole gripper-cloud representation used by every pipeline stage."""

    if num_bins != DEFAULT_NUM_BINS or num_points != DEFAULT_NUM_POINTS:
        raise ValueError(
            "The gripper cloud contract is fixed at 128 bins and 512 points"
        )
    if isinstance(asset, GeneratedGripperAsset):
        return _build_generated_parallel_cache_payload(
            asset, num_bins=num_bins, num_points=num_points
        )
    return _build_one_dof_cache_payload(
        asset, num_bins=num_bins, num_points=num_points
    )


@dataclass(frozen=True)
class GripperCloudCache:
    gripper_id: str
    source_manifest: str
    source_asset_root: str
    opening_fractions: torch.Tensor
    body_names: tuple[str, ...]
    point_body_index: torch.Tensor
    points_body: torch.Tensor
    state_clouds_palm: torch.Tensor

    def bin_index(self, fraction: float) -> int:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Opening fraction must be in [0, 1], got {value}")
        return int(round(value * (self.state_clouds_palm.shape[0] - 1)))

    def cloud_at_fraction(self, fraction: float) -> torch.Tensor:
        return self.state_clouds_palm[self.bin_index(fraction)]


def load_gripper_cloud_cache(
    path: str | Path,
    *,
    expected_gripper_id: str | None = None,
    expected_source_manifest: str | Path | None = None,
    expected_source_asset_root: str | Path | None = None,
) -> GripperCloudCache:
    cache_path = Path(path).expanduser().resolve()
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Invalid one-DoF cloud cache: {cache_path}")
    gripper_id = str(payload["gripper_id"])
    if expected_gripper_id is not None and gripper_id != expected_gripper_id:
        raise ValueError(
            f"Cloud cache id {gripper_id!r} != expected {expected_gripper_id!r}"
        )
    source_manifest = str(payload["source_manifest"])
    source_asset_root = str(payload["source_asset_root"])
    if expected_source_manifest is not None and Path(
        source_manifest
    ).expanduser().resolve() != Path(expected_source_manifest).expanduser().resolve():
        raise ValueError(
            f"Cloud cache source manifest {source_manifest!r} != expected "
            f"{str(expected_source_manifest)!r}"
        )
    if expected_source_asset_root is not None and Path(
        source_asset_root
    ).expanduser().resolve() != Path(expected_source_asset_root).expanduser().resolve():
        raise ValueError(
            f"Cloud cache source asset root {source_asset_root!r} != expected "
            f"{str(expected_source_asset_root)!r}"
        )
    cache = GripperCloudCache(
        gripper_id=gripper_id,
        source_manifest=source_manifest,
        source_asset_root=source_asset_root,
        opening_fractions=torch.as_tensor(payload["opening_fractions"]).float(),
        body_names=tuple(payload["body_names"]),
        point_body_index=torch.as_tensor(payload["point_body_index"]).long(),
        points_body=torch.as_tensor(payload["points_body"]).float(),
        state_clouds_palm=torch.as_tensor(payload["state_clouds_palm"]).float(),
    )
    if cache.state_clouds_palm.shape != (128, 512, 3):
        raise ValueError(
            f"Expected cache shape (128, 512, 3), got "
            f"{tuple(cache.state_clouds_palm.shape)} at {cache_path}"
        )
    if cache.points_body.shape != (512, 3):
        raise ValueError(
            f"Expected points_body shape (512, 3), got "
            f"{tuple(cache.points_body.shape)} at {cache_path}"
        )
    if cache.point_body_index.shape != (512,):
        raise ValueError(
            f"Expected point_body_index shape (512,), got "
            f"{tuple(cache.point_body_index.shape)} at {cache_path}"
        )
    expected_fractions = torch.linspace(
        0.0, 1.0, 128, dtype=torch.float64
    ).float()
    if not torch.equal(cache.opening_fractions, expected_fractions):
        raise ValueError(
            f"Cache does not contain the exact canonical 128 opening bins: "
            f"{cache_path}"
        )
    return cache

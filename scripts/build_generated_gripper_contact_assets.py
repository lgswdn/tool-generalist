#!/usr/bin/env python3
"""Materialize generated grippers as selected-tool contact assets."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.assets import GeneratedGripperAsset, RigidTransformSpec, load_generated_gripper_manifest
from utils.geometry.gripper_cloud_cache import load_gripper_cloud_cache
from scripts.render_generated_gripper_contact_sheet import (
    _one_dof_gripper_named_polygons,
)


DEFAULT_MANIFEST = Path("/mnt/project/world_model/tool_generalist/gripper/generated_grippers.json")
DEFAULT_OUTPUT_DIR = Path("configs/generated_gripper_contact_assets")
DEFAULT_GENERATED_CACHE_DIR = Path(
    "gripper/generated_parallel_128/kinematic_cloud_cache"
)
NUM_OPENING_BINS = 128
DISTAL_FINGER_TIP_BAND_M = 0.005


class ObjMesh:
    def __init__(self, vertices: list[tuple[float, float, float]], faces: list[list[int]]) -> None:
        self.vertices = vertices
        self.faces = faces


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert generated_grippers.json entries into the selected-tool asset "
            "layout expected by contact_generation."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generated-cache-dir",
        type=Path,
        default=DEFAULT_GENERATED_CACHE_DIR,
    )
    parser.add_argument(
        "--one-dof-manifest",
        type=Path,
        default=None,
        help=(
            "Optional one-DoF manifest to merge into the generated-parallel "
            "contact population."
        ),
    )
    parser.add_argument(
        "--opening-seed",
        type=int,
        default=0,
        help="Seed for deterministic per-gripper opening randomization.",
    )
    return parser.parse_args()


def _read_mesh(path: Path) -> ObjMesh:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _read_obj(path)
    if suffix == ".stl":
        return _read_stl(path)
    raise ValueError(f"Unsupported mesh extension for generated gripper contact asset: {path}")


def _read_obj(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.split()[1:]:
                    raw = token.split("/", 1)[0]
                    index = int(raw)
                    face.append(len(vertices) + index + 1 if index < 0 else index)
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return ObjMesh(vertices, faces)


def _read_stl(path: Path) -> ObjMesh:
    raw = path.read_bytes()
    if raw[:5].lower() == b"solid":
        try:
            return _read_ascii_stl(path)
        except ValueError:
            pass
    if len(raw) < 84:
        raise ValueError(f"STL is too small: {path}")
    tri_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + tri_count * 50
    if expected > len(raw):
        raise ValueError(f"Binary STL is truncated: {path}")
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    offset = 84
    for _ in range(tri_count):
        coords = struct.unpack_from("<12f", raw, offset)
        base = len(vertices) + 1
        vertices.extend(
            [
                (coords[3], coords[4], coords[5]),
                (coords[6], coords[7], coords[8]),
                (coords[9], coords[10], coords[11]),
            ]
        )
        faces.append([base, base + 1, base + 2])
        offset += 50
    if not vertices:
        raise ValueError(f"STL has no vertices: {path}")
    return ObjMesh(vertices, faces)


def _read_ascii_stl(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    current: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            _, x, y, z = stripped.split()[:4]
            vertices.append((float(x), float(y), float(z)))
            current.append(len(vertices))
            if len(current) == 3:
                faces.append(current)
                current = []
    if not vertices:
        raise ValueError(f"ASCII STL has no vertices: {path}")
    return ObjMesh(vertices, faces)


def _write_obj(path: Path, parts: Iterable[tuple[str, ObjMesh, list[tuple[float, float, float]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offset = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated gripper contact mesh.\n")
        for name, mesh, vertices in parts:
            f.write(f"o {name}\n")
            for x, y, z in vertices:
                f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
            for face in mesh.faces:
                f.write("f " + " ".join(str(index + vertex_offset) for index in face) + "\n")
            vertex_offset += len(vertices)


def _bounds(vertices: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not vertices:
        raise ValueError("Cannot compute bounds from an empty vertex list")
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    zs = [point[2] for point in vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _mesh_part(name: str, mesh: ObjMesh) -> tuple[str, ObjMesh, list[tuple[float, float, float]]]:
    return name, mesh, mesh.vertices


def _distal_finger_surface(mesh: ObjMesh, *, band_m: float) -> ObjMesh:
    """Extract actual terminal finger faces; never replace them with a box."""

    max_x = max(vertex[0] for vertex in mesh.vertices)
    threshold = max_x - float(band_m)
    selected_faces = [
        face
        for face in mesh.faces
        if sum(mesh.vertices[index - 1][0] for index in face) / len(face)
        >= threshold
    ]
    if not selected_faces:
        raise ValueError(
            "Finger mesh has no faces in its strict distal tip band: "
            f"max_x={max_x} band_m={band_m}"
        )

    old_indices = sorted({index for face in selected_faces for index in face})
    remap = {old: new for new, old in enumerate(old_indices, start=1)}
    return ObjMesh(
        [mesh.vertices[index - 1] for index in old_indices],
        [[remap[index] for index in face] for face in selected_faces],
    )


def _transform_part(
    part: tuple[str, ObjMesh, list[tuple[float, float, float]]],
    transform: RigidTransformSpec,
) -> tuple[str, ObjMesh, list[tuple[float, float, float]]]:
    name, mesh, vertices = part
    return name, mesh, _transform_by_spec(vertices, transform)


def _joint_part(
    part: tuple[str, ObjMesh, list[tuple[float, float, float]]],
    joint,
    opening: float,
) -> tuple[str, ObjMesh, list[tuple[float, float, float]]]:
    name, mesh, vertices = part
    return name, mesh, _apply_joint_pose(vertices, joint, opening)


def _quat_wxyz_to_matrix(quat: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _rpy_xyz_to_matrix(rpy: tuple[float, float, float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def _add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(
    point: tuple[float, float, float],
    value: float,
) -> tuple[float, float, float]:
    return (point[0] * value, point[1] * value, point[2] * value)


def _transform_by_spec(
    vertices: list[tuple[float, float, float]],
    transform: RigidTransformSpec,
) -> list[tuple[float, float, float]]:
    rot = _quat_wxyz_to_matrix(transform.quat_wxyz)
    return [_add(_mat_vec(rot, vertex), transform.translation) for vertex in vertices]


def _apply_joint_pose(
    vertices: list[tuple[float, float, float]],
    joint,
    opening: float,
) -> list[tuple[float, float, float]]:
    rot = _rpy_xyz_to_matrix(joint.origin_rpy)
    translation = _add(joint.origin_xyz, _mat_vec(rot, _scale(joint.axis_xyz, opening)))
    return [_add(_mat_vec(rot, vertex), translation) for vertex in vertices]


def _exact_generated_mesh_parts(
    gripper: GeneratedGripperAsset,
    opening: float,
) -> tuple[
    list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
    list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
    str,
]:
    plank = _read_mesh(gripper.plank_mesh)
    finger = _read_mesh(gripper.finger_mesh)

    parts = [
        _transform_part(
            _mesh_part("plank_mesh", plank),
            gripper.mesh_to_body_frame["plank"],
        )
    ]
    finger_part = _transform_part(
        _mesh_part("finger_mesh", finger),
        gripper.mesh_to_body_frame["finger"],
    )
    left_finger = _joint_part(
        (f"left_{finger_part[0]}", finger_part[1], finger_part[2]),
        gripper.finger_joint_local_poses[0],
        opening,
    )
    right_finger = _joint_part(
        (f"right_{finger_part[0]}", finger_part[1], finger_part[2]),
        gripper.finger_joint_local_poses[1],
        opening,
    )
    parts.extend((left_finger, right_finger))

    tip_parts: list[
        tuple[str, ObjMesh, list[tuple[float, float, float]]]
    ] = []
    tip_source = "distal_finger_5mm"
    if gripper.has_tip:
        if gripper.finger_tip_mesh is None or gripper.finger_tip_to_finger_frame is None:
            raise ValueError(f"Generated gripper {gripper.gripper_id} is missing tip mesh metadata")
        tip = _read_mesh(gripper.finger_tip_mesh)
        tip_part = _transform_part(
            _mesh_part("tip_mesh", tip),
            gripper.mesh_to_body_frame["finger_tip"],
        )
        tip_part = _transform_part(tip_part, gripper.finger_tip_to_finger_frame)
        tip_parts = [
            _joint_part(
                (f"left_{tip_part[0]}", tip_part[1], tip_part[2]),
                gripper.finger_joint_local_poses[0],
                opening,
            ),
            _joint_part(
                (f"right_{tip_part[0]}", tip_part[1], tip_part[2]),
                gripper.finger_joint_local_poses[1],
                opening,
            ),
        ]
        parts.extend(tip_parts)
        tip_source = "explicit_tip_mesh"
    else:
        distal = _distal_finger_surface(
            finger,
            band_m=DISTAL_FINGER_TIP_BAND_M,
        )
        distal_part = _transform_part(
            _mesh_part("distal_finger_tip", distal),
            gripper.mesh_to_body_frame["finger"],
        )
        tip_parts = [
            _joint_part(
                (f"left_{distal_part[0]}", distal_part[1], distal_part[2]),
                gripper.finger_joint_local_poses[0],
                opening,
            ),
            _joint_part(
                (f"right_{distal_part[0]}", distal_part[1], distal_part[2]),
                gripper.finger_joint_local_poses[1],
                opening,
            ),
        ]
    if not tip_parts:
        raise ValueError(
            f"Generated gripper {gripper.gripper_id} produced no fingertip geometry"
        )
    return parts, tip_parts, tip_source


def _random_opening(gripper: GeneratedGripperAsset, seed: int) -> float:
    rng = random.Random(f"{int(seed)}:{gripper.gripper_id}")
    fraction = rng.randrange(NUM_OPENING_BINS) / float(NUM_OPENING_BINS - 1)
    return fraction * float(gripper.open_joint_pos)


def _random_opening_fraction(gripper_id: str, seed: int) -> float:
    rng = random.Random(f"{int(seed)}:one_dof:{gripper_id}")
    return rng.randrange(NUM_OPENING_BINS) / float(NUM_OPENING_BINS - 1)


def _polygons_to_mesh(polygons: list) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        base = len(vertices) + 1
        vertices.extend(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in polygon
        )
        faces.append([base + offset for offset in range(len(polygon))])
    if not vertices:
        raise ValueError("One-DoF contact mesh produced no vertices")
    return ObjMesh(vertices, faces)


def _one_dof_mesh_parts(
    entry: dict,
    manifest_dir: Path,
    opening_fraction: float,
) -> tuple[
    list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
    list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
]:
    named_parts = _one_dof_gripper_named_polygons(
        entry,
        opening_fraction,
        manifest_dir,
    )
    body_counts: dict[str, int] = {}
    parts = []
    tip_parts = []
    for body_name, polygons, _ in named_parts:
        visual_index = body_counts.get(body_name, 0)
        body_counts[body_name] = visual_index + 1
        name = f"{body_name}__visual_{visual_index}"
        mesh = _polygons_to_mesh(polygons)
        part = (name, mesh, mesh.vertices)
        parts.append(part)
        if body_name in {"left_top_link", "right_top_link"}:
            tip_parts.append(part)
    if not tip_parts:
        raise ValueError("One-DoF contact mesh has no terminal top-link geometry")
    return parts, tip_parts


def _normalized_head_area(
    parts: list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
    head_parts: list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
) -> list[list[float]]:
    all_vertices = [vertex for _, _, vertices in parts for vertex in vertices]
    head_vertices = [
        vertex for _, _, vertices in head_parts for vertex in vertices
    ]
    if not all_vertices or not head_vertices:
        raise ValueError("Exact generated gripper mesh is missing fingertip geometry")

    all_min, all_max = _bounds(all_vertices)
    head_min, head_max = _bounds(head_vertices)
    lo: list[float] = []
    hi: list[float] = []
    for axis in range(3):
        axis_range = all_max[axis] - all_min[axis]
        if axis_range <= 1e-12:
            lo.append(0.0)
            hi.append(1.0)
            continue
        lo.append(max(0.0, min(1.0, (head_min[axis] - all_min[axis]) / axis_range)))
        hi.append(max(0.0, min(1.0, (head_max[axis] - all_min[axis]) / axis_range)))
    return [lo, hi]


def main() -> None:
    args = _parse_args()
    grippers = load_generated_gripper_manifest(args.manifest)
    mesh_root = args.output_dir / "meshdata_adjusted"
    selected_path = args.output_dir / "tools_selected.json"
    adjusted_path = args.output_dir / "tools_adjusted.json"

    selected: list[str] = []
    adjusted: list[dict] = []
    generated_cache_dir = args.generated_cache_dir.expanduser().resolve()
    for gripper in grippers:
        tool_id = f"generated_gripper_{gripper.gripper_id}"
        opening = _random_opening(gripper, args.opening_seed)
        opening_fraction = opening / gripper.open_joint_pos
        cache_path = generated_cache_dir / f"{gripper.gripper_id}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Generated parallel cloud cache does not exist: {cache_path}"
            )
        load_gripper_cloud_cache(
            cache_path,
            expected_gripper_id=gripper.gripper_id,
            expected_source_manifest=args.manifest,
            expected_source_asset_root=gripper.root_dir,
        )
        output_obj = mesh_root / tool_id / "coacd" / "decomposed.obj"
        parts, tip_parts, tip_source = _exact_generated_mesh_parts(
            gripper,
            opening,
        )
        tip_obj = output_obj.with_name("contact_tip.obj")
        head_area = _normalized_head_area(parts, tip_parts)
        _write_obj(output_obj, parts)
        _write_obj(tip_obj, tip_parts)
        selected.append(tool_id)
        adjusted.append(
            {
                "name": tool_id,
                "head_area": head_area,
                "source_generated_gripper_id": gripper.gripper_id,
                "source_manifest": str(args.manifest),
                "opening": opening,
                "opening_fraction": opening_fraction,
                "opening_seed": args.opening_seed,
                "proxy": "exact_generated_mesh",
                "contact_tip_mesh": str(tip_obj.resolve()),
                "contact_tip_source": tip_source,
                "open_joint_pos": gripper.open_joint_pos,
                "kinematic_cloud_cache": str(cache_path),
            }
        )

    if args.one_dof_manifest is not None:
        one_dof_manifest = args.one_dof_manifest.expanduser().resolve()
        payload = json.loads(one_dof_manifest.read_text(encoding="utf-8"))
        entries = payload.get("grippers")
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"One-DoF manifest has no grippers: {one_dof_manifest}"
            )
        for entry in entries:
            gripper_id = str(entry["id"])
            tool_id = f"one_dof_gripper_{gripper_id}"
            opening_fraction = _random_opening_fraction(
                gripper_id,
                args.opening_seed,
            )
            output_obj = mesh_root / tool_id / "coacd" / "decomposed.obj"
            parts, tip_parts = _one_dof_mesh_parts(
                entry,
                one_dof_manifest.parent,
                opening_fraction,
            )
            tip_obj = output_obj.with_name("contact_tip.obj")
            cache_path = (
                one_dof_manifest.parent
                / "kinematic_cloud_cache"
                / f"{gripper_id}.pt"
            ).resolve()
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"One-DoF kinematic cloud cache does not exist: {cache_path}"
                )
            source_root = Path(str(entry["root_dir"])).expanduser()
            if not source_root.is_absolute():
                source_root = one_dof_manifest.parent / source_root
            load_gripper_cloud_cache(
                cache_path,
                expected_gripper_id=gripper_id,
                expected_source_manifest=one_dof_manifest,
                expected_source_asset_root=source_root,
            )
            head_area = _normalized_head_area(parts, tip_parts)
            _write_obj(output_obj, parts)
            _write_obj(tip_obj, tip_parts)
            selected.append(tool_id)
            adjusted.append(
                {
                    "name": tool_id,
                    "head_area": head_area,
                    "source_one_dof_gripper_id": gripper_id,
                    "source_manifest": str(one_dof_manifest),
                    "opening_fraction": opening_fraction,
                    "opening_seed": args.opening_seed,
                    "proxy": "urdf_visual_geometry",
                    "contact_tip_mesh": str(tip_obj.resolve()),
                    "contact_tip_source": "terminal_top_links",
                    "kinematic_cloud_cache": str(cache_path),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    adjusted_path.write_text(json.dumps(adjusted, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} generated gripper tools")
    print(f"meshdata_adjusted_root={mesh_root}")
    print(f"tools_selected_json={selected_path}")
    print(f"tools_adjusted_json={adjusted_path}")


if __name__ == "__main__":
    main()

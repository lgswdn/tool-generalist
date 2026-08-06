#!/usr/bin/env python3
"""Export canonical Panda gripper cloud/mesh OBJ files for inspection.

This is a pure-Python debug script. It mirrors the current official Panda
gripper policy-cloud construction without importing Isaac/IsaacLab.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROPS_DIR = Path("/mnt/project/world_model/tool_generalist/eef_panda/Robots/Props")
DEFAULT_OUTPUT_DIR = Path("debug_outputs/panda_gripper_cloud_objs")
DEFAULT_BUCKETS = "0,16,32,63"
NUM_BUCKETS = 64
OPEN_JOINT_POS = 0.04
FINGER_MOUNT_OFFSET_Y = 0.0584
FRAME_CORRECTION_NAME = "official_hand_frame_rx90"


@dataclass
class ObjMesh:
    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]


def parse_obj(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    raw = token.split("/", 1)[0]
                    idx = int(raw)
                    indices.append(len(vertices) + idx + 1 if idx < 0 else idx)
                if len(indices) >= 3:
                    faces.append(indices)
    if not vertices:
        raise RuntimeError(f"OBJ has no vertices: {path}")
    return ObjMesh(vertices=vertices, faces=faces)


def offset_vertices(
    vertices: list[tuple[float, float, float]],
    offset: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    ox, oy, oz = offset
    return [(x + ox, y + oy, z + oz) for x, y, z in vertices]


def mesh_to_official_hand_frame(
    vertices: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    return [(x, -z, y) for x, y, z in vertices]


def mesh_bounds(vertices: list[tuple[float, float, float]]) -> tuple[list[float], list[float], list[float]]:
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    extents = [maxs[i] - mins[i] for i in range(3)]
    return mins, maxs, extents


def write_combined_mesh_obj(
    path: Path,
    parts: list[tuple[str, ObjMesh, list[tuple[float, float, float]]]],
    *,
    include_axes: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offset = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("# Canonical Panda gripper mesh bucket exported for debug.\n")
        f.write("# Frame axes are OBJ line elements: +X, +Y, +Z each length 0.12m.\n")
        for name, mesh, vertices in parts:
            f.write(f"o {name}\n")
            for x, y, z in vertices:
                f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
            for face in mesh.faces:
                shifted = [str(idx + vertex_offset) for idx in face]
                f.write("f " + " ".join(shifted) + "\n")
            vertex_offset += len(vertices)
        if include_axes:
            f.write("o frame_axes\n")
            axis_vertices = [(0.0, 0.0, 0.0), (0.12, 0.0, 0.0), (0.0, 0.12, 0.0), (0.0, 0.0, 0.12)]
            for x, y, z in axis_vertices:
                f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
            base = vertex_offset + 1
            f.write(f"l {base} {base + 1}\n")
            f.write(f"l {base} {base + 2}\n")
            f.write(f"l {base} {base + 3}\n")


def deterministic_resample(points: list[tuple[float, float, float]], target_count: int) -> list[tuple[float, float, float]]:
    if target_count <= 0:
        return []
    if len(points) == target_count:
        return points
    if len(points) > target_count:
        if target_count == 1:
            return [points[0]]
        return [points[round(i * (len(points) - 1) / (target_count - 1))] for i in range(target_count)]
    repeats = target_count // len(points) + 1
    return (points * repeats)[:target_count]


def write_pointcloud_obj(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Canonical Panda gripper policy point cloud exported for debug.\n")
        f.write("# OBJ point elements are included; some viewers hide point primitives by default.\n")
        f.write("o policy_pointcloud\n")
        for x, y, z in points:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for idx in range(1, len(points) + 1):
            f.write(f"p {idx}\n")


def parse_buckets(raw: str) -> list[int]:
    buckets = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value >= NUM_BUCKETS:
            raise argparse.ArgumentTypeError(f"bucket must be in [0, {NUM_BUCKETS - 1}]: {value}")
        buckets.append(value)
    if not buckets:
        raise argparse.ArgumentTypeError("at least one bucket is required")
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Panda gripper canonical OBJ debug files.")
    parser.add_argument("--props-dir", type=Path, default=DEFAULT_PROPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--buckets", type=parse_buckets, default=parse_buckets(DEFAULT_BUCKETS))
    args = parser.parse_args()

    if args.num_points <= 0:
        parser.error("--num-points must be positive")

    props_dir = args.props_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = {
        "panda_hand": props_dir / "panda_hand.obj",
        "panda_leftfinger": props_dir / "panda_leftfinger.obj",
        "panda_rightfinger": props_dir / "panda_rightfinger.obj",
        "panda_hand_with_fingers_reference": props_dir / "panda_hand_with_fingers.obj",
    }
    for name, path in raw_paths.items():
        if path.is_file():
            shutil.copy2(path, output_dir / f"raw_{name}.obj")

    hand = parse_obj(raw_paths["panda_hand"])
    left = parse_obj(raw_paths["panda_leftfinger"])
    right = parse_obj(raw_paths["panda_rightfinger"])

    hand_count = args.num_points // 2
    left_count = (args.num_points - hand_count) // 2
    right_count = args.num_points - hand_count - left_count
    hand_points = deterministic_resample(hand.vertices, hand_count)
    left_points = deterministic_resample(left.vertices, left_count)
    right_points = deterministic_resample(right.vertices, right_count)

    print(f"props_dir={props_dir}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"num_points={args.num_points} counts hand/left/right={hand_count}/{left_count}/{right_count}")

    for bucket_id in args.buckets:
        opening = OPEN_JOINT_POS * bucket_id / float(NUM_BUCKETS - 1)
        left_offset = (0.0, FINGER_MOUNT_OFFSET_Y, -opening)
        right_offset = (0.0, FINGER_MOUNT_OFFSET_Y, opening)

        left_vertices = offset_vertices(left.vertices, left_offset)
        right_vertices = offset_vertices(right.vertices, right_offset)
        combined_vertices = hand.vertices + left_vertices + right_vertices
        mins, maxs, extents = mesh_bounds(combined_vertices)
        suffix = f"bucket_{bucket_id:02d}_opening_{opening:.5f}"

        write_combined_mesh_obj(
            output_dir / f"current_kinematic_mesh_{suffix}.obj",
            [
                ("panda_hand_raw_frame", hand, hand.vertices),
                ("panda_leftfinger_offset", left, left_vertices),
                ("panda_rightfinger_offset", right, right_vertices),
            ],
        )
        corrected_hand_vertices = mesh_to_official_hand_frame(hand.vertices)
        corrected_left_vertices = mesh_to_official_hand_frame(left_vertices)
        corrected_right_vertices = mesh_to_official_hand_frame(right_vertices)
        corrected_combined = corrected_hand_vertices + corrected_left_vertices + corrected_right_vertices
        corrected_mins, corrected_maxs, corrected_extents = mesh_bounds(corrected_combined)
        write_combined_mesh_obj(
            output_dir / f"corrected_{FRAME_CORRECTION_NAME}_mesh_{suffix}.obj",
            [
                ("panda_hand_official_hand_frame", hand, corrected_hand_vertices),
                ("panda_leftfinger_official_hand_frame", left, corrected_left_vertices),
                ("panda_rightfinger_official_hand_frame", right, corrected_right_vertices),
            ],
        )

        cloud_points = (
            hand_points
            + offset_vertices(left_points, left_offset)
            + offset_vertices(right_points, right_offset)
        )
        write_pointcloud_obj(output_dir / f"current_policy_pointcloud_{suffix}.obj", cloud_points)
        write_pointcloud_obj(
            output_dir / f"corrected_{FRAME_CORRECTION_NAME}_policy_pointcloud_{suffix}.obj",
            mesh_to_official_hand_frame(cloud_points),
        )

        print(
            f"bucket={bucket_id:02d} opening={opening:.5f} "
            f"bbox_min={[round(v, 6) for v in mins]} "
            f"bbox_max={[round(v, 6) for v in maxs]} "
            f"bbox_extent={[round(v, 6) for v in extents]}"
        )
        print(
            f"  corrected={FRAME_CORRECTION_NAME} "
            f"bbox_min={[round(v, 6) for v in corrected_mins]} "
            f"bbox_max={[round(v, 6) for v in corrected_maxs]} "
            f"bbox_extent={[round(v, 6) for v in corrected_extents]}"
        )


if __name__ == "__main__":
    main()

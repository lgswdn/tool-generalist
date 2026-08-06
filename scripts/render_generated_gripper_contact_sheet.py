#!/usr/bin/env python3
"""Render generated grippers directly from their manifest meshes.

This intentionally avoids Isaac/Kit rendering so it also works on headless
machines where viewport capture may hang or return black frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tool_generalist_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _transform(vertices: np.ndarray, xyz: list[float], rpy: list[float]) -> np.ndarray:
    return vertices @ _rpy_matrix(rpy).T + np.asarray(xyz, dtype=float)


def _pose_matrix(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _axis_angle_matrix(axis: list[float], angle: float) -> np.ndarray:
    axis_np = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_np)
    if norm <= 1.0e-12 or abs(angle) <= 1.0e-12:
        return np.eye(4, dtype=float)
    x, y, z = axis_np / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.asarray(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    return transform


def _apply_pose(polygons: list[np.ndarray], pose: np.ndarray) -> list[np.ndarray]:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return [polygon @ rotation.T + translation for polygon in polygons]


def _box_polygons(size: list[float]) -> list[np.ndarray]:
    half = np.asarray(size, dtype=float) * 0.5
    vertices = np.asarray(
        [
            (sx * half[0], sy * half[1], sz * half[2])
            for sx, sy, sz in (
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            )
        ]
    )
    faces = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    return [vertices[np.asarray(face)] for face in faces]


def _cylinder_polygons(radius: float, length: float, segments: int = 24) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    bottom_center = np.asarray((0.0, 0.0, -length * 0.5))
    top_center = np.asarray((0.0, 0.0, length * 0.5))
    for index in range(segments):
        a0 = 2.0 * math.pi * index / segments
        a1 = 2.0 * math.pi * (index + 1) / segments
        b0 = np.asarray((radius * math.cos(a0), radius * math.sin(a0), -length * 0.5))
        b1 = np.asarray((radius * math.cos(a1), radius * math.sin(a1), -length * 0.5))
        t0 = b0.copy()
        t1 = b1.copy()
        t0[2] = length * 0.5
        t1[2] = length * 0.5
        polygons.append(np.asarray((b0, b1, t1, t0)))
        polygons.append(np.asarray((bottom_center, b1, b0)))
        polygons.append(np.asarray((top_center, t0, t1)))
    return polygons


def _origin_values(node: ET.Element | None) -> tuple[list[float], list[float]]:
    if node is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = [float(value) for value in node.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in node.get("rpy", "0 0 0").split()]
    return xyz, rpy


def _load_obj(path: Path) -> list[np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
    points = np.asarray(vertices, dtype=float)
    return [points[np.asarray(face, dtype=int)] for face in faces]


def _load_stl(path: Path) -> list[np.ndarray]:
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + count * 50 == len(raw):
            triangles = []
            for index in range(count):
                values = struct.unpack_from("<12fH", raw, 84 + index * 50)
                triangles.append(np.asarray(values[3:12], dtype=float).reshape(3, 3))
            return triangles

    vertices = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0] == "vertex":
            vertices.append([float(value) for value in fields[1:]])
    return [np.asarray(vertices[i : i + 3], dtype=float) for i in range(0, len(vertices), 3)]


def _load_mesh(path: Path) -> list[np.ndarray]:
    if path.suffix.lower() == ".obj":
        return _load_obj(path)
    if path.suffix.lower() == ".stl":
        return _load_stl(path)
    raise ValueError(f"Unsupported mesh format: {path}")


def _joint_poses(urdf_path: Path, opening: float) -> dict[str, tuple[list[float], list[float]]]:
    root = ET.parse(urdf_path).getroot()
    poses: dict[str, tuple[list[float], list[float]]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.get("link") not in {"panda_leftfinger", "panda_rightfinger"}:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = [float(value) for value in (origin.get("xyz", "0 0 0").split())]
        rpy = [float(value) for value in (origin.get("rpy", "0 0 0").split())]
        axis_xyz = [float(value) for value in (axis.get("xyz", "0 0 0").split())]
        displacement = _rpy_matrix(rpy) @ (np.asarray(axis_xyz) * opening)
        xyz = (np.asarray(xyz) + displacement).tolist()
        poses[child.get("link")] = (xyz, rpy)
    return poses


def _legacy_gripper_polygons(entry: dict, opening_fraction: float) -> list[tuple[list[np.ndarray], str]]:
    root = Path(entry["root_dir"])
    mesh_dir = root / entry["mesh_dir"]
    opening = float(entry.get("open_joint_pos", 0.04)) * opening_fraction
    poses = _joint_poses(root / entry["urdf_path"], opening)

    result: list[tuple[list[np.ndarray], str]] = [
        (_load_mesh(mesh_dir / entry["plank_mesh"]), "#777777")
    ]
    finger_meshes = [entry["finger_mesh"]]
    if entry.get("has_tip") and entry.get("finger_tip_mesh"):
        finger_meshes.append(entry["finger_tip_mesh"])
    for link, color in (("panda_leftfinger", "#3b82f6"), ("panda_rightfinger", "#ef4444")):
        xyz, rpy = poses[link]
        for mesh_name in finger_meshes:
            polygons = [
                _transform(polygon, xyz, rpy) for polygon in _load_mesh(mesh_dir / mesh_name)
            ]
            result.append((polygons, color))
    return result


def _one_dof_gripper_named_polygons(
    entry: dict,
    opening_fraction: float,
    manifest_dir: Path,
) -> list[tuple[str, list[np.ndarray], str]]:
    root_dir = Path(entry["root_dir"])
    if not root_dir.is_absolute():
        root_dir = manifest_dir / root_dir
    root = ET.parse(root_dir / entry["urdf_path"]).getroot()
    links = {link.get("name"): link for link in root.findall("link")}

    control = entry["control"]
    names = control["actuated_joint_names"]
    opened = control["open_joint_positions"]
    closed = control["closed_joint_positions"]
    joint_positions = {
        name: float(closed_value)
        + opening_fraction * (float(open_value) - float(closed_value))
        for name, open_value, closed_value in zip(names, opened, closed)
    }

    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None:
            children.setdefault(parent.get("link"), []).append(joint)

    body_poses: dict[str, np.ndarray] = {"gripper_palm": np.eye(4, dtype=float)}
    pending = ["gripper_palm"]
    while pending:
        parent_name = pending.pop()
        for joint in children.get(parent_name, []):
            child = joint.find("child")
            if child is None:
                continue
            child_name = child.get("link")
            xyz, rpy = _origin_values(joint.find("origin"))
            pose = body_poses[parent_name] @ _pose_matrix(xyz, rpy)
            if joint.get("type") in {"revolute", "continuous"}:
                axis_node = joint.find("axis")
                axis = (
                    [float(value) for value in axis_node.get("xyz", "1 0 0").split()]
                    if axis_node is not None
                    else [1.0, 0.0, 0.0]
                )
                pose = pose @ _axis_angle_matrix(axis, joint_positions.get(joint.get("name"), 0.0))
            body_poses[child_name] = pose
            pending.append(child_name)

    result: list[tuple[str, list[np.ndarray], str]] = []
    for body_name, body_pose in body_poses.items():
        link = links.get(body_name)
        if link is None:
            continue
        color = "#777777"
        if body_name.startswith("finger_1_"):
            color = "#3b82f6"
        elif body_name.startswith("finger_2_"):
            color = "#ef4444"
        elif body_name.startswith("finger_3_"):
            color = "#22a06b"
        elif body_name.startswith("left_"):
            color = "#3b82f6"
        elif body_name.startswith("right_"):
            color = "#ef4444"
        for visual in link.findall("visual"):
            xyz, rpy = _origin_values(visual.find("origin"))
            geometry_pose = body_pose @ _pose_matrix(xyz, rpy)
            box = visual.find("geometry/box")
            cylinder = visual.find("geometry/cylinder")
            mesh = visual.find("geometry/mesh")
            if box is not None:
                polygons = _box_polygons([float(value) for value in box.get("size").split()])
            elif cylinder is not None:
                polygons = _cylinder_polygons(
                    float(cylinder.get("radius")),
                    float(cylinder.get("length")),
                )
            elif mesh is not None:
                mesh_path = Path(mesh.get("filename"))
                if not mesh_path.is_absolute():
                    mesh_path = root_dir / mesh_path
                polygons = _load_mesh(mesh_path)
            else:
                continue
            result.append((body_name, _apply_pose(polygons, geometry_pose), color))
    return result


def _one_dof_gripper_polygons(
    entry: dict,
    opening_fraction: float,
    manifest_dir: Path,
) -> list[tuple[list[np.ndarray], str]]:
    return [
        (polygons, color)
        for _, polygons, color in _one_dof_gripper_named_polygons(
            entry, opening_fraction, manifest_dir
        )
    ]


def _gripper_polygons(
    entry: dict,
    opening_fraction: float,
    manifest_dir: Path,
) -> list[tuple[list[np.ndarray], str]]:
    if "control" in entry:
        return _one_dof_gripper_polygons(entry, opening_fraction, manifest_dir)
    return _legacy_gripper_polygons(entry, opening_fraction)


def _set_equal_limits(axis, points: np.ndarray) -> None:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float((high - low).max()) * 0.58, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _render_page(
    entries: list[dict],
    output: Path,
    opening: float,
    manifest_dir: Path,
) -> None:
    columns = min(4, len(entries))
    rows = math.ceil(len(entries) / columns)
    figure = plt.figure(figsize=(4 * columns, 4 * rows), facecolor="white")

    for plot_index, entry in enumerate(entries, start=1):
        axis = figure.add_subplot(rows, columns, plot_index, projection="3d")
        all_points = []
        for polygons, color in _gripper_polygons(entry, opening, manifest_dir):
            if not polygons:
                continue
            # Large generated meshes remain clear with a bounded face sample.
            stride = max(1, len(polygons) // 2500)
            shown = polygons[::stride]
            axis.add_collection3d(
                Poly3DCollection(shown, facecolor=color, edgecolor="none", alpha=0.9)
            )
            all_points.extend(shown)
        _set_equal_limits(axis, np.concatenate(all_points, axis=0))
        axis.view_init(elev=22, azim=-52)
        if "control" in entry:
            params = entry["params"]
            if params.get("family") == "three_finger_high_dof":
                lengths = params["finger_lengths"]
                closed_deg = math.degrees(float(params["closed_angle_rad"]))
                axis.set_title(
                    f"{entry['id'].rsplit('_', 1)[-1]}  3×3 joints\n"
                    f"L={min(lengths)*1000:.0f}–{max(lengths)*1000:.0f} mm  "
                    f"close={closed_deg:.0f}°/joint"
                )
            else:
                open_deg = -math.degrees(float(params.get("open_angle_rad", 0.0)))
                closed_deg = math.degrees(float(params.get("closed_angle_rad", 0.0)))
                mode = params.get("closure_mode", "unknown").replace("_tip", "")
                ending = f"{mode}  tip={params.get('tip_shape', 'none')}"
                axis.set_title(
                    f"{entry['id'].rsplit('_', 1)[-1]}\n"
                    f"open={open_deg:.0f}° out  closed={closed_deg:.0f}° in  {ending}"
                )
        else:
            params_path = Path(entry["root_dir"]) / entry.get("params_path", "params.json")
            params = json.loads(params_path.read_text(encoding="utf-8"))
            length_mm = 1000.0 * float(params["finger_length"])
            thickness_mm = 1000.0 * float(params["finger_thickness"])
            axis.set_title(
                f"gripper {entry['id']}\nL={length_mm:.1f} mm  T={thickness_mm:.1f} mm"
            )
        axis.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="videos/generated_gripper_new/contact_sheet.png")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--per-page", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening", type=float, default=1.0, help="Opening fraction in [0, 1].")
    args = parser.parse_args()
    if args.num <= 0:
        parser.error("--num must be positive")
    if args.per_page <= 0:
        parser.error("--per-page must be positive")
    if not 0.0 <= args.opening <= 1.0:
        parser.error("--opening must be in [0, 1]")

    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload["grippers"]
    rng = random.Random(args.seed)
    selected = rng.sample(entries, min(args.num, len(entries)))
    output = Path(args.output).expanduser().resolve()
    pages = [selected[i : i + args.per_page] for i in range(0, len(selected), args.per_page)]
    for page_index, page in enumerate(pages, start=1):
        page_output = (
            output
            if len(pages) == 1
            else output.with_name(f"{output.stem}_page_{page_index:02d}{output.suffix}")
        )
        _render_page(page, page_output, args.opening, manifest_path.parent)


if __name__ == "__main__":
    main()

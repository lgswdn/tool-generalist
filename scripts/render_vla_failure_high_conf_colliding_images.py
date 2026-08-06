#!/usr/bin/env python3
"""Render PNGs for VLA-failure objects with only high-confidence colliding grasps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

import check_graspgen_episode_consistency as common


DEFAULT_REPORT = Path("scripts/outputs/graspgen_episode_consistency_full_1024_top256.jsonl")
DEFAULT_MESH_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")
DEFAULT_GRASPGEN_ROOT = Path("/mnt/project/world_model/tool_generalist/GraspGen")
DEFAULT_OUTPUT = Path("scripts/outputs/vla_failure_high_conf_colliding_images")
COLLISION_KEYS = (
    "panda_hand_object_collision",
    "panda_hand_ground_collision",
    "panda_fingers_object_collision",
    "panda_fingers_ground_collision",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-jsonl", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--graspgen-root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--grasps-per-image", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _camera_axes(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ...]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def _load_gripper(root: Path) -> tuple[Any, Any]:
    hand = trimesh.load(root / "assets/panda_gripper/hand.stl", force="mesh", process=False)
    left = trimesh.load(root / "assets/panda_gripper/finger.stl", force="mesh", process=False)
    right = left.copy()
    rotate = np.eye(4)
    rotate[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    left.apply_transform(rotate)
    left.apply_translation([0.04, 0.0, 0.0584])
    right.apply_translation([-0.04, 0.0, 0.0584])
    return hand, trimesh.util.concatenate([left, right])


def _colliding(candidate: dict[str, Any], threshold: float) -> bool:
    return (
        float(candidate.get("confidence", float("-inf"))) >= threshold
        and bool(candidate.get("valid_se3"))
        and any(bool(candidate.get(key)) for key in COLLISION_KEYS)
    )


def _strict_free(candidate: dict[str, Any], threshold: float) -> bool:
    return (
        float(candidate.get("confidence", float("-inf"))) >= threshold
        and bool(candidate.get("hand_and_fingers_collision_free"))
    )


def _collision_label(candidate: dict[str, Any]) -> str:
    labels = []
    if candidate["panda_hand_object_collision"]:
        labels.append("hand-object")
    if candidate["panda_hand_ground_collision"]:
        labels.append("hand-ground")
    if candidate["panda_fingers_object_collision"]:
        labels.append("fingers-object")
    if candidate["panda_fingers_ground_collision"]:
        labels.append("fingers-ground")
    return "+".join(labels)


def _candidate_colors(candidate: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    hand_collision = bool(candidate["panda_hand_object_collision"]) or bool(
        candidate["panda_hand_ground_collision"]
    )
    finger_object = bool(candidate["panda_fingers_object_collision"])
    finger_ground = bool(candidate["panda_fingers_ground_collision"])
    guide = (
        (245, 45, 45, 255)
        if hand_collision
        else (250, 190, 20, 255)
        if finger_object
        else (250, 95, 20, 255)
    )
    hand = (245, 45, 45, 115) if hand_collision else (35, 205, 235, 105)
    fingers = (
        (250, 190, 20, 150)
        if finger_object
        else (250, 95, 20, 150)
        if finger_ground
        else (45, 105, 245, 105)
    )
    return guide, hand, fingers


def _render(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    object_mesh: Any,
    hand_mesh: Any,
    finger_mesh: Any,
    width: int,
    height: int,
    threshold: float,
) -> Image.Image:
    meshes: list[tuple[Any, np.ndarray, tuple[int, int, int, int], bool]] = []
    object_pose = np.eye(4)
    object_pose[:3, :3] = common._quat_wxyz_matrix(
        np.asarray(row["orientation_wxyz"], dtype=np.float64)
    ) * float(row["scale"])
    object_pose[:3, 3] = np.asarray(row["position"], dtype=np.float64)
    meshes.append((object_mesh, object_pose, (190, 78, 65, 255), True))

    support_z = float(row["collision"]["support_top_z"])
    for candidate in candidates:
        matrix = np.asarray(candidate["grasp_matrix_world"], dtype=np.float64)
        guide_color, hand_color, finger_color = _candidate_colors(candidate)
        for guide in common._grasp_guide_meshes(
            matrix, width=0.08, depth=0.10, thickness=0.0025
        ):
            vertices, faces = guide
            meshes.append(
                (
                    trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                    np.eye(4), guide_color, True,
                )
            )
        meshes.append((hand_mesh, matrix, hand_color, True))
        meshes.append((finger_mesh, matrix, finger_color, True))

    world_meshes: list[
        tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], bool]
    ] = []
    focus_points: list[np.ndarray] = []
    for mesh, pose, color, focus in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) > 6000:
            faces = faces[np.linspace(0, len(faces) - 1, 6000, dtype=np.int64)]
        world_meshes.append((vertices, faces, color, focus))
        if focus:
            focus_points.append(vertices)
    focus_vertices = np.concatenate(focus_points, axis=0)
    bounds = np.stack([focus_vertices.min(axis=0), focus_vertices.max(axis=0)])
    target = bounds.mean(axis=0)
    target[2] = max(target[2], support_z + 0.03)
    extent = max(float(np.max(bounds[1] - bounds[0])), 0.12)
    distance = max(0.38, extent * 2.9)
    direction = np.asarray([1.25, -1.45, 1.05], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eye = target + direction * distance
    right, up, forward = _camera_axes(eye, target)
    focal = (height - 90) * 0.5 / math.tan(math.radians(48.0) * 0.5)
    center_x, center_y = width * 0.5, (height + 70) * 0.5
    light = np.asarray([0.35, -0.55, 0.76], dtype=np.float64)
    light /= np.linalg.norm(light)
    triangles: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    for vertices, faces, color, focus in world_meshes:
        relative = vertices - eye
        camera = np.column_stack(
            [relative @ right, relative @ up, relative @ forward]
        )
        face_camera = camera[faces]
        valid = np.all(face_camera[:, :, 2] > 0.01, axis=1)
        if not np.any(valid):
            continue
        chosen_faces = faces[valid]
        chosen_camera = face_camera[valid]
        projected_x = center_x + focal * chosen_camera[:, :, 0] / chosen_camera[:, :, 2]
        projected_y = center_y - focal * chosen_camera[:, :, 1] / chosen_camera[:, :, 2]
        face_world = vertices[chosen_faces]
        normals = np.cross(face_world[:, 1] - face_world[:, 0], face_world[:, 2] - face_world[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-12)
        shade = 0.58 + 0.42 * np.abs(normals @ light)
        alpha = float(color[3]) / 255.0
        base = np.asarray(color[:3], dtype=np.float64) * alpha + 248.0 * (1.0 - alpha)
        for index in range(len(chosen_faces)):
            rgb = tuple(np.clip(base * shade[index], 0, 255).astype(np.uint8).tolist())
            points = list(zip(projected_x[index].tolist(), projected_y[index].tolist()))
            depth = float(chosen_camera[index, :, 2].mean())
            triangles.append((depth if focus else depth + 1000.0, points, rgb))
    triangles.sort(key=lambda item: item[0], reverse=True)
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image, "RGBA")
    ground_radius = max(0.16, extent * 0.9)
    ground_corners = np.asarray(
        [
            [target[0] - ground_radius, target[1] - ground_radius, support_z],
            [target[0] + ground_radius, target[1] - ground_radius, support_z],
            [target[0] + ground_radius, target[1] + ground_radius, support_z],
            [target[0] - ground_radius, target[1] + ground_radius, support_z],
        ]
    )
    ground_camera = np.column_stack(
        [
            (ground_corners - eye) @ right,
            (ground_corners - eye) @ up,
            (ground_corners - eye) @ forward,
        ]
    )
    ground_points = list(
        zip(
            (center_x + focal * ground_camera[:, 0] / ground_camera[:, 2]).tolist(),
            (center_y - focal * ground_camera[:, 1] / ground_camera[:, 2]).tolist(),
        )
    )
    draw.polygon(ground_points, fill=(225, 225, 225, 255), outline=(145, 145, 145, 255))
    for _, points, color in triangles:
        draw.polygon(points, fill=(*color, 255), outline=(35, 35, 35, 38))
    draw.rectangle((0, 0, width, 70), fill=(255, 255, 255, 225))
    draw.text((12, 8), str(row["object"]), fill=(10, 10, 10, 255))
    details = "; ".join(
        f"r{int(item['rank'])} {float(item['confidence']):.3f} {_collision_label(item)}"
        for item in candidates
    )
    draw.text((12, 29), f"VLA failed | no strict-free confidence >= {threshold:g}", fill=(35, 35, 35, 255))
    draw.text((12, 49), details, fill=(65, 65, 65, 255))
    return image


def main() -> int:
    args = _args()
    if args.grasps_per_image <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("grasp and image dimensions must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    hand, fingers = _load_gripper(args.graspgen_root.expanduser().resolve())
    rendered = 0
    matched = 0
    with args.report_jsonl.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row["episode_success"]):
                continue
            candidates = row["candidates"]
            if any(_strict_free(item, args.min_confidence) for item in candidates):
                continue
            colliding = [
                item for item in candidates if _colliding(item, args.min_confidence)
            ]
            if not colliding:
                continue
            matched += 1
            selected = colliding[: args.grasps_per_image]
            path = output / (
                f"task_{int(row['task_index']):06d}_{common._safe_filename(str(row['object']))}.png"
            )
            if path.exists() and not args.overwrite:
                rendered += 1
            else:
                mesh = trimesh.load(
                    args.mesh_dir.expanduser().resolve() / f"{row['object']}.obj",
                    force="mesh",
                    process=False,
                )
                image = _render(
                    row, selected, mesh, hand, fingers,
                    args.width, args.height, args.min_confidence,
                )
                image.save(path, format="PNG", optimize=True)
                rendered += 1
            if rendered % 25 == 0:
                print(f"rendered {rendered}", flush=True)
            if args.limit is not None and rendered >= args.limit:
                break
    print(f"[DONE] matched={matched} rendered={rendered} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

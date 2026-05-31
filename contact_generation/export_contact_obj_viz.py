#!/usr/bin/env python3
"""Export sampled contact_pt_env_v1 pre/post contact states as colored OBJ files.

Each exported OBJ contains five groups: floor, pre object, post object, pre tool,
and post tool.  Mesh reconstruction follows the training loader contract:
source OBJ vertices are scaled, shifted by the persisted mesh bbox center, then
posed with the persisted env-frame contact poses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

import numpy as np

from utils.geometry.mesh_io import load_mesh_vertices_faces, scale_vertices
from utils.geometry.pose import rotation_from_pose9d_np


BLOCKED_PT_SUFFIXES = (
    ".candidate.pt",
    ".physics_debug.pt",
    ".stabilized_success.pt",
    ".stabilized.pt",
)

MATERIALS = {
    "floor": (0.50, 0.50, 0.50),
    "pre_object": (0.18, 0.42, 0.86),
    "post_object": (0.10, 0.68, 0.35),
    "pre_tool": (0.94, 0.48, 0.12),
    "post_tool": (0.80, 0.18, 0.72),
}


@dataclass(frozen=True)
class PtEntry:
    path: str
    num_contacts: int
    tool_id: str
    object_id: str


@dataclass(frozen=True)
class ExportTask:
    output_index: int
    pt_path: str
    contact_index: int
    obj_path: str
    floor_z: float
    floor_margin: float
    floor_min_size: float
    tool_post_delta_field: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool_name", help="Tool family/name to export, e.g. ball_peen_hammer_end_effector.")
    parser.add_argument("num_outputs", type=int, help="Number of sampled OBJ scenes to export.")
    parser.add_argument("--data-dir", required=True, help="Root contact dataset/artifact directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where OBJ/MTL files will be written.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="CPU worker processes.",
    )
    parser.add_argument(
        "--weighted-by-contacts",
        action="store_true",
        help="Sample contact cases uniformly instead of first sampling a .pt uniformly.",
    )
    parser.add_argument(
        "--tool-post-delta-field",
        choices=("post_tool_achieved_delta_pose9d_E", "post_tool_delta_pose9d_E"),
        default="post_tool_achieved_delta_pose9d_E",
        help="Field used to place the post-contact tool mesh.",
    )
    parser.add_argument("--floor-z", type=float, default=0.0, help="Floor plane z coordinate.")
    parser.add_argument("--floor-margin", type=float, default=0.20, help="Extra XY margin around all meshes.")
    parser.add_argument("--floor-min-size", type=float, default=0.60, help="Minimum square floor side length.")
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=25,
        help="Print scan/metadata progress every N files. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_outputs) <= 0:
        raise ValueError("num_outputs must be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")

    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[export_contact_obj_viz] scanning "
        f"data_dir={data_dir} tool_name={args.tool_name} workers={int(args.workers)}",
        flush=True,
    )
    entries = collect_tool_entries(data_dir, args.tool_name, progress_every=int(args.scan_progress_every))
    if not entries:
        raise RuntimeError(f"No training .pt files found for tool {args.tool_name!r} under {data_dir}")
    print(
        "[export_contact_obj_viz] matched "
        f"pt_files={len(entries)} contacts={sum(entry.num_contacts for entry in entries)}",
        flush=True,
    )

    tasks = sample_tasks(
        entries,
        output_dir=output_dir,
        num_outputs=int(args.num_outputs),
        seed=int(args.seed),
        floor_z=float(args.floor_z),
        floor_margin=float(args.floor_margin),
        floor_min_size=float(args.floor_min_size),
        weighted_by_contacts=bool(args.weighted_by_contacts),
        tool_post_delta_field=str(args.tool_post_delta_field),
    )

    manifest: dict[str, Any] = {
        "schema_version": "contact_obj_viz_manifest_v1",
        "data_dir": str(data_dir.resolve()),
        "tool_name": str(args.tool_name),
        "num_outputs": int(args.num_outputs),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "tool_post_delta_field": str(args.tool_post_delta_field),
        "weighted_by_contacts": bool(args.weighted_by_contacts),
        "outputs": [],
    }

    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(export_one, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            manifest["outputs"].append(result)
            completed += 1
            print(
                "[export_contact_obj_viz] "
                f"{completed}/{len(tasks)} wrote {Path(task.obj_path).name}",
                flush=True,
            )

    manifest["outputs"] = sorted(manifest["outputs"], key=lambda item: int(item["output_index"]))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[export_contact_obj_viz] wrote manifest {manifest_path}", flush=True)
    return 0


def collect_tool_entries(data_dir: Path, tool_name: str, *, progress_every: int = 25) -> list[PtEntry]:
    target = normalize_tool_name(tool_name)
    candidates, filtered_by_dir = collect_candidate_pt_paths(data_dir, target, progress_every=progress_every)
    if filtered_by_dir:
        print(f"[export_contact_obj_viz] candidate training pt_files={len(candidates)} from matched variant dirs", flush=True)
    else:
        print(f"[export_contact_obj_viz] candidate training pt_files={len(candidates)} from full recursive scan", flush=True)

    entries: list[PtEntry] = []
    total = len(candidates)
    for idx, pt_path in enumerate(candidates, start=1):
        try:
            meta = load_pt_metadata(pt_path)
        except Exception as exc:
            print(f"[export_contact_obj_viz] warning: skip unreadable {pt_path}: {exc}", flush=True)
            continue
        if int(meta["num_contacts"]) <= 0:
            continue
        tool_id = str(meta["tool_id"])
        if filtered_by_dir or tool_id_matches(tool_id, target):
            entries.append(
                PtEntry(
                    path=str(pt_path),
                    num_contacts=int(meta["num_contacts"]),
                    tool_id=tool_id,
                    object_id=str(meta["object_id"]),
                )
            )
        if progress_every > 0 and (idx == total or idx % progress_every == 0):
            print(
                "[export_contact_obj_viz] metadata "
                f"{idx}/{total} scanned matched_pt_files={len(entries)}",
                flush=True,
            )
    return sorted(entries, key=lambda entry: entry.path)


def collect_candidate_pt_paths(
    data_dir: Path,
    normalized_tool_name: str,
    *,
    progress_every: int = 25,
) -> tuple[list[Path], bool]:
    variant_dirs: list[Path] = []
    if variant_dir_matches(data_dir, normalized_tool_name):
        variant_dirs.append(data_dir)
    else:
        for child in sorted(data_dir.iterdir()):
            if child.is_dir() and variant_dir_matches(child, normalized_tool_name):
                variant_dirs.append(child)

    if variant_dirs:
        print(
            "[export_contact_obj_viz] matched variant dirs "
            f"count={len(variant_dirs)} first={variant_dirs[0].name}",
            flush=True,
        )
        paths: list[Path] = []
        for variant_dir in variant_dirs:
            paths.extend(
                path
                for path in sorted(variant_dir.glob("*.pt"))
                if path.is_file() and not any(str(path).endswith(suffix) for suffix in BLOCKED_PT_SUFFIXES)
            )
        return paths, True

    print(
        "[export_contact_obj_viz] warning: no variant directory matched; falling back to full recursive scan",
        flush=True,
    )
    return collect_training_pt_paths(data_dir, progress_every=progress_every), False


def collect_training_pt_paths(data_dir: Path, *, progress_every: int = 25) -> list[Path]:
    paths: list[Path] = []
    seen = 0
    for path in data_dir.rglob("*.pt"):
        seen += 1
        if path.is_file() and not any(str(path).endswith(suffix) for suffix in BLOCKED_PT_SUFFIXES):
            paths.append(path)
        if progress_every > 0 and seen % progress_every == 0:
            print(
                "[export_contact_obj_viz] recursive scan "
                f"seen_pt_like={seen} candidate_training_pt={len(paths)}",
                flush=True,
            )
    return sorted(paths)


def load_pt_metadata(path: Path) -> dict[str, Any]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("payload is not a dict")
    if data.get("schema_version") != "contact_pt_env_v1":
        raise ValueError(f"unexpected schema_version={data.get('schema_version')!r}")
    if str(data.get("generation_status", "")) != "complete":
        raise ValueError(f"generation_status={data.get('generation_status')!r}")
    return {
        "num_contacts": int(data.get("num_contacts", 0)),
        "tool_id": str(data.get("tool_id", "")),
        "object_id": str(data.get("object_id", "")),
    }


def sample_tasks(
    entries: Sequence[PtEntry],
    *,
    output_dir: Path,
    num_outputs: int,
    seed: int,
    floor_z: float,
    floor_margin: float,
    floor_min_size: float,
    weighted_by_contacts: bool,
    tool_post_delta_field: str,
) -> list[ExportTask]:
    rng = random.Random(seed)
    tasks: list[ExportTask] = []
    for output_index in range(num_outputs):
        if weighted_by_contacts:
            total_contacts = sum(entry.num_contacts for entry in entries)
            pick = rng.randrange(total_contacts)
            running = 0
            entry = entries[-1]
            contact_index = 0
            for candidate in entries:
                next_running = running + candidate.num_contacts
                if pick < next_running:
                    entry = candidate
                    contact_index = pick - running
                    break
                running = next_running
        else:
            entry = rng.choice(entries)
            contact_index = rng.randrange(entry.num_contacts)
        obj_path = output_dir / f"{output_index:06d}_{safe_stem(Path(entry.path).stem)}_c{contact_index:04d}.obj"
        tasks.append(
            ExportTask(
                output_index=output_index,
                pt_path=entry.path,
                contact_index=int(contact_index),
                obj_path=str(obj_path),
                floor_z=floor_z,
                floor_margin=floor_margin,
                floor_min_size=floor_min_size,
                tool_post_delta_field=tool_post_delta_field,
            )
        )
    return tasks


def export_one(task: ExportTask) -> dict[str, Any]:
    import torch

    data = torch.load(task.pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"payload is not a dict: {task.pt_path}")
    i = int(task.contact_index)
    n = int(data["num_contacts"])
    if i < 0 or i >= n:
        raise IndexError(f"contact index {i} out of range for {n}: {task.pt_path}")

    object_local, object_faces = load_centered_mesh(
        data["object_mesh_path"],
        scale=float(data["object_scale"]),
        bbox_center=np_array(data["object_bbox_center_M"], (3,), "object_bbox_center_M"),
    )
    tool_local, tool_faces = load_centered_mesh(
        data["tool_mesh_path"],
        scale=np_array(data["tool_scale_xyz"], (3,), "tool_scale_xyz"),
        bbox_center=np_array(data["tool_bbox_center_M"], (3,), "tool_bbox_center_M"),
    )

    object_R = np_array(data["object_rotation_E"][i], (3, 3), "object_rotation_E")
    object_t = np_array(data["object_bbox_center_E"][i], (3,), "object_bbox_center_E")
    tool_R = np_array(data["tool_rotation_E"][i], (3, 3), "tool_rotation_E")
    tool_t = np_array(data["tool_translation_E"][i], (3,), "tool_translation_E")

    object_delta = np_array(data["post_object_delta_pose9d_E"][i], (9,), "post_object_delta_pose9d_E")
    tool_delta_key = task.tool_post_delta_field
    if tool_delta_key not in data:
        tool_delta_key = "post_tool_delta_pose9d_E"
    tool_delta = np_array(data[tool_delta_key][i], (9,), tool_delta_key)

    object_delta_R = rotation_from_pose9d_np(object_delta)
    tool_delta_R = rotation_from_pose9d_np(tool_delta)

    pre_object = transform_vertices(object_local, object_R, object_t)
    post_object = transform_vertices(object_local, object_delta_R @ object_R, object_t + object_delta[:3])
    pre_tool = transform_vertices(tool_local, tool_R, tool_t)
    post_tool = transform_vertices(tool_local, tool_delta_R @ tool_R, tool_t + tool_delta[:3])
    floor_vertices, floor_faces = make_floor(
        [pre_object, post_object, pre_tool, post_tool],
        floor_z=float(task.floor_z),
        margin=float(task.floor_margin),
        min_size=float(task.floor_min_size),
    )

    obj_path = Path(task.obj_path)
    mtl_path = obj_path.with_suffix(".mtl")
    write_mtl(mtl_path)
    groups = [
        ("floor", floor_vertices, floor_faces),
        ("pre_object", pre_object, object_faces),
        ("post_object", post_object, object_faces),
        ("pre_tool", pre_tool, tool_faces),
        ("post_tool", post_tool, tool_faces),
    ]
    write_obj(
        obj_path,
        mtl_path=mtl_path,
        groups=groups,
        comments=[
            f"source_pt {task.pt_path}",
            f"contact_index {i}",
            f"tool_id {data.get('tool_id', '')}",
            f"object_id {data.get('object_id', '')}",
            f"object_mesh_path {data.get('object_mesh_path', '')}",
            f"tool_mesh_path {data.get('tool_mesh_path', '')}",
            f"tool_post_delta_field {tool_delta_key}",
            "mesh_transform scaled_vertices_minus_bbox_center_then_env_pose",
        ],
    )
    return {
        "output_index": int(task.output_index),
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "pt_path": str(task.pt_path),
        "contact_index": i,
        "tool_id": str(data.get("tool_id", "")),
        "object_id": str(data.get("object_id", "")),
        "tool_post_delta_field": tool_delta_key,
    }


def load_centered_mesh(
    mesh_path: str | Path,
    *,
    scale: float | Sequence[float],
    bbox_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = load_mesh_vertices_faces(mesh_path, process=False)
    scaled = scale_vertices(vertices, scale)
    centered = np.asarray(scaled, dtype=np.float64) - bbox_center.reshape(1, 3)
    return centered, np.asarray(faces, dtype=np.int64)


def transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)


def make_floor(
    vertex_groups: Sequence[np.ndarray],
    *,
    floor_z: float,
    margin: float,
    min_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([np.asarray(group, dtype=np.float64).reshape(-1, 3) for group in vertex_groups], axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        raise ValueError("cannot build floor around non-finite or empty vertices")
    mins = finite[:, :2].min(axis=0)
    maxs = finite[:, :2].max(axis=0)
    center = (mins + maxs) * 0.5
    half = np.maximum((maxs - mins) * 0.5 + float(margin), float(min_size) * 0.5)
    vertices = np.array(
        [
            [center[0] - half[0], center[1] - half[1], floor_z],
            [center[0] + half[0], center[1] - half[1], floor_z],
            [center[0] + half[0], center[1] + half[1], floor_z],
            [center[0] - half[0], center[1] + half[1], floor_z],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def write_mtl(path: Path) -> None:
    lines: list[str] = []
    for name, color in MATERIALS.items():
        r, g, b = color
        lines.extend(
            [
                f"newmtl {name}",
                f"Ka {r:.6f} {g:.6f} {b:.6f}",
                f"Kd {r:.6f} {g:.6f} {b:.6f}",
                "Ks 0.050000 0.050000 0.050000",
                "Ns 16.000000",
                "d 1.000000",
                "illum 2",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def write_obj(
    path: Path,
    *,
    mtl_path: Path,
    groups: Sequence[tuple[str, np.ndarray, np.ndarray]],
    comments: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# contact obj visualization export"]
    lines.extend(f"# {comment}" for comment in comments)
    lines.append(f"mtllib {mtl_path.name}")
    vertex_offset = 1
    for name, vertices, faces in groups:
        lines.append("")
        lines.append(f"g {name}")
        lines.append(f"usemtl {name}")
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"group {name} vertices must have shape (N, 3), got {verts.shape}")
        if not np.isfinite(verts).all():
            raise ValueError(f"group {name} has non-finite vertices")
        for x, y, z in verts:
            lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        faces_arr = np.asarray(faces, dtype=np.int64)
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError(f"group {name} faces must have shape (F, 3), got {faces_arr.shape}")
        for a, b, c in faces_arr:
            lines.append(f"f {int(a) + vertex_offset} {int(b) + vertex_offset} {int(c) + vertex_offset}")
        vertex_offset += int(verts.shape[0])
    path.write_text("\n".join(lines) + "\n")


def np_array(value: Any, shape: tuple[int, ...], key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64)
    if tuple(arr.shape) != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains non-finite values")
    return arr


def path_matches_tool_dir(path: Path, normalized_tool_name: str) -> bool:
    for parent in path.parents:
        if variant_dir_matches(parent, normalized_tool_name):
            return True
    return False


def variant_dir_matches(path: Path, normalized_tool_name: str) -> bool:
    family = variant_family_name(path.name)
    return bool(family and tool_family_matches(normalize_tool_name(family), normalized_tool_name))


def variant_family_name(dirname: str) -> str | None:
    match = re.match(r"^(?:\d+_)?(.+)_var_\d+$", dirname)
    if not match:
        return None
    return match.group(1)


def tool_id_matches(tool_id: str, normalized_tool_name: str) -> bool:
    normalized = normalize_tool_name(tool_id)
    family = variant_family_name(normalized) or normalized
    return tool_family_matches(normalize_tool_name(family), normalized_tool_name)


def tool_family_matches(normalized_family: str, normalized_query: str) -> bool:
    if normalized_family == normalized_query:
        return True
    if normalized_family.startswith(normalized_query + "_"):
        return True
    return normalized_query in normalized_family


def normalize_tool_name(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^(?:\d+_)?", "", text)
    return text


def safe_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        return "contact"
    return text[:120]


if __name__ == "__main__":
    raise SystemExit(main())

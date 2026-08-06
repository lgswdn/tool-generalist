#!/usr/bin/env python3
"""Export VLA/GraspGen mismatch cases as OBJ scenes."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import check_graspgen_episode_consistency as common


DEFAULT_REPORT = Path("scripts/outputs/graspgen_episode_consistency_full_1024_top256.jsonl")
DEFAULT_MESH_DIR = Path("/mnt/project/world_model/tool_generalist/assets/DGN/coacd_normalized")
DEFAULT_GRASPGEN_ROOT = Path("/mnt/project/world_model/tool_generalist/GraspGen")
DEFAULT_OUTPUT = Path("scripts/outputs/vla_success_no_hand_fingers_free_viz_seed0")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-jsonl", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--graspgen-root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection",
        choices=(
            "vla_success_no_free",
            "vla_failure_has_free",
            "vla_failure_no_free",
        ),
        default="vla_success_no_free",
        help=(
            "Mismatch direction to export. vla_failure_has_free means VLA failed "
            "but GraspGen has at least one hand+fingers collision-free candidate; "
            "vla_failure_no_free means both sides fail under the confidence threshold."
        ),
    )
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--candidates-per-case", type=int, default=10)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float("-inf"),
        help="A strict collision-free grasp only counts when confidence >= this value.",
    )
    parser.add_argument(
        "--sampling",
        choices=("random", "representative"),
        default="random",
        help=(
            "representative samples across GraspGen evidence quantiles while preferring "
            "different object categories."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opacity", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _strict_free(candidate: dict[str, Any], min_confidence: float) -> bool:
    return bool(candidate.get("hand_and_fingers_collision_free")) and float(
        candidate.get("confidence", float("-inf"))
    ) >= min_confidence


def _matches(row: dict[str, Any], selection: str, min_confidence: float) -> bool:
    vla_success = bool(row.get("episode_success"))
    graspgen_free = any(_strict_free(item, min_confidence) for item in row["candidates"])
    if selection == "vla_success_no_free":
        return vla_success and not graspgen_free
    if selection == "vla_failure_has_free":
        return not vla_success and graspgen_free
    if selection == "vla_failure_no_free":
        return not vla_success and not graspgen_free
    raise ValueError(f"Unsupported selection {selection!r}")


def _sample_cases(
    path: Path,
    count: int,
    seed: int,
    selection: str,
    min_confidence: float,
    sampling: str,
) -> tuple[list[dict[str, Any]], int]:
    if sampling == "representative":
        return _representative_cases(path, count, seed, selection, min_confidence)
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    matched = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not _matches(row, selection, min_confidence):
                continue
            matched += 1
            if len(reservoir) < count:
                reservoir.append(row)
            else:
                replace = rng.randrange(matched)
                if replace < count:
                    reservoir[replace] = row
    reservoir.sort(key=lambda row: int(row["task_index"]))
    return reservoir, matched


def _object_category(name: str) -> str:
    match = re.match(r"^(core|sem)-([^-]+)-", name)
    if match:
        return match.group(2)
    if name.startswith("ddg-ycb_"):
        return name.split("-", 1)[0].removeprefix("ddg-ycb_")
    return name.split("-", 1)[0]


def _representative_cases(
    path: Path,
    count: int,
    seed: int,
    selection: str,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], int]:
    summaries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not _matches(row, selection, min_confidence):
                continue
            strict = [
                item for item in row["candidates"]
                if bool(item.get("hand_and_fingers_collision_free"))
            ]
            qualifying = [item for item in strict if _strict_free(item, min_confidence)]
            best_strict = max(
                (float(item["confidence"]) for item in strict), default=float("-inf")
            )
            score = (
                max(float(item["confidence"]) for item in qualifying)
                if qualifying
                else best_strict
            )
            summaries.append(
                {
                    "offset": offset,
                    "task_index": int(row["task_index"]),
                    "category": _object_category(str(row["object"])),
                    "score": score,
                }
            )
    if len(summaries) <= count:
        chosen = summaries
    else:
        # One item from each evidence quantile, preferring categories not yet represented.
        ordered = sorted(summaries, key=lambda item: (item["score"], item["task_index"]))
        rng = random.Random(seed)
        chosen = []
        used_categories: set[str] = set()
        for index in range(count):
            lo = index * len(ordered) // count
            hi = (index + 1) * len(ordered) // count
            bucket = ordered[lo:hi]
            diverse = [item for item in bucket if item["category"] not in used_categories]
            candidates = diverse or bucket
            item = candidates[rng.randrange(len(candidates))]
            chosen.append(item)
            used_categories.add(item["category"])
    offsets = {int(item["offset"]) for item in chosen}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for item in chosen:
            stream.seek(int(item["offset"]))
            rows.append(json.loads(stream.readline()))
    rows.sort(key=lambda row: int(row["task_index"]))
    return rows, len(summaries)


def _load_gripper_meshes(root: Path) -> tuple[Any, Any]:
    hand = trimesh.load(str(root / "assets/panda_gripper/hand.stl"), force="mesh", process=False)
    left = trimesh.load(str(root / "assets/panda_gripper/finger.stl"), force="mesh", process=False)
    right = left.copy()
    rotate = np.eye(4)
    rotate[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    left.apply_transform(rotate)
    left.apply_translation([0.04, 0.0, 0.0584])
    right.apply_translation([-0.04, 0.0, 0.0584])
    return hand, trimesh.util.concatenate([left, right])


def _write_mtl(path: Path, opacity: float) -> None:
    materials = {
        "ground": ((0.58, 0.58, 0.58), 1.0),
        "object": ((0.82, 0.30, 0.25), 1.0),
        "guide_finger_object": ((1.00, 0.78, 0.05), 1.0),
        "guide_finger_ground": ((1.00, 0.35, 0.05), 1.0),
        "guide_hand_collision": ((1.00, 0.05, 0.05), 1.0),
        "guide_strict_safe": ((0.10, 1.00, 0.10), 1.0),
        "guide_strict_safe_low_confidence": ((0.72, 0.20, 1.00), 1.0),
        "hand_clear": ((0.15, 0.85, 0.95), opacity),
        "hand_collision": ((1.00, 0.10, 0.10), opacity),
        "fingers_clear": ((0.20, 0.45, 1.00), opacity),
        "fingers_object_collision": ((1.00, 0.75, 0.05), max(opacity, 0.45)),
        "fingers_ground_collision": ((1.00, 0.25, 0.05), max(opacity, 0.45)),
    }
    with path.open("w", encoding="utf-8") as stream:
        for name, (color, alpha) in materials.items():
            stream.write(f"newmtl {name}\n")
            stream.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            stream.write("Ka 0.050000 0.050000 0.050000\nKs 0.100000 0.100000 0.100000\n")
            stream.write(f"d {alpha:.6f}\nTr {1.0-alpha:.6f}\nillum 2\n\n")


def _candidate_materials(
    candidate: dict[str, Any], min_confidence: float
) -> tuple[str, str, str]:
    hand_collision = bool(candidate["panda_hand_object_collision"]) or bool(
        candidate["panda_hand_ground_collision"]
    )
    finger_object = bool(candidate["panda_fingers_object_collision"])
    finger_ground = bool(candidate["panda_fingers_ground_collision"])
    if bool(candidate["hand_and_fingers_collision_free"]):
        guide = (
            "guide_strict_safe"
            if float(candidate["confidence"]) >= min_confidence
            else "guide_strict_safe_low_confidence"
        )
    elif hand_collision:
        guide = "guide_hand_collision"
    elif finger_object:
        guide = "guide_finger_object"
    else:
        guide = "guide_finger_ground"
    hand = "hand_collision" if hand_collision else "hand_clear"
    fingers = (
        "fingers_object_collision"
        if finger_object
        else "fingers_ground_collision"
        if finger_ground
        else "fingers_clear"
    )
    return guide, hand, fingers


def _selected_candidates(
    row: dict[str, Any],
    limit: int,
    selection: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    candidates = sorted(row["candidates"], key=lambda item: int(item["rank"]))
    strict_safe = [item for item in candidates if bool(item["hand_and_fingers_collision_free"])]
    qualifying_strict_safe = [
        item for item in strict_safe if _strict_free(item, min_confidence)
    ]
    low_confidence_strict_safe = [
        item for item in strict_safe if not _strict_free(item, min_confidence)
    ]
    hand_only_safe = [
        item
        for item in candidates
        if bool(item["collision_free"]) and not bool(item["hand_and_fingers_collision_free"])
    ]
    if selection == "vla_failure_has_free":
        return qualifying_strict_safe[:limit]
    selected_ids = {id(item) for item in low_confidence_strict_safe + hand_only_safe}
    high_confidence_remainder = [
        item for item in candidates
        if id(item) not in selected_ids and float(item["confidence"]) >= min_confidence
    ]
    remainder = [
        item for item in candidates
        if id(item) not in selected_ids and item not in high_confidence_remainder
    ]
    return (
        low_confidence_strict_safe + hand_only_safe + high_confidence_remainder + remainder
    )[:limit]


def _transform_vertices(mesh: Any, transform: np.ndarray) -> np.ndarray:
    return np.asarray(mesh.vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def _export_case(
    row: dict[str, Any],
    scene_index: int,
    output_dir: Path,
    mesh_dir: Path,
    hand_mesh: Any,
    finger_mesh: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_name = str(row["object"])
    object_mesh = trimesh.load(str(mesh_dir / f"{object_name}.obj"), force="mesh", process=False)
    object_transform = np.eye(4)
    object_transform[:3, :3] = common._quat_wxyz_matrix(
        np.asarray(row["orientation_wxyz"], dtype=np.float64)
    ) * float(row["scale"])
    object_transform[:3, 3] = np.asarray(row["position"], dtype=np.float64)
    support_z = float(row["collision"]["support_top_z"])
    ground_mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.04])
    ground_transform = np.eye(4)
    ground_transform[:3, 3] = [row["position"][0], row["position"][1], support_z - 0.02]
    selected = _selected_candidates(
        row,
        int(args.candidates_per_case),
        str(args.selection),
        float(args.min_confidence),
    )

    stem = f"case_{scene_index:02d}_task_{int(row['task_index']):06d}_{common._safe_filename(object_name)}"
    obj_path = output_dir / f"{stem}.obj"
    mtl_path = output_dir / f"{stem}.mtl"
    json_path = output_dir / f"{stem}.json"
    _write_mtl(mtl_path, float(args.opacity))
    vertex_offset = 0
    temporary = obj_path.with_name(f".{obj_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(f"mtllib {mtl_path.name}\n\n")
            vertex_offset = common._append_obj_mesh(
                stream, "ground", "ground", _transform_vertices(ground_mesh, ground_transform),
                ground_mesh.faces, vertex_offset,
            )
            vertex_offset = common._append_obj_mesh(
                stream, "object", "object", _transform_vertices(object_mesh, object_transform),
                object_mesh.faces, vertex_offset,
            )
            for candidate in selected:
                rank = int(candidate["rank"])
                matrix = np.asarray(candidate["grasp_matrix_world"], dtype=np.float64)
                guide_material, hand_material, finger_material = _candidate_materials(
                    candidate, float(args.min_confidence)
                )
                for part, (vertices, faces) in enumerate(
                    common._grasp_guide_meshes(matrix, width=0.08, depth=0.10, thickness=0.0025)
                ):
                    vertex_offset = common._append_obj_mesh(
                        stream, f"grasp_rank_{rank:03d}_{part}", guide_material,
                        vertices, faces, vertex_offset,
                    )
                vertex_offset = common._append_obj_mesh(
                    stream, f"panda_hand_rank_{rank:03d}", hand_material,
                    _transform_vertices(hand_mesh, matrix), hand_mesh.faces, vertex_offset,
                )
                vertex_offset = common._append_obj_mesh(
                    stream, f"panda_fingers_rank_{rank:03d}", finger_material,
                    _transform_vertices(finger_mesh, matrix), finger_mesh.faces, vertex_offset,
                )
        os.replace(temporary, obj_path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "object": object_name,
        "task_index": int(row["task_index"]),
        "episode_success": bool(row["episode_success"]),
        "hand_only_collision_free_grasps": int(row["graspgen"]["collision_free"]),
        "hand_and_fingers_collision_free_grasps": int(
            row["graspgen"]["hand_and_fingers_collision_free"]
        ),
        "min_confidence": float(args.min_confidence),
        "qualifying_hand_and_fingers_collision_free_grasps": sum(
            _strict_free(item, float(args.min_confidence)) for item in row["candidates"]
        ),
        "rendered_candidates": selected,
        "obj_path": str(obj_path),
        "color_legend": {
            "yellow": "hand clear, fingers collide with object",
            "orange": "hand clear, fingers collide with ground",
            "red": "panda_hand collides",
            "green": "hand and both fingers are collision-free",
            "purple": "hand and both fingers are collision-free but confidence is below threshold",
            "cyan_transparent": "panda_hand clear",
            "blue_transparent": "fingers clear",
        },
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    args = _args()
    if args.num_cases <= 0 or args.candidates_per_case <= 0:
        raise ValueError("--num-cases and --candidates-per-case must be positive")
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be in [0, 1]")
    args.report_jsonl = args.report_jsonl.expanduser().resolve()
    args.mesh_dir = args.mesh_dir.expanduser().resolve()
    args.graspgen_root = args.graspgen_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, matched = _sample_cases(
        args.report_jsonl,
        int(args.num_cases),
        int(args.seed),
        str(args.selection),
        float(args.min_confidence),
        str(args.sampling),
    )
    hand, fingers = _load_gripper_meshes(args.graspgen_root)
    scenes = [
        _export_case(row, index, args.output_dir, args.mesh_dir, hand, fingers, args)
        for index, row in enumerate(rows)
    ]
    manifest = {
        "source_report": str(args.report_jsonl),
        "selection": str(args.selection),
        "matched_cases": matched,
        "sampled_cases": len(scenes),
        "seed": int(args.seed),
        "sampling": str(args.sampling),
        "min_confidence": float(args.min_confidence),
        "candidates_per_case": int(args.candidates_per_case),
        "scenes": scenes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] matched={matched} sampled={len(scenes)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

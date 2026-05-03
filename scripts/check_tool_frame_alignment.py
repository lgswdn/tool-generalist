#!/usr/bin/env python3
"""Check whether pretrain and RL use the same tool local frame.

Pretrain contact generation uses:
    P_tool_pretrain = raw_obj_vertices * TOOL_SCALE

RL observations use:
    P_tool_rl = raw_obj_vertices * TOOL_SCALE - body_origin

where body_origin is computed from tools.json/base_center in the same way as
get_tool_pointcloud_in_env_frame().  A non-zero body_origin means the frozen
encoder sees a translated tool cloud in RL compared with pretraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_scalar(value: str) -> str | float:
    value = value.strip()
    if value == "":
        return ""
    number_chars = set("0123456789.-+eE")
    if all(ch in number_chars for ch in value):
        return float(value)
    return value


def load_tools_config(path_yaml: Path) -> dict[str, str | float]:
    text = path_yaml.read_text()
    sections: dict[str, dict[str, str | float]] = {}
    current_section = ""

    for line_no, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].rstrip()
        if content.strip() == "":
            continue

        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()

        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"{path_yaml}:{line_no}: expected a top-level section ending with ':'")
            current_section = stripped[:-1]
            sections[current_section] = {}
            continue

        if indent == 2:
            if current_section == "":
                raise ValueError(f"{path_yaml}:{line_no}: key appears before any top-level section")
            if ":" not in stripped:
                raise ValueError(f"{path_yaml}:{line_no}: expected 'key: value'")
            key, value = stripped.split(":", 1)
            sections[current_section][key.strip()] = _parse_scalar(value)
            continue

        raise ValueError(f"{path_yaml}:{line_no}: unsupported indentation level {indent}")

    if "tools" not in sections:
        raise KeyError(f"{path_yaml} does not contain a 'tools' section")

    tools_cfg = sections["tools"]
    for key in ("tools_json", "tools_selected_json", "obj_dir", "scale"):
        if key not in tools_cfg:
            raise KeyError(f"{path_yaml} tools section is missing '{key}'")

    return tools_cfg


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw in obj_path.read_text().splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) == 0:
        raise ValueError(f"{obj_path} contains no OBJ vertex lines")

    return np.asarray(vertices, dtype=np.float64)


def format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.6f}" for x in vec.tolist()) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect frame offsets between pretrain contact_gen tool clouds and RL tool observations."
    )
    parser.add_argument(
        "--paths-yaml",
        type=Path,
        default=Path("paths_multitool.yaml"),
        help="Path config containing tools.tools_json, tools.tools_selected_json, tools.obj_dir, and tools.scale.",
    )
    parser.add_argument(
        "--warn-threshold-mm",
        type=float,
        default=1.0,
        help="Report tools with pretrain-vs-RL local-frame offset above this threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of largest-offset tools to print.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only check the first N selected tools. Use 0 to check all selected tools.",
    )
    args = parser.parse_args()

    tools_cfg = load_tools_config(args.paths_yaml)
    tools_json = Path(str(tools_cfg["tools_json"]))
    selected_json = Path(str(tools_cfg["tools_selected_json"]))
    obj_dir = Path(str(tools_cfg["obj_dir"]))
    scale = float(tools_cfg["scale"])

    meta_list = json.loads(tools_json.read_text())
    selected_names = json.loads(selected_json.read_text())
    meta_by_name = {entry["name"]: entry for entry in meta_list}

    names = selected_names
    if args.limit > 0:
        names = selected_names[: args.limit]

    rows = []
    for name in names:
        if name not in meta_by_name:
            raise KeyError(f"Selected tool '{name}' is missing from {tools_json}")

        meta = meta_by_name[name]
        if "base_center" not in meta:
            raise KeyError(f"Tool '{name}' is missing base_center in {tools_json}")

        obj_path = obj_dir / f"{name}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(f"Tool OBJ is missing: {obj_path}")

        vertices = load_obj_vertices(obj_path)
        bbox_min_raw = vertices.min(axis=0)
        bbox_max_raw = vertices.max(axis=0)
        bbox_size_raw = bbox_max_raw - bbox_min_raw
        base_center_norm = np.asarray(meta["base_center"], dtype=np.float64)

        body_origin = (bbox_min_raw + base_center_norm * bbox_size_raw) * scale
        pretrain_centroid = (vertices * scale).mean(axis=0)
        rl_centroid = pretrain_centroid - body_origin
        offset_m = float(np.linalg.norm(body_origin))

        rows.append(
            {
                "name": name,
                "offset_m": offset_m,
                "offset_mm": offset_m * 1000.0,
                "body_origin": body_origin,
                "pretrain_centroid": pretrain_centroid,
                "rl_centroid": rl_centroid,
                "bbox_size_scaled": bbox_size_raw * scale,
            }
        )

    rows.sort(key=lambda row: row["offset_m"], reverse=True)
    threshold_m = args.warn_threshold_mm / 1000.0
    flagged = [row for row in rows if row["offset_m"] > threshold_m]

    offsets_mm = np.asarray([row["offset_mm"] for row in rows], dtype=np.float64)
    print("Tool frame alignment check")
    print(f"  paths_yaml       : {args.paths_yaml}")
    print(f"  tools_json       : {tools_json}")
    print(f"  selected_json    : {selected_json}")
    print(f"  obj_dir          : {obj_dir}")
    print(f"  tool_scale       : {scale}")
    print(f"  checked_tools    : {len(rows)}")
    print(f"  warn_threshold   : {args.warn_threshold_mm:.3f} mm")
    print(f"  flagged_tools    : {len(flagged)}")
    print(
        "  offset_mm stats : "
        f"min={offsets_mm.min():.3f}  mean={offsets_mm.mean():.3f}  "
        f"p95={np.percentile(offsets_mm, 95):.3f}  max={offsets_mm.max():.3f}"
    )
    print()

    print(f"Top {min(args.top_k, len(rows))} offsets:")
    for row in rows[: args.top_k]:
        status = "FLAG" if row["offset_m"] > threshold_m else "ok"
        print(
            f"{status:4s}  {row['offset_mm']:9.3f} mm  {row['name']}\n"
            f"      body_origin_m      = {format_vec(row['body_origin'])}\n"
            f"      pretrain_centroid_m = {format_vec(row['pretrain_centroid'])}\n"
            f"      rl_centroid_m       = {format_vec(row['rl_centroid'])}\n"
            f"      bbox_size_scaled_m  = {format_vec(row['bbox_size_scaled'])}"
        )

    if len(flagged) > 0:
        print()
        print("Interpretation:")
        print(
            "  Non-zero offset means contact_gen pretraining and RL observations use different tool local frames."
        )
        print(
            "  In world/env frame the point-cloud difference is -R_tool @ body_origin, so its norm is the offset above."
        )
        print(
            "  If these tools were used for SDF pretraining, pretrain should apply the same body_origin shift as RL."
        )


if __name__ == "__main__":
    main()

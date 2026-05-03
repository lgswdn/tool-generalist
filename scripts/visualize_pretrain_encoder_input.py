#!/usr/bin/env python3
"""Visualize the point clouds actually fed to the SDF pretrain encoder.

This reconstructs the same tensors as pretrain/dataset.py:

    tool_pc = tool_pts_canonical @ tool_rotations[idx].T + tool_translations[idx]
    obj_pc  = obj_pts_canonical @ object_rotation.T
    obj_pc[:, 2] -= obj_z_shift

The figure is saved as a PNG so it can be inspected without launching Isaac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_frame(ax, origin: np.ndarray, rotation: np.ndarray, scale: float, prefix: str) -> None:
    colors = ("red", "green", "blue")
    labels = (f"{prefix} x", f"{prefix} y", f"{prefix} z")
    for axis_id in range(3):
        vec = rotation[:, axis_id] * scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=colors[axis_id],
            linewidth=1.5,
            label=labels[axis_id],
        )


def load_obj_vertices(obj_path: Path) -> torch.Tensor:
    vertices: list[list[float]] = []
    for raw in obj_path.read_text().splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) == 0:
        raise ValueError(f"{obj_path} contains no OBJ vertex lines")

    return torch.tensor(vertices, dtype=torch.float32)


def compute_body_origin(tool_mesh_path: Path, tools_json: Path, tool_scale: float) -> torch.Tensor:
    tool_name = tool_mesh_path.stem
    tools_meta = json.loads(tools_json.read_text())
    matches = [entry for entry in tools_meta if entry["name"] == tool_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one metadata entry for '{tool_name}' in {tools_json}, got {len(matches)}")

    base_center = torch.tensor(matches[0]["base_center"], dtype=torch.float32)
    vertices = load_obj_vertices(tool_mesh_path)
    bbox_min = vertices.min(dim=0).values
    bbox_max = vertices.max(dim=0).values
    return (bbox_min + base_center * (bbox_max - bbox_min)) * tool_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SDF pretrain encoder input point clouds from a .pt file.")
    parser.add_argument("contact_pt", type=Path, help="Path to one contact dataset .pt file.")
    parser.add_argument("--index", type=int, default=0, help="Contact config index inside the .pt file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/outputs/pretrain_encoder_input.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--object-centered",
        action="store_true",
        help="Subtract object point-cloud centroid before plotting. This is for inspection only; pure SDF pretrain does not do this.",
    )
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--overlay-rl-shifted",
        action="store_true",
        help="Also plot tool_pc reconstructed with RL's body_origin shift for the same stored R,t.",
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        help="tools.json containing base_center. Required with --overlay-rl-shifted.",
    )
    args = parser.parse_args()

    if args.overlay_rl_shifted and args.tools_json is None:
        raise ValueError("--tools-json is required when --overlay-rl-shifted is used")

    data = torch.load(args.contact_pt, map_location="cpu", weights_only=False)

    tool_points = data["tool_pts_canonical"].float()
    obj_points = data["obj_pts_canonical"].float()

    tool_rotation = data["tool_rotations"][args.index].float()
    tool_translation = data["tool_translations"][args.index].float()
    object_rotation = data["object_rotation"].float()
    object_z_shift = data["obj_z_shift"].float()

    tool_pc = tool_points @ tool_rotation.T + tool_translation
    rl_shifted_tool_pc = None
    body_origin = None
    if args.overlay_rl_shifted:
        body_origin = compute_body_origin(
            Path(data["tool_mesh_path"]),
            args.tools_json,
            float(data["tool_scale"]),
        )
        rl_shifted_tool_pc = (tool_points - body_origin) @ tool_rotation.T + tool_translation

    obj_pc = obj_points @ object_rotation.T
    obj_pc = obj_pc.clone()
    obj_pc[:, 2] -= object_z_shift

    contact_pts = data["contact_pts_world"][args.index].float()
    contact_normals = data["contact_normals"][args.index].float()

    plot_tool_pc = tool_pc.clone()
    plot_rl_shifted_tool_pc = None
    if rl_shifted_tool_pc is not None:
        plot_rl_shifted_tool_pc = rl_shifted_tool_pc.clone()
    plot_obj_pc = obj_pc.clone()
    plot_contact_pts = contact_pts.clone()
    tool_origin = tool_translation.clone()
    object_origin = torch.zeros(3, dtype=torch.float32)

    object_center = obj_pc.mean(dim=0)
    if args.object_centered:
        plot_tool_pc -= object_center
        if plot_rl_shifted_tool_pc is not None:
            plot_rl_shifted_tool_pc -= object_center
        plot_obj_pc -= object_center
        plot_contact_pts -= object_center
        tool_origin -= object_center
        object_origin -= object_center

    tool_np = plot_tool_pc.numpy()
    rl_shifted_tool_np = None
    if plot_rl_shifted_tool_pc is not None:
        rl_shifted_tool_np = plot_rl_shifted_tool_pc.numpy()
    obj_np = plot_obj_pc.numpy()
    contact_np = plot_contact_pts.numpy()
    normal_np = contact_normals.numpy()
    tool_origin_np = tool_origin.numpy()
    object_origin_np = object_origin.numpy()
    tool_rotation_np = tool_rotation.numpy()
    object_rotation_np = object_rotation.numpy()

    point_groups = [tool_np, obj_np, contact_np]
    if rl_shifted_tool_np is not None:
        point_groups.append(rl_shifted_tool_np)
    all_points = np.concatenate(point_groups, axis=0)
    span = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    frame_scale = max(span * 0.12, 0.01)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(obj_np[:, 0], obj_np[:, 1], obj_np[:, 2], s=args.point_size, c="#2f6fed", alpha=0.65, label="object_pc")
    ax.scatter(tool_np[:, 0], tool_np[:, 1], tool_np[:, 2], s=args.point_size, c="#d97706", alpha=0.75, label="pretrain_tool_pc")
    if rl_shifted_tool_np is not None:
        ax.scatter(
            rl_shifted_tool_np[:, 0],
            rl_shifted_tool_np[:, 1],
            rl_shifted_tool_np[:, 2],
            s=args.point_size,
            c="#c026d3",
            alpha=0.35,
            label="rl_shifted_tool_pc_same_Rt",
        )
    ax.scatter(contact_np[:, 0], contact_np[:, 1], contact_np[:, 2], s=args.point_size * 3.0, c="#111111", label="contact_pts")

    for point, normal in zip(contact_np, normal_np):
        ax.quiver(
            point[0],
            point[1],
            point[2],
            normal[0] * frame_scale,
            normal[1] * frame_scale,
            normal[2] * frame_scale,
            color="#111111",
            linewidth=1.0,
        )

    draw_frame(ax, tool_origin_np, tool_rotation_np, frame_scale, "tool")
    draw_frame(ax, object_origin_np, object_rotation_np, frame_scale, "object")

    title_mode = "object-centered view" if args.object_centered else "pretrain env-frame encoder input"
    ax.set_title(f"{title_mode}\n{args.contact_pt.name}  index={args.index}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=8)
    set_axes_equal(ax)
    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"saved: {args.output}")
    print(f"tool_mesh_path: {data['tool_mesh_path']}")
    print(f"object_mesh_path: {data['object_mesh_path']}")
    print(f"index: {args.index}")
    print(f"object_center: {object_center.tolist()}")
    print(f"tool_pc mean: {tool_pc.mean(dim=0).tolist()}")
    print(f"obj_pc mean: {obj_pc.mean(dim=0).tolist()}")
    print(f"tool_pc min: {tool_pc.min(dim=0).values.tolist()}")
    print(f"tool_pc max: {tool_pc.max(dim=0).values.tolist()}")
    print(f"obj_pc min: {obj_pc.min(dim=0).values.tolist()}")
    print(f"obj_pc max: {obj_pc.max(dim=0).values.tolist()}")
    if rl_shifted_tool_pc is not None:
        diff = rl_shifted_tool_pc - tool_pc
        print(f"body_origin: {body_origin.tolist()}")
        print(f"body_origin_norm_mm: {float(body_origin.norm() * 1000.0):.6f}")
        print(f"rl_shifted_tool_pc mean: {rl_shifted_tool_pc.mean(dim=0).tolist()}")
        print(f"rl_shifted_minus_pretrain mean: {diff.mean(dim=0).tolist()}")
        print(f"rl_shifted_minus_pretrain norm_mm min: {float(diff.norm(dim=1).min() * 1000.0):.6f}")
        print(f"rl_shifted_minus_pretrain norm_mm max: {float(diff.norm(dim=1).max() * 1000.0):.6f}")


if __name__ == "__main__":
    main()

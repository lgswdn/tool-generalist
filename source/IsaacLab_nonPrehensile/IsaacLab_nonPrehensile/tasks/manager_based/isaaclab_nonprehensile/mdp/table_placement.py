"""Shared table placement helpers for reset events and target commands."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


def table_contract_from_env(env: Any) -> SimpleNamespace:
    cfg = env.cfg
    return SimpleNamespace(
        enabled=bool(getattr(cfg, "table_enabled", False)),
        size_xyz=tuple(getattr(cfg, "table_size_xyz", (1.0, 1.0, 0.04))),
        pose_xyz=tuple(getattr(cfg, "table_pose_xyz", (0.0, 0.0, -0.02))),
        placement_margin_xy=float(getattr(cfg, "table_placement_margin_xy", 0.02)),
        placement_max_attempts=int(getattr(cfg, "table_placement_max_attempts", 64)),
    )


def table_bounds_from_contract(table_cfg: Any, device: torch.device | str) -> torch.Tensor:
    pose = torch.as_tensor(table_cfg.pose_xyz, dtype=torch.float32, device=device)
    size = torch.as_tensor(table_cfg.size_xyz, dtype=torch.float32, device=device)
    half_xy = 0.5 * size[:2]
    return torch.stack((pose[:2] - half_xy, pose[:2] + half_xy), dim=0)


def table_top_z_from_contract(table_cfg: Any, device: torch.device | str) -> torch.Tensor:
    pose = torch.as_tensor(table_cfg.pose_xyz, dtype=torch.float32, device=device)
    size = torch.as_tensor(table_cfg.size_xyz, dtype=torch.float32, device=device)
    return pose[2] + 0.5 * size[2]


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.stack(
        (
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def rotate_points(points: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    points = points.to(dtype=torch.float32)
    quat = quat.to(device=points.device, dtype=torch.float32)
    rot = quat_wxyz_to_matrix(quat)
    if quat.ndim == 1:
        return points @ rot.transpose(-1, -2)
    if points.ndim == 2:
        return points.unsqueeze(0) @ rot.transpose(-1, -2)
    return points @ rot.transpose(-1, -2)


def rotated_xy_half_extents(points: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rotated = rotate_points(points, quat)
    min_xy = rotated[..., :2].amin(dim=-2)
    max_xy = rotated[..., :2].amax(dim=-2)
    return 0.5 * (max_xy - min_xy)


def rotated_aabb(points: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotated = rotate_points(points, quat)
    return rotated.amin(dim=-2), rotated.amax(dim=-2)


def table_safe_xy_range(
    bounds: torch.Tensor,
    half_extents: torch.Tensor,
    margin: float | torch.Tensor,
    center_offset_xy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    margin_t = torch.as_tensor(margin, dtype=torch.float32, device=bounds.device)
    half_extents = half_extents.to(device=bounds.device, dtype=torch.float32)
    center_offset = (
        torch.zeros_like(half_extents)
        if center_offset_xy is None
        else center_offset_xy.to(device=bounds.device, dtype=torch.float32)
    )
    low = bounds[0] - center_offset + half_extents + margin_t
    high = bounds[1] - center_offset - half_extents - margin_t
    if bool((low > high).any().item()):
        raise ValueError(
            "Object XY extents plus table placement margin exceed the configured table footprint."
        )
    return low, high


def sample_table_xy(
    bounds: torch.Tensor,
    half_extents: torch.Tensor,
    margin: float | torch.Tensor,
    *,
    device: torch.device | str,
    center_offset_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    low, high = table_safe_xy_range(
        bounds.to(device=device),
        half_extents,
        margin,
        center_offset_xy=center_offset_xy,
    )
    return low + torch.rand(low.shape, dtype=torch.float32, device=device) * (high - low)


def surface_z_for_points(points: torch.Tensor, quat: torch.Tensor, top_z: torch.Tensor) -> torch.Tensor:
    rotated = rotate_points(points, quat)
    min_z = rotated[..., 2].amin(dim=-1)
    return top_z.to(device=points.device, dtype=torch.float32) - min_z

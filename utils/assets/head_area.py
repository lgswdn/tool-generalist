"""Tool functional head-area helpers."""

from __future__ import annotations

from typing import Any, Optional, Tuple


def compute_head_bounds(
    points: Any,
    head_area: Optional[Tuple[list, list]],
) -> Optional[Tuple[Any, Any]]:
    """Convert normalized bbox head-area ratios into point-space bounds."""

    if head_area is None:
        return None
    import torch

    device = points.device
    bbox_min = points.min(dim=0).values
    bbox_range = points.max(dim=0).values - bbox_min
    lo = torch.tensor(head_area[0], device=device, dtype=points.dtype)
    hi = torch.tensor(head_area[1], device=device, dtype=points.dtype)
    return bbox_min + lo * bbox_range, bbox_min + hi * bbox_range


def split_head_body(
    points: Any,
    bounds: Optional[Tuple[Any, Any]],
) -> Tuple[Any, Any]:
    """Split a tool surface cloud into head/body subsets with full-cloud fallback."""

    if bounds is None:
        return points, points
    head_min, head_max = bounds
    in_head = ((points >= head_min.unsqueeze(0)) & (points <= head_max.unsqueeze(0))).all(dim=-1)
    head_points = points[in_head]
    body_points = points[~in_head]
    if head_points.shape[0] == 0:
        head_points = points
    if body_points.shape[0] == 0:
        body_points = points
    return head_points, body_points

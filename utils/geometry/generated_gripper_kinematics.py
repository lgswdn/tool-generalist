"""Strict cached point correspondence for generated gripper candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from utils.geometry.gripper_cloud_cache import (
    GripperCloudCache,
    load_gripper_cloud_cache,
)


_METADATA_CACHE: dict[Path, dict[str, Mapping[str, Any]]] = {}


def _load_tool_metadata(
    mesh_path: Path,
    tool_id: str,
) -> Mapping[str, Any]:
    try:
        asset_root = mesh_path.parents[3]
    except IndexError as exc:
        raise ValueError(
            f"Generated-gripper mesh path is too shallow: {mesh_path}"
        ) from exc
    metadata_path = asset_root / "tools_adjusted.json"
    by_id = _METADATA_CACHE.get(metadata_path)
    if by_id is None:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in {metadata_path}")
        by_id = {str(item["name"]): item for item in payload}
        if len(by_id) != len(payload):
            raise ValueError(f"Duplicate generated-gripper names in {metadata_path}")
        _METADATA_CACHE[metadata_path] = by_id
    try:
        return by_id[tool_id]
    except KeyError as exc:
        raise KeyError(f"{tool_id!r} is absent from {metadata_path}") from exc


@dataclass(frozen=True)
class CachedGripperPointKinematics:
    """The same 128-bin cloud used by contact generation, pretraining, and RL."""

    opening_fraction: float
    bbox_center: torch.Tensor
    cache: GripperCloudCache

    @classmethod
    def from_candidate(
        cls,
        data: Mapping[str, Any],
        tool_points_centered: torch.Tensor,
    ) -> "CachedGripperPointKinematics":
        tool_id = str(data["tool_id"])
        if not tool_id.startswith(
            ("generated_gripper_", "one_dof_gripper_")
        ):
            raise ValueError(f"Expected a cached gripper tool, got {tool_id!r}")
        metadata = _load_tool_metadata(
            Path(str(data["tool_mesh_path"])).expanduser().resolve(),
            tool_id,
        )
        source_ids = [
            metadata[key]
            for key in (
                "source_generated_gripper_id",
                "source_one_dof_gripper_id",
            )
            if key in metadata
        ]
        if len(source_ids) != 1:
            raise ValueError(
                f"{tool_id!r} must identify exactly one cached gripper source"
            )
        raw_cache = metadata.get("kinematic_cloud_cache")
        if raw_cache is None:
            raise ValueError(f"{tool_id!r} has no kinematic_cloud_cache")
        cache_path = Path(str(raw_cache)).expanduser()
        if not cache_path.is_absolute():
            cache_path = (
                Path(str(data["tool_mesh_path"])).parents[3] / cache_path
            )
        cache = load_gripper_cloud_cache(
            cache_path,
            expected_gripper_id=str(source_ids[0]),
            expected_source_manifest=str(metadata["source_manifest"]),
        )
        opening_fraction = float(metadata["opening_fraction"])
        bbox_center = torch.as_tensor(
            data["tool_bbox_center_M"], dtype=torch.float32
        )
        expected = cache.cloud_at_fraction(opening_fraction) - bbox_center
        points = tool_points_centered.detach().to(
            dtype=torch.float32, device="cpu"
        )
        if not torch.equal(points, expected):
            max_error = float((points - expected).abs().max())
            raise ValueError(
                "Candidate cloud is not its canonical cached bin: "
                f"tool={tool_id} max_error={max_error}"
            )
        return cls(
            opening_fraction=opening_fraction,
            bbox_center=bbox_center,
            cache=cache,
        )

    def cloud_at_fraction(
        self,
        tool_points_centered: torch.Tensor,
        fraction: float,
        *,
        canonical_local: bool,
    ) -> torch.Tensor:
        result = self.cache.cloud_at_fraction(fraction).to(
            tool_points_centered
        )
        if not canonical_local:
            result = result - self.bbox_center.to(result)
        return result.contiguous()

    def static_state_clouds(
        self,
        tool_points_centered: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                self.cloud_at_fraction(
                    tool_points_centered,
                    fraction,
                    canonical_local=True,
                )
                for fraction in (0.0, 0.5, 1.0)
            ]
        )


def point_kinematics_from_candidate(
    data: Mapping[str, Any],
    tool_points_centered: torch.Tensor,
) -> CachedGripperPointKinematics:
    return CachedGripperPointKinematics.from_candidate(
        data, tool_points_centered
    )

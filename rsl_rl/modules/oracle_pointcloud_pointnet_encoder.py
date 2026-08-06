"""Fast patchwise PointNet initialized from the rank-10 token probe."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch3d.ops import knn_points
except ImportError:  # CPU/test environments.
    knn_points = None


FAST11_PROBE_V1_INPUT_MEAN = (
    3.703237234731205e-05,
    3.045662879230804e-06,
    -7.395312309199653e-07,
    -0.00014748447574675083,
    5.159009015187621e-05,
    0.04665474593639374,
    0.12567077577114105,
    -0.0005540500278584659,
    -0.001838304684497416,
    0.038589391857385635,
    0.5,
)
FAST11_PROBE_V1_INPUT_STD = (
    0.014583528973162174,
    0.014586255885660648,
    0.014202220365405083,
    0.09321845322847366,
    0.09306279569864273,
    0.07845285534858704,
    0.07759512960910797,
    0.5629075765609741,
    0.5628711581230164,
    0.6040019989013672,
    0.5,
)


class OraclePointCloudEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class OraclePointCloudPreparedGeometry(NamedTuple):
    """Compact, non-learned neighborhood data cached with a rollout."""

    indices: torch.Tensor


class OraclePointCloudMaterializedGeometry(NamedTuple):
    """Continuous patch geometry reconstructed from cached discrete indices."""

    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor
    tool_patches: torch.Tensor
    obj_patches: torch.Tensor
    tool_distance: torch.Tensor
    obj_distance: torch.Tensor
    tool_direction: torch.Tensor
    obj_direction: torch.Tensor


class OraclePointCloudPointNetEncoder(nn.Module):
    """Encode patches from point-cloud nearest distance/direction only.

    The per-point input is ``(relative_xyz, patch_center_xyz, unsigned_distance,
    direction_xyz, patch_is_tool)``.  The fitted 10D token is projected through
    the source rank-10 policy's learned 10D-to-128D reconstruction before RL
    fusion.
    """

    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        feature_dim: int = 128,
        nearest_frame_batch_size: int = 64,
        feature_mode: str = "fast11",
        use_rank10_bottleneck: bool = True,
        token_mode: str = "patches",
        input_normalization: str = "identity",
    ) -> None:
        super().__init__()
        if min(num_points, num_patches, patch_size, feature_dim, nearest_frame_batch_size) <= 0:
            raise ValueError("oracle point-cloud PointNet dimensions must be positive")
        if feature_dim != 128:
            raise ValueError("rank-10 probe reconstruction requires feature_dim=128")
        self.num_points = int(num_points)
        self._P = int(num_patches)
        self.patch_size = int(patch_size)
        self._D = int(feature_dim)
        self.nearest_frame_batch_size = int(nearest_frame_batch_size)
        self.feature_mode = str(feature_mode).strip().lower()
        if self.feature_mode not in {"fast11", "rich21"}:
            raise ValueError("oracle point-cloud feature_mode must be 'fast11' or 'rich21'")
        self.use_rank10_bottleneck = bool(use_rank10_bottleneck)
        self.token_mode = str(token_mode).strip().lower()
        if self.token_mode not in {"patches", "points"}:
            raise ValueError("oracle point-cloud token_mode must be 'patches' or 'points'")
        if self.feature_mode == "rich21" and self.token_mode != "patches":
            raise ValueError("rich21 point-cloud features require patch token mode")
        input_dim = 21 if self.feature_mode == "rich21" else 11
        self.input_normalization = str(input_normalization).strip().lower()
        if self.input_normalization not in {"identity", "fast11_probe_v1"}:
            raise ValueError(
                "oracle point-cloud input_normalization must be "
                "'identity' or 'fast11_probe_v1'"
            )
        if self.input_normalization != "identity" and self.feature_mode != "fast11":
            raise ValueError("fast11_probe_v1 normalization requires feature_mode='fast11'")
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        if self.use_rank10_bottleneck:
            self.patch_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 10),
            )
            self.token_up = nn.Linear(10, feature_dim)
        else:
            self.patch_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, feature_dim),
                nn.GELU(),
            )
            self.token_up = nn.Identity()
        if self.token_mode == "points":
            # These checkpoint modules are intentionally bypassed in point mode.
            # Mark them non-trainable so distributed PPO does not see unused
            # trainable parameters.
            for module in (self.patch_mlp, self.token_up):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if self.input_normalization == "fast11_probe_v1":
            input_mean = torch.tensor(FAST11_PROBE_V1_INPUT_MEAN, dtype=torch.float32)
            input_std = torch.tensor(FAST11_PROBE_V1_INPUT_STD, dtype=torch.float32)
        else:
            input_mean = torch.zeros(input_dim)
            input_std = torch.ones(input_dim)
        self.register_buffer("input_mean", input_mean, persistent=True)
        self.register_buffer("input_std", input_std, persistent=True)

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def num_patches(self) -> int:
        return self._P * self.patch_size if self.token_mode == "points" else self._P

    @staticmethod
    def _fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        count = min(int(count), num_points)
        centroids = torch.zeros(batch_size, count, dtype=torch.long, device=points.device)
        distance = torch.full(
            (batch_size, num_points), float("inf"), dtype=points.dtype, device=points.device
        )
        farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
        batch = torch.arange(batch_size, device=points.device)
        for index in range(count):
            centroids[:, index] = farthest
            center = points[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, ((points - center) ** 2).sum(dim=-1))
            farthest = distance.max(dim=1).indices
        return centroids

    def _patch_indices_and_centers(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, centers, _ = self._patch_indices_centers_and_center_indices(points)
        return indices, centers

    def _patch_indices_centers_and_center_indices(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = points.shape
        center_idx = self._fps_indices(points, self._P)
        batch = torch.arange(batch_size, device=points.device).view(batch_size, 1)
        centers = points[batch, center_idx]
        k = min(self.patch_size, num_points)
        if knn_points is not None and points.is_cuda:
            _, indices, _ = knn_points(
                centers.contiguous(), points.contiguous(), K=k, return_nn=False
            )
        else:
            indices = torch.cdist(centers, points).topk(
                k=k, dim=-1, largest=False
            ).indices
        if k < self.patch_size:
            indices = torch.cat(
                (
                    indices,
                    indices[..., -1:].expand(
                        batch_size, indices.shape[1], self.patch_size - k
                    ),
                ),
                dim=-1,
            )
        return indices, centers, center_idx

    @staticmethod
    def _gather_patches(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
        return points[batch, indices]

    def _nearest_indices_both(
        self,
        tool_points: torch.Tensor,
        object_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest-opposite indices without retaining distance matrices."""
        if knn_points is not None and tool_points.is_cuda:
            batch_size = tool_points.shape[0]
            query = torch.cat((tool_points, object_points), dim=0).contiguous()
            reference = torch.cat((object_points, tool_points), dim=0).contiguous()
            _, nearest_index, _ = knn_points(
                query,
                reference,
                K=1,
                return_nn=False,
            )
            tool_index, object_index = nearest_index.squeeze(-1).split(batch_size)
            return tool_index, object_index

        tool_indices = []
        object_indices = []
        for start in range(0, tool_points.shape[0], self.nearest_frame_batch_size):
            tool = tool_points[start : start + self.nearest_frame_batch_size]
            obj = object_points[start : start + self.nearest_frame_batch_size]
            pairwise = torch.cdist(tool, obj)
            tool_indices.append(pairwise.argmin(dim=-1))
            object_indices.append(pairwise.argmin(dim=-2))
        return torch.cat(tool_indices, dim=0), torch.cat(object_indices, dim=0)

    @property
    def prepared_index_dim(self) -> int:
        # tool patches + object patches + nearest indices in each direction,
        # followed by the FPS center indices for both bodies.
        return 4 * self._P * self.patch_size + 2 * self._P

    def prepare_geometry(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> OraclePointCloudPreparedGeometry:
        """Compute all discrete searches once for a stored observation.

        The result deliberately contains only int16 indices.  It is independent
        of every learned PointNet parameter and is therefore safe to reuse over
        all PPO epochs while the PointNet itself is updated.
        """
        expected = (tool_pc.shape[0], self.num_points, 3)
        if tool_pc.shape != expected or obj_pc.shape != expected:
            raise RuntimeError(
                f"oracle point-cloud inputs must both have shape {expected}; "
                f"got tool={tuple(tool_pc.shape)} object={tuple(obj_pc.shape)}"
            )
        if max(self.num_points, self._P * self.patch_size) > torch.iinfo(torch.int16).max:
            raise RuntimeError("prepared oracle point-cloud indices exceed int16 capacity")

        batch_size = tool_pc.shape[0]
        both_clouds = torch.cat((tool_pc, obj_pc), dim=0)
        both_idx, _, both_center_idx = self._patch_indices_centers_and_center_indices(
            both_clouds
        )
        tool_idx, obj_idx = both_idx.split(batch_size)
        tool_center_idx, obj_center_idx = both_center_idx.split(batch_size)
        both_patches = self._gather_patches(both_clouds, both_idx)
        tool_patches, obj_patches = both_patches.split(batch_size)
        tool_flat = tool_patches.reshape(tool_pc.shape[0], -1, 3)
        obj_flat = obj_patches.reshape(obj_pc.shape[0], -1, 3)
        tool_nearest, obj_nearest = self._nearest_indices_both(tool_flat, obj_flat)

        # Store FPS centers explicitly so reconstruction does not depend on KNN
        # tie ordering.
        packed = torch.cat(
            (
                tool_idx.flatten(1),
                obj_idx.flatten(1),
                tool_nearest,
                obj_nearest,
                tool_center_idx,
                obj_center_idx,
            ),
            dim=1,
        )
        if packed.shape[1] != self.prepared_index_dim:
            raise RuntimeError(
                f"prepared index width mismatch: {packed.shape[1]} != {self.prepared_index_dim}"
            )
        return OraclePointCloudPreparedGeometry(indices=packed.to(torch.int16))

    def encode_prepared(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        prepared: OraclePointCloudPreparedGeometry | torch.Tensor,
    ) -> OraclePointCloudEncodeResult:
        """Run the trainable PointNet using cached discrete search results."""
        geometry = self._materialize_prepared_geometry(tool_pc, obj_pc, prepared)
        tool_inputs = self._normalized_point_inputs(
            geometry.tool_patches,
            geometry.tool_patch_centers,
            geometry.tool_distance,
            geometry.tool_direction,
            is_tool=True,
        )
        obj_inputs = self._normalized_point_inputs(
            geometry.obj_patches,
            geometry.obj_patch_centers,
            geometry.obj_distance,
            geometry.obj_direction,
            is_tool=False,
        )
        both_inputs = torch.cat((tool_inputs, obj_inputs), dim=1)
        if self.token_mode == "points":
            # Keep only the pretrained PointNet's pointwise stem.  There is no
            # patch max-pool, patch MLP, rank-10 bottleneck, or token_up here.
            both_tokens = self.point_mlp(both_inputs)
            tool_tokens = both_tokens[:, : self._P].flatten(1, 2)
            obj_tokens = both_tokens[:, self._P :].flatten(1, 2)
        else:
            both_tokens = self.token_up(
                self.patch_mlp(self.point_mlp(both_inputs).amax(dim=-2))
            )
            tool_tokens = both_tokens[:, : self._P]
            obj_tokens = both_tokens[:, self._P :]
        return OraclePointCloudEncodeResult(
            fused_tokens=torch.cat((tool_tokens, obj_tokens), dim=1),
            tool_patch_idx=geometry.tool_patch_idx,
            obj_patch_idx=geometry.obj_patch_idx,
            tool_patch_centers=geometry.tool_patch_centers,
            obj_patch_centers=geometry.obj_patch_centers,
        )

    def _materialize_prepared_geometry(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        prepared: OraclePointCloudPreparedGeometry | torch.Tensor,
    ) -> OraclePointCloudMaterializedGeometry:
        """Reconstruct differentiable geometry without repeating any search."""
        packed = (
            prepared.indices
            if isinstance(prepared, OraclePointCloudPreparedGeometry)
            else prepared
        )
        if packed.ndim != 2 or packed.shape != (tool_pc.shape[0], self.prepared_index_dim):
            raise RuntimeError(
                "prepared oracle point-cloud indices must have shape "
                f"({tool_pc.shape[0]}, {self.prepared_index_dim}), got {tuple(packed.shape)}"
            )
        packed = packed.long()
        patch_count = self._P * self.patch_size
        cursor = 0
        tool_idx = packed[:, cursor : cursor + patch_count].reshape(
            -1, self._P, self.patch_size
        )
        cursor += patch_count
        obj_idx = packed[:, cursor : cursor + patch_count].reshape(
            -1, self._P, self.patch_size
        )
        cursor += patch_count
        tool_nearest = packed[:, cursor : cursor + patch_count]
        cursor += patch_count
        obj_nearest = packed[:, cursor : cursor + patch_count]
        cursor += patch_count
        tool_center_idx = packed[:, cursor : cursor + self._P]
        cursor += self._P
        obj_center_idx = packed[:, cursor : cursor + self._P]

        batch_size = tool_pc.shape[0]
        both_clouds = torch.cat((tool_pc, obj_pc), dim=0)
        both_idx = torch.cat((tool_idx, obj_idx), dim=0)
        both_center_idx = torch.cat((tool_center_idx, obj_center_idx), dim=0)
        both_nearest_idx = torch.cat((tool_nearest, obj_nearest), dim=0)
        both_batch = torch.arange(2 * batch_size, device=tool_pc.device).view(-1, 1)
        both_centers = both_clouds[both_batch, both_center_idx]
        both_patches = self._gather_patches(both_clouds, both_idx)
        both_flat = both_patches.reshape(2 * batch_size, patch_count, 3)
        opposite_flat = torch.cat(
            (both_flat[batch_size:], both_flat[:batch_size]), dim=0
        )
        nearest_opposite = opposite_flat.gather(
            1, both_nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        both_delta = nearest_opposite - both_flat
        both_distance = torch.linalg.vector_norm(both_delta, dim=-1).reshape(
            -1, self._P, self.patch_size
        )
        both_direction = F.normalize(both_delta, dim=-1, eps=1e-8).reshape(
            -1, self._P, self.patch_size, 3
        )
        tool_centers, obj_centers = both_centers.split(batch_size)
        tool_patches, obj_patches = both_patches.split(batch_size)
        tool_distance, obj_distance = both_distance.split(batch_size)
        tool_direction, obj_direction = both_direction.split(batch_size)
        return OraclePointCloudMaterializedGeometry(
            tool_patch_idx=tool_idx,
            obj_patch_idx=obj_idx,
            tool_patch_centers=tool_centers,
            obj_patch_centers=obj_centers,
            tool_patches=tool_patches,
            obj_patches=obj_patches,
            tool_distance=tool_distance,
            obj_distance=obj_distance,
            tool_direction=tool_direction,
            obj_direction=obj_direction,
        )

    def _raw_point_inputs(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        distance: torch.Tensor,
        direction: torch.Tensor,
        *,
        is_tool: bool,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        local = patches - centers.unsqueeze(-2)
        center = centers.unsqueeze(-2).expand(-1, -1, self.patch_size, -1)
        body = torch.full(
            (batch_size, self._P, self.patch_size, 1),
            1.0 if is_tool else 0.0,
            device=patches.device,
            dtype=patches.dtype,
        )
        base = torch.cat((local, center, distance.unsqueeze(-1), direction, body), dim=-1)
        if self.feature_mode == "fast11":
            return base

        displacement = direction * distance.unsqueeze(-1)
        # Preserve raw metric distance for long-range approach and add a
        # separately saturated log channel for millimeter-scale distinctions.
        fine_log_distance = (
            torch.log1p(distance / 0.001) / torch.log1p(
                torch.as_tensor(100.0, device=distance.device, dtype=distance.dtype)
            )
        ).clamp(max=1.0)
        patch_min = distance.amin(dim=-1, keepdim=True).expand_as(distance)
        patch_mean = distance.mean(dim=-1, keepdim=True).expand_as(distance)
        patch_std = distance.std(dim=-1, keepdim=True, unbiased=False).expand_as(distance)
        contacts = torch.stack(
            tuple((distance <= threshold).to(distance.dtype) for threshold in (0.002, 0.005, 0.010)),
            dim=-1,
        )
        return torch.cat(
            (
                base,
                displacement,
                fine_log_distance.unsqueeze(-1),
                patch_min.unsqueeze(-1),
                patch_mean.unsqueeze(-1),
                patch_std.unsqueeze(-1),
                contacts,
            ),
            dim=-1,
        )

    def _pointnet_tokens(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        distance: torch.Tensor,
        direction: torch.Tensor,
        *,
        is_tool: bool,
    ) -> torch.Tensor:
        normalized = self._normalized_point_inputs(
            patches,
            centers,
            distance,
            direction,
            is_tool=is_tool,
        )
        latent = self.patch_mlp(self.point_mlp(normalized).amax(dim=-2))
        return self.token_up(latent)

    def _normalized_point_inputs(
        self,
        patches: torch.Tensor,
        centers: torch.Tensor,
        distance: torch.Tensor,
        direction: torch.Tensor,
        *,
        is_tool: bool,
    ) -> torch.Tensor:
        raw = self._raw_point_inputs(
            patches, centers, distance, direction, is_tool=is_tool
        )
        if self.feature_mode == "rich21":
            normalized = raw.clone()
            normalized[..., 0:3] /= 0.05
            normalized[..., 3:6] /= 0.30
            normalized[..., 6] /= 0.10
            # Direction 7:10 is already unit scale.
            normalized[..., 10] = normalized[..., 10] * 2.0 - 1.0
            normalized[..., 11:14] /= 0.10
            # Fine log distance 14 is already in [0, 1].
            normalized[..., 15:18] /= 0.10
            normalized[..., 18:21] = normalized[..., 18:21] * 2.0 - 1.0
            return normalized.clamp(-12.0, 12.0)
        return ((raw - self.input_mean) / self.input_std).clamp(-12.0, 12.0)

    def raw_point_features(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact pre-normalization PointNet inputs used by RL."""

        prepared = self.prepare_geometry(tool_pc, obj_pc)
        geometry = self._materialize_prepared_geometry(
            tool_pc, obj_pc, prepared
        )
        tool_inputs = self._raw_point_inputs(
            geometry.tool_patches,
            geometry.tool_patch_centers,
            geometry.tool_distance,
            geometry.tool_direction,
            is_tool=True,
        )
        object_inputs = self._raw_point_inputs(
            geometry.obj_patches,
            geometry.obj_patch_centers,
            geometry.obj_distance,
            geometry.obj_direction,
            is_tool=False,
        )
        return torch.cat((tool_inputs, object_inputs), dim=1)

    def encode(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        **_: object,
    ) -> OraclePointCloudEncodeResult:
        prepared = self.prepare_geometry(tool_pc, obj_pc)
        return self.encode_prepared(tool_pc, obj_pc, prepared)

    def forward(self, tool_pc: torch.Tensor, obj_pc: torch.Tensor, **kwargs):
        return self.encode(tool_pc, obj_pc, **kwargs)

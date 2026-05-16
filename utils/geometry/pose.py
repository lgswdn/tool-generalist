"""Pose helpers that use bbox centers as canonical rotation pivots."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .bbox import bbox_center_mesh


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _zero_translation_like(points: Any) -> Any:
    if _is_torch_tensor(points):
        return torch.zeros(3, dtype=points.dtype, device=points.device)
    return np.zeros(3, dtype=np.asarray(points).dtype)


def apply_pose_about_bbox_center(
    points: Any,
    rotation: Any,
    translation: Optional[Any] = None,
    bbox_center: Optional[Any] = None,
) -> Any:
    """Apply a rigid pose while rotating around the point set's AABB center.

    The row-vector convention is used for returned points:
    ``posed = (points - center) @ rotation.T + center + translation``.
    Batched rotations with shape ``(N, 3, 3)`` are supported for unbatched
    points, returning ``(N, P, 3)``.
    """

    center = bbox_center if bbox_center is not None else bbox_center_mesh(points)[0]
    trans = translation if translation is not None else _zero_translation_like(points)

    if _is_torch_tensor(points):
        rot = rotation.to(dtype=points.dtype, device=points.device)
        cen = center.to(dtype=points.dtype, device=points.device) if _is_torch_tensor(center) else torch.as_tensor(center, dtype=points.dtype, device=points.device)
        trn = trans.to(dtype=points.dtype, device=points.device) if _is_torch_tensor(trans) else torch.as_tensor(trans, dtype=points.dtype, device=points.device)
        centered = points - cen
        if rot.ndim == 2:
            return centered @ rot.T + cen + trn
        if points.ndim == 2:
            return torch.einsum("pi,nji->npj", centered, rot) + cen + trn.unsqueeze(-2)
        if points.ndim == 3 and rot.ndim == 3:
            return torch.einsum("npi,nji->npj", centered, rot) + cen + trn.unsqueeze(-2)
        raise ValueError(f"Unsupported torch pose shapes points={tuple(points.shape)}, rotation={tuple(rot.shape)}")

    pts = np.asarray(points)
    rot_np = np.asarray(rotation)
    cen_np = np.asarray(center, dtype=pts.dtype)
    trn_np = np.asarray(trans, dtype=pts.dtype)
    centered_np = pts - cen_np
    if rot_np.ndim == 2:
        return centered_np @ rot_np.T + cen_np + trn_np
    if pts.ndim == 2:
        trn_b = trn_np[None, None, :] if trn_np.ndim == 1 else trn_np[:, None, :]
        return np.einsum("pi,nji->npj", centered_np, rot_np) + cen_np + trn_b
    if pts.ndim == 3 and rot_np.ndim == 3:
        trn_b = trn_np[None, None, :] if trn_np.ndim == 1 else trn_np[:, None, :]
        return np.einsum("npi,nji->npj", centered_np, rot_np) + cen_np + trn_b
    raise ValueError(f"Unsupported numpy pose shapes points={pts.shape}, rotation={rot_np.shape}")


def pose9d_from_rt(translation: Any, rotation: Any) -> Any:
    """Pack translation plus the first two rotation columns into pose9d."""

    if _is_torch_tensor(rotation) or _is_torch_tensor(translation):
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for torch pose9d helpers")
        if _is_torch_tensor(rotation):
            rot = rotation
            trans = translation if _is_torch_tensor(translation) else torch.as_tensor(translation, dtype=rot.dtype, device=rot.device)
        else:
            trans = translation
            rot = torch.as_tensor(rotation, dtype=trans.dtype, device=trans.device)
        rot6 = rot[..., :, :2].reshape(*rot.shape[:-2], 6)
        return torch.cat((trans, rot6), dim=-1).to(dtype=torch.float32)

    rot_np = np.asarray(rotation, dtype=np.float64)
    trans_np = np.asarray(translation, dtype=np.float64)
    return np.concatenate((trans_np, rot_np[..., :, :2].reshape(*rot_np.shape[:-2], 6)), axis=-1)


def rotation_from_pose9d(pose9d: Any) -> Any:
    """Recover an orthonormal rotation from a pose9d tensor/array."""

    if _is_torch_tensor(pose9d):
        rot6 = pose9d[..., 3:].reshape(*pose9d.shape[:-1], 3, 2)
        v1 = rot6[..., :, 0]
        v2 = rot6[..., :, 1]
        u1 = torch.nn.functional.normalize(v1, dim=-1)
        u2 = v2 - (u1 * v2).sum(dim=-1, keepdim=True) * u1
        u2 = torch.nn.functional.normalize(u2, dim=-1)
        u3 = torch.cross(u1, u2, dim=-1)
        return torch.stack((u1, u2, u3), dim=-1)

    mat = np.asarray(pose9d, dtype=np.float64)[..., 3:].reshape(*np.asarray(pose9d).shape[:-1], 3, 2)
    x = mat[..., :, 0]
    y = mat[..., :, 1]
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x = np.divide(x, x_norm, out=np.zeros_like(x), where=x_norm > 1e-12)
    fallback_x = np.zeros_like(x)
    fallback_x[..., 0] = 1.0
    x = np.where(x_norm > 1e-12, x, fallback_x)
    y = y - x * np.sum(x * y, axis=-1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    y = np.divide(y, y_norm, out=np.zeros_like(y), where=y_norm > 1e-12)
    fallback_y = np.zeros_like(y)
    fallback_y[..., 1] = 1.0
    y = np.where(y_norm > 1e-12, y, fallback_y)
    z = np.cross(x, y, axis=-1)
    z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    z = np.divide(z, z_norm, out=np.zeros_like(z), where=z_norm > 1e-12)
    fallback_z = np.zeros_like(z)
    fallback_z[..., 2] = 1.0
    z = np.where(z_norm > 1e-12, z, fallback_z)
    y = np.cross(z, x, axis=-1)
    return np.stack((x, y, z), axis=-1)


def apply_pose9d_delta(current_translation: Any, current_rotation: Any, delta_pose9d: Any) -> tuple[Any, Any]:
    delta_t = delta_pose9d[..., :3]
    delta_R = rotation_from_pose9d(delta_pose9d)
    return current_translation + delta_t, delta_R @ current_rotation


def pose9d_from_transform_np(delta_t: Any, delta_r: Any) -> np.ndarray:
    return np.asarray(pose9d_from_rt(np.asarray(delta_t), np.asarray(delta_r)), dtype=np.float32)


def rotation_from_pose9d_np(pose9d: Any) -> np.ndarray:
    return np.asarray(rotation_from_pose9d(np.asarray(pose9d)), dtype=np.float64)

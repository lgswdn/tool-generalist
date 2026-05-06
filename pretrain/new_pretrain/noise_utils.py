"""noise_utils.py — SE(3) noising for RPDiff-style iterative pose denoising.

Fully batched PyTorch implementation (no scipy, no per-sample Python loops).
  - Batched quaternion SLERP on GPU
  - Vectorized trajectory interpolation
  - Batch-level rejection sampling via compute_on_the_fly_sdf
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import kaolin


# ============================================================================ #
# Batched quaternion utilities (replaces scipy)
# ============================================================================ #

def _random_quaternions(B: int, max_rot_deg: float, device: torch.device) -> torch.Tensor:
    """Sample B random quaternions with angle in [0, max_rot_deg].

    Returns (B, 4) in (w, x, y, z) convention.
    """
    max_rad = max_rot_deg * (torch.pi / 180.0)
    # Random axes (unit vectors)
    axes = torch.randn(B, 3, device=device)
    axes = F.normalize(axes, dim=-1)
    # Random angles uniform in [0, max_rad]
    angles = torch.rand(B, device=device) * max_rad
    half = angles * 0.5
    w = torch.cos(half)                       # (B,)
    xyz = axes * torch.sin(half).unsqueeze(-1)  # (B, 3)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # (B, 4) wxyz


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation matrix. (B, 4) → (B, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (w, x, y, z). (B, 3, 3) → (B, 4).

    Robust implementation handling all cases.
    """
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(trace[m1] + 1.0) * 2  # s = 4w
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = torch.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    # Case 3: R[1,1] > R[2,2]
    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = torch.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    # Case 4: else
    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        s = torch.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s

    return F.normalize(q, dim=-1)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched quaternion SLERP.  q0, q1: (B, 4) wxyz.  t: (B,) in [0,1].

    Returns (B, 4).
    """
    # Ensure shortest path
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # (B, 1)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=0.9999)

    theta = torch.acos(dot)                # (B, 1)
    t = t.unsqueeze(-1)                    # (B, 1)
    sin_theta = torch.sin(theta)
    # Guard against sin_theta ≈ 0 (nearly identical quaternions)
    safe = sin_theta.abs() > 1e-6
    s0 = torch.where(safe, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    s1 = torch.where(safe, torch.sin(t * theta) / sin_theta, t)
    return F.normalize(s0 * q0 + s1 * q1, dim=-1)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched quaternion multiplication. (B, 4) wxyz × (B, 4) wxyz → (B, 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse (conjugate for unit quaternions). (B, 4) wxyz."""
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


# ============================================================================ #
# Fully batched noise sampling
# ============================================================================ #

def sample_noised_poses_batch(
    contact_R: torch.Tensor,     # (B, 3, 3)
    contact_t: torch.Tensor,     # (B, 3)
    num_steps: int,
    max_trans: float,
    max_rot_deg: float,
    interp: bool = True,
    precise_prob: bool = False,
    # For rejection sampling
    tool_canonical: torch.Tensor = None,  # (B, P, 3)
    obj_pc: torch.Tensor = None,          # (B, Q, 3) centered at origin
    obj_centroid: torch.Tensor = None,    # (B, 3) world-frame object centroid; fixes guard
    pen_threshold: float = 0.001,
    max_retries: int = 10,
) -> dict:
    """Fully batched SE(3) noise sampling with GPU rejection.

    All operations are vectorized on GPU — no Python loops over batch items.

    Guard: min tool-to-object NN distance >= pen_threshold.
    Fallback: rejected items get contact pose (t_idx=0).
    """
    B = contact_R.shape[0]
    device = contact_R.device
    T = num_steps
    do_guards = (tool_canonical is not None and obj_pc is not None)

    # Identity quaternion (w, x, y, z)
    q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(B, -1)

    # Convert contact pose to quaternion
    contact_q = _rotmat_to_quat(contact_R)  # (B, 4) wxyz

    # Pre-allocate results — start as contact pose (safe fallback)
    result_t_idx = torch.zeros(B, dtype=torch.long, device=device)
    result_noised_R = contact_R.clone()
    result_noised_t = contact_t.clone()
    result_target_trans = torch.zeros(B, 3, device=device)
    result_target_rot = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

    # Mask of items that still need a valid sample
    pending = torch.ones(B, dtype=torch.bool, device=device)

    for _retry in range(max_retries):
        n_pending = pending.sum().item()
        if n_pending == 0:
            break

        # ── 1. Sample perturbations for pending items ─────────────────
        pert_q = _random_quaternions(n_pending, max_rot_deg, device)  # (n, 4)
        pert_t = (torch.rand(n_pending, 3, device=device) * 2 - 1) * max_trans

        # ── 2. Sample timesteps ───────────────────────────────────────
        if precise_prob:
            weights = torch.exp(-1.0 * torch.arange(T + 1, device=device, dtype=torch.float32))
            t_idx = torch.multinomial(weights.expand(n_pending, -1), 1).squeeze(-1)
        else:
            t_idx = torch.randint(0, T + 1, (n_pending,), device=device)

        # ── 3. Interpolate to sampled timestep (batched SLERP) ────────
        alpha = t_idx.float() / T  # (n,) in [0, 1]

        # Cumulative rotation at t_idx: SLERP(identity, pert_q, alpha)
        cum_q = _quat_slerp(q_id[:n_pending], pert_q, alpha)  # (n, 4)
        cum_R = _quat_to_rotmat(cum_q)                         # (n, 3, 3)
        # Cumulative translation at t_idx: linear interp
        cum_t = pert_t * alpha.unsqueeze(-1)                   # (n, 3)

        # ── 4. Apply perturbation to contact pose ─────────────────────
        contact_R_p = contact_R[pending]  # (n, 3, 3)
        contact_t_p = contact_t[pending]  # (n, 3)

        noised_R = torch.bmm(cum_R, contact_R_p)                              # (n, 3, 3)
        noised_t = torch.bmm(cum_R, contact_t_p.unsqueeze(-1)).squeeze(-1) + cum_t  # (n, 3)

        # ── 5. Compute incremental step (denoising target) ────────────
        # Step at t_idx: SLERP at (t_idx-1)/T vs t_idx/T
        alpha_prev = (t_idx.float() - 1).clamp(min=0) / T
        cum_q_prev = _quat_slerp(q_id[:n_pending], pert_q, alpha_prev)
        cum_R_prev = _quat_to_rotmat(cum_q_prev)  # (n, 3, 3)

        # Incremental rotation: cum_q @ inv(cum_q_prev)
        step_q = _quat_multiply(cum_q, _quat_inverse(cum_q_prev))
        step_R = _quat_to_rotmat(step_q)  # (n, 3, 3)

        cum_t_prev = pert_t * alpha_prev.unsqueeze(-1)

        # World-space translations at current and previous timesteps.
        # noised_t  = cum_R   @ contact_t + cum_t     (already computed above)
        # prev_t    = cum_R_prev @ contact_t + cum_t_prev
        # The correct delta is prev_t - noised_t, NOT just -(cum_t - cum_t_prev).
        # The missing term (cum_R_prev - cum_R) @ contact_t can be ~30mm for 9°/step.
        prev_t_world = torch.bmm(cum_R_prev, contact_t_p.unsqueeze(-1)).squeeze(-1) + cum_t_prev

        # Target = inverse of step (rotation) + world-space translation correction
        target_rot = step_R.transpose(1, 2)          # (n, 3, 3)
        target_trans = prev_t_world - noised_t        # (n, 3)  world-space delta

        # t_idx == 0 → identity target (no denoising needed)
        is_zero = (t_idx == 0)
        if is_zero.any():
            noised_R[is_zero] = contact_R_p[is_zero]
            noised_t[is_zero] = contact_t_p[is_zero]
            target_trans[is_zero] = 0.0
            target_rot[is_zero] = torch.eye(3, device=device)

        # ── 6. Rejection: min NN distance guard ──────────────────────
        accepted = torch.ones(n_pending, dtype=torch.bool, device=device)

        if do_guards:
            # Only check non-zero timesteps
            check_mask = ~is_zero
            if check_mask.any():
                # Transform canonical tool to world frame at noised pose
                tool_p = tool_canonical[pending]  # (n, P, 3)
                obj_p  = obj_pc[pending]          # (n, Q, 3)  centered at origin
                tool_world = torch.bmm(tool_p, noised_R.transpose(1, 2)) + noised_t.unsqueeze(1)

                # Reconstruct world-frame object cloud.
                # obj_pc is centered: obj_world = obj_pc + obj_centroid.
                # Without obj_centroid the check fires at the wrong distance (~|obj_centroid|)
                if obj_centroid is not None:
                    obj_world_guard = obj_p + obj_centroid[pending].unsqueeze(1)  # (n, Q, 3)
                else:
                    obj_world_guard = obj_p  # legacy / flat-world fallback

                # Pairwise NN distances (n, P, Q) → min per sample
                dist_matrix = torch.cdist(tool_world, obj_world_guard, p=2)
                min_dists = dist_matrix.min(dim=-1).values.min(dim=-1).values  # (n,)

                # Reject if too close
                rejected = check_mask & (min_dists < pen_threshold)
                accepted[rejected] = False

        # ── 7. Write accepted results ─────────────────────────────────
        pending_idx = pending.nonzero(as_tuple=True)[0]  # absolute indices
        acc_idx = pending_idx[accepted]

        result_t_idx[acc_idx] = t_idx[accepted]
        result_noised_R[acc_idx] = noised_R[accepted]
        result_noised_t[acc_idx] = noised_t[accepted]
        result_target_trans[acc_idx] = target_trans[accepted]
        result_target_rot[acc_idx] = target_rot[accepted]

        # Update pending mask
        pending[acc_idx] = False

    return {
        "t_idx":         result_t_idx,
        "noised_R":      result_noised_R,
        "noised_t":      result_noised_t,
        "target_trans":  result_target_trans,
        "target_rot_mat": result_target_rot,
    }


# ============================================================================ #
# On-the-fly SDF via NN distances
# ============================================================================ #

def _point_mesh_signed_sdf(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        verts.unsqueeze(0), faces
    )
    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.unsqueeze(0).contiguous(), face_verts
    )
    dist = torch.sqrt(sq_dist.squeeze(0).clamp(min=1e-12))
    inside = kaolin.ops.mesh.check_sign(
        verts.unsqueeze(0), faces, points.unsqueeze(0)
    ).squeeze(0)
    return torch.where(inside, -dist, dist)


def compute_on_the_fly_sdf(
    tool_canonical: torch.Tensor,  # (B, P, 3) centered (centroid at origin)
    obj_pc: torch.Tensor,          # (B, Q, 3) centered (obj_centroid subtracted)
    noised_R: torch.Tensor,        # (B, 3, 3) tool rotation at current pose
    noised_t: torch.Tensor,        # (B, 3)   tool centroid world position
    tool_verts: list[torch.Tensor],
    tool_faces: list[torch.Tensor],
    obj_verts: list[torch.Tensor],
    obj_faces: list[torch.Tensor],
    obj_centroid: torch.Tensor = None,  # (B, 3) obj centroid in world frame; None = zeros
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute signed mutual SDF at the actual (noised) tool pose.

    Args:
        tool_canonical: tool surface points in canonical frame (centroid at origin).
        obj_pc:         object surface points, centered (world = obj_pc + obj_centroid).
        noised_R:       tool rotation at current pose.
        noised_t:       tool centroid world position at current pose.
        obj_centroid:   object centroid in world frame. Required to correctly reconstruct
                        world-frame object positions before transforming to tool frame.
                        If None, obj_pc is treated as world-frame (legacy / wrong if != 0).

    Returns:
        tool_sdf: (B, P) signed distance from each (world-frame) tool point to obj mesh.
        obj_sdf:  (B, Q) signed distance from each obj point (in tool canonical frame) to tool mesh.
    """
    B = tool_canonical.shape[0]
    dev = tool_canonical.device
    # Tool points in world frame: p_world = tool_canonical @ R.T + noised_t
    tool_world = torch.bmm(tool_canonical, noised_R.transpose(1, 2)) + noised_t.unsqueeze(1)

    tool_sdfs = []
    obj_sdfs  = []
    for i in range(B):
        obj_v  = obj_verts[i].to(device=dev, dtype=tool_canonical.dtype)
        obj_f  = obj_faces[i].to(device=dev)
        tool_v = tool_verts[i].to(device=dev, dtype=tool_canonical.dtype)
        tool_f = tool_faces[i].to(device=dev)

        # tool → object SDF (tool points in world frame vs object mesh in world frame)
        tool_sdfs.append(_point_mesh_signed_sdf(tool_world[i], obj_v, obj_f))

        # object → tool SDF: transform obj world points to tool canonical frame
        # obj_world = obj_pc + obj_centroid  (reconstruct world positions)
        # tool_frame: p_tool = (obj_world - noised_t) @ R   (row-vector convention)
        if obj_centroid is not None:
            obj_world_i = obj_pc[i] + obj_centroid[i].unsqueeze(0)  # (Q, 3)
        else:
            obj_world_i = obj_pc[i]  # fallback: assume obj_pc already in world frame
        obj_pts_tool = torch.matmul(obj_world_i - noised_t[i].unsqueeze(0), noised_R[i])
        obj_sdfs.append(_point_mesh_signed_sdf(obj_pts_tool, tool_v, tool_f))

    return torch.stack(tool_sdfs), torch.stack(obj_sdfs)

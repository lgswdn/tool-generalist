"""noise_utils.py — SE(3) noising for RPDiff-style iterative pose denoising.

Fully batched PyTorch implementation (no scipy, no per-sample Python loops).
  - Batched quaternion SLERP on GPU
  - Vectorized trajectory interpolation
  - Batch-level rejection sampling via compute_on_the_fly_sdf
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    tool_canonical: torch.Tensor = None,   # (B, P, 3) centered
    obj_pc: torch.Tensor = None,           # (B, Q, 3) centered
    obj_centroid: torch.Tensor = None,     # (B, 3)    object centroid in world
    mesh_cache_entry: dict = None,         # from dataset._mesh_cache[pt_path]
    pen_threshold: float = 0.001,
    max_retries: int = 10,
) -> dict:
    """Fully batched SE(3) noise sampling with GPU rejection (signed SDF).

    All operations are vectorized on GPU — no Python loops over batch items.

    Rejection: tool must not penetrate object by more than pen_threshold.
    Uses kaolin signed SDF when mesh_cache_entry is available;
    falls back to unsigned NN distance otherwise.
    Fallback: rejected items get contact pose (t_idx=0).
    """
    B = contact_R.shape[0]
    device = contact_R.device
    T = num_steps
    do_guards = (
        tool_canonical is not None
        and obj_pc is not None
        and obj_centroid is not None
    )

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

        # ── 6. Rejection: signed SDF penetration guard ───────────────
        accepted = torch.ones(n_pending, dtype=torch.bool, device=device)

        if do_guards:
            # Only check non-zero timesteps (t_idx=0 is the contact pose itself)
            check_mask = ~is_zero
            if check_mask.any():
                tool_p  = tool_canonical[pending]   # (n, P, 3) centered
                obj_p   = obj_pc[pending]            # (n, Q, 3) centered
                obj_cen = obj_centroid[pending]      # (n, 3)

                tool_world = (
                    torch.bmm(tool_p, noised_R.transpose(1, 2))
                    + noised_t.unsqueeze(1)
                )  # (n, P, 3)
                obj_world = obj_p + obj_cen.unsqueeze(1)   # (n, Q, 3)

                if (
                    _KAOLIN_AVAILABLE
                    and mesh_cache_entry is not None
                    and mesh_cache_entry.get("obj_verts") is not None
                ):
                    # Signed SDF: reject only if tool is penetrating the object
                    # (min signed SDF < -pen_threshold  →  inside the object)
                    obj_v = mesh_cache_entry["obj_verts"].to(device)
                    obj_f = mesh_cache_entry["obj_faces"].to(device)
                    fv = kaolin.ops.mesh.index_vertices_by_faces(
                        obj_v.unsqueeze(0), obj_f
                    ).expand(n_pending, -1, -1, -1)
                    sq_d, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        tool_world.contiguous(), fv
                    )
                    dist = torch.sqrt(sq_d.clamp(min=0))   # (n, P)
                    inside = kaolin.ops.mesh.check_sign(
                        obj_v.unsqueeze(0).expand(n_pending, -1, -1),
                        obj_f.long(), tool_world.contiguous(),
                    )  # (n, P)
                    signed = torch.where(inside, -dist, dist)  # (n, P)
                    min_sdf = signed.min(dim=-1).values         # (n,)
                    # Reject if penetrating (min_sdf < -pen_threshold)
                    rejected = check_mask & (min_sdf < -pen_threshold)
                else:
                    # Fallback: reject if any tool point is within pen_threshold
                    # of any object point (approximate, may reject valid poses)
                    dist_matrix = torch.cdist(tool_world, obj_world, p=2)
                    min_dists = dist_matrix.min(dim=-1).values.min(dim=-1).values
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
# Kaolin signed SDF helpers
# ============================================================================ #

try:
    import kaolin.ops.mesh
    import kaolin.metrics.trianglemesh
    _KAOLIN_AVAILABLE = True
except ImportError:
    _KAOLIN_AVAILABLE = False


def _kaolin_signed_sdf(
    points: torch.Tensor,       # (B, P, 3)  query points in mesh frame
    mesh_verts: torch.Tensor,   # (V, 3)
    mesh_faces: torch.Tensor,   # (F, 3) int64
) -> torch.Tensor:
    """Signed SDF: positive = outside mesh, negative = inside."""
    B = points.shape[0]
    device = points.device
    verts = mesh_verts.to(device)
    faces = mesh_faces.to(device)
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        verts.unsqueeze(0), faces
    ).expand(B, -1, -1, -1)          # (B, F, 3, 3)
    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )
    dist = torch.sqrt(sq_dist.clamp(min=0))   # (B, P)
    inside = kaolin.ops.mesh.check_sign(
        verts.unsqueeze(0).expand(B, -1, -1),
        faces.long(),
        points.contiguous(),
    )                                # (B, P) bool  True = inside
    return torch.where(inside, -dist, dist)   # signed: inside < 0


# ============================================================================ #
# On-the-fly SDF (signed, positive = outside)
# ============================================================================ #

def compute_on_the_fly_sdf(
    tool_canonical: torch.Tensor,   # (B, P, 3) centered canonical (R=I)
    obj_pc: torch.Tensor,           # (B, Q, 3) centered at origin
    noised_R: torch.Tensor,         # (B, 3, 3)
    noised_t: torch.Tensor,         # (B, 3)   tool centroid world pos
    obj_centroid: torch.Tensor,     # (B, 3)   object centroid world pos
    mesh_cache_entry: dict = None,  # from dataset._mesh_cache[pt_path]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute signed SDF (positive=outside) for tool and object point clouds.

    Uses kaolin with mesh data when available; falls back to unsigned NN otherwise.

    Convention (NEW):
      - tool_canonical is centered; encoder input is tool_canonical @ noised_R.T
      - obj_pc is centered; its world position is obj_centroid
      - Tool world pts  = tool_canonical @ noised_R.T + noised_t
      - Object world pts = obj_pc + obj_centroid

    Returns:
        tool_sdf: (B, P) signed — positive = tool point is outside the object mesh
        obj_sdf:  (B, Q) signed — positive = obj point is outside the tool mesh
    """
    # Tool points in world frame
    tool_world = (
        torch.bmm(tool_canonical, noised_R.transpose(1, 2))
        + noised_t.unsqueeze(1)
    )  # (B, P, 3)

    # Object points in world frame
    obj_world = obj_pc + obj_centroid.unsqueeze(1)   # (B, Q, 3)

    if (
        _KAOLIN_AVAILABLE
        and mesh_cache_entry is not None
        and mesh_cache_entry.get("obj_verts") is not None
        and mesh_cache_entry.get("tool_verts") is not None
    ):
        obj_verts  = mesh_cache_entry["obj_verts"]    # (V_o, 3) world frame
        obj_faces  = mesh_cache_entry["obj_faces"]
        tool_verts = mesh_cache_entry["tool_verts"]   # (V_t, 3) centered
        tool_faces = mesh_cache_entry["tool_faces"]

        # tool SDF: how far are tool world pts from object surface? (signed)
        tool_sdf = _kaolin_signed_sdf(tool_world, obj_verts, obj_faces)

        # obj SDF: transform obj world pts into tool frame, then query tool mesh
        # tool frame = centered tool at (0,0,0) with noised_R applied
        # world → tool frame: (p - noised_t) @ noised_R  (undo rotation + translation)
        obj_in_tool = torch.bmm(
            obj_world - noised_t.unsqueeze(1),
            noised_R,                              # (B, Q, 3) @ (B, 3, 3)
        )
        obj_sdf = _kaolin_signed_sdf(obj_in_tool, tool_verts, tool_faces)
    else:
        # Fallback: unsigned NN distance (no sign info)
        dist_matrix = torch.cdist(tool_world, obj_world, p=2)
        tool_sdf = dist_matrix.min(dim=-1).values
        obj_sdf  = dist_matrix.min(dim=-2).values

    return tool_sdf, obj_sdf

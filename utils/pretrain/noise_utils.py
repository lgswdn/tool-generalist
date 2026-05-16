"""noise_utils.py — translation-only noising for iterative pose denoising.

Fully batched PyTorch implementation (no scipy, no per-sample Python loops).
  - Vectorized trajectory interpolation
"""

from __future__ import annotations

import torch

from utils.geometry.pose import pose9d_from_rt


def _tool_world_points(tool_points_T: torch.Tensor, rotation_E: torch.Tensor, translation_E: torch.Tensor) -> torch.Tensor:
    return tool_points_T @ rotation_E.T + translation_E.unsqueeze(0)


def _is_pose_legal(
    *,
    tool_points_T: torch.Tensor,
    object_points_O: torch.Tensor,
    object_rotation_E: torch.Tensor,
    object_bbox_center_E: torch.Tensor,
    tool_rotation_E: torch.Tensor,
    tool_translation_E: torch.Tensor,
    floor_eps: float,
    min_separation: float,
) -> bool:
    tool_world = _tool_world_points(tool_points_T, tool_rotation_E, tool_translation_E)
    if bool(tool_world[:, 2].min() < -float(floor_eps)):
        return False
    if min_separation <= 0.0:
        return True
    object_world = object_points_O @ object_rotation_E.T + object_bbox_center_E.unsqueeze(0)
    min_dist = torch.cdist(tool_world.unsqueeze(0), object_world.unsqueeze(0)).min()
    return bool(min_dist >= float(min_separation))


def sample_legal_noised_pose(
    *,
    tool_points_T: torch.Tensor,
    object_points_O: torch.Tensor,
    object_rotation_E: torch.Tensor,
    object_bbox_center_E: torch.Tensor,
    contact_tool_rotation_E: torch.Tensor,
    contact_tool_translation_E: torch.Tensor,
    noise_max_trans: float,
    noise_max_rot_deg: float,
    max_retries: int,
    floor_eps: float,
    min_separation: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a CPU-safe translation-only noised tool pose."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    dtype = tool_points_T.dtype
    device = tool_points_T.device
    for _ in range(max(1, int(max_retries))):
        offset = (torch.rand(3, generator=gen, dtype=dtype, device=device) * 2.0 - 1.0) * float(noise_max_trans)
        offset[2] = offset[2].abs()
        noised_R = contact_tool_rotation_E.clone()
        noised_t = contact_tool_translation_E + offset
        if _is_pose_legal(
            tool_points_T=tool_points_T,
            object_points_O=object_points_O,
            object_rotation_E=object_rotation_E,
            object_bbox_center_E=object_bbox_center_E,
            tool_rotation_E=noised_R,
            tool_translation_E=noised_t,
            floor_eps=floor_eps,
            min_separation=min_separation,
        ):
            return noised_R, noised_t

    # Deterministic fallback: lift away from the object until the CPU guards pass.
    noised_R = contact_tool_rotation_E.clone()
    step = max(float(noise_max_trans), float(min_separation), 1e-3)
    for lift_i in range(1, max(4, int(max_retries)) + 2):
        noised_t = contact_tool_translation_E + torch.tensor(
            [0.0, 0.0, step * lift_i],
            dtype=dtype,
            device=device,
        )
        if _is_pose_legal(
            tool_points_T=tool_points_T,
            object_points_O=object_points_O,
            object_rotation_E=object_rotation_E,
            object_bbox_center_E=object_bbox_center_E,
            tool_rotation_E=noised_R,
            tool_translation_E=noised_t,
            floor_eps=floor_eps,
            min_separation=min_separation,
        ):
            return noised_R, noised_t
    return noised_R, contact_tool_translation_E + torch.tensor([0.0, 0.0, step], dtype=dtype, device=device)


def build_precontact_trajectory(
    *,
    tool_points_T: torch.Tensor,
    object_points_O: torch.Tensor,
    object_rotation_E: torch.Tensor,
    object_bbox_center_E: torch.Tensor,
    contact_tool_rotation_E: torch.Tensor,
    contact_tool_translation_E: torch.Tensor,
    num_precontact_steps: int,
    noise_max_trans: float,
    noise_max_rot_deg: float,
    max_retries: int,
    floor_eps: float,
    min_separation: float,
    seed: int,
    target_mode: str = "one_step",
) -> dict[str, torch.Tensor]:
    """Build K+1 states with k=0 at contact and k=K at the sampled noised pose."""

    if target_mode != "one_step":
        raise ValueError(f"Unsupported denoise target mode {target_mode!r}; expected 'one_step'")
    K = int(num_precontact_steps)
    if K < 0:
        raise ValueError("num_precontact_steps must be non-negative")

    noised_R, noised_t = sample_legal_noised_pose(
        tool_points_T=tool_points_T,
        object_points_O=object_points_O,
        object_rotation_E=object_rotation_E,
        object_bbox_center_E=object_bbox_center_E,
        contact_tool_rotation_E=contact_tool_rotation_E,
        contact_tool_translation_E=contact_tool_translation_E,
        noise_max_trans=noise_max_trans,
        noise_max_rot_deg=noise_max_rot_deg,
        max_retries=max_retries,
        floor_eps=floor_eps,
        min_separation=min_separation,
        seed=seed,
    )

    if K == 0:
        rotations = contact_tool_rotation_E.unsqueeze(0)
        translations = contact_tool_translation_E.unsqueeze(0)
    else:
        alpha = torch.linspace(0.0, 1.0, K + 1, dtype=tool_points_T.dtype, device=tool_points_T.device)
        rotations = contact_tool_rotation_E.unsqueeze(0).expand(K + 1, -1, -1).contiguous()
        translations = (1.0 - alpha).unsqueeze(-1) * contact_tool_translation_E + alpha.unsqueeze(-1) * noised_t

    object_centered_E = object_points_O @ object_rotation_E.T
    object_points_E_k = object_centered_E.unsqueeze(0).expand(K + 1, -1, -1).contiguous()
    tool_points_E_k = torch.stack([tool_points_T @ rotations[k].T for k in range(K + 1)], dim=0)
    rel_tool_object_t_k = translations - object_bbox_center_E.unsqueeze(0)

    targets = []
    identity_R = torch.eye(3, dtype=tool_points_T.dtype, device=tool_points_T.device)
    for k in range(1, K + 1):
        delta_t = translations[k - 1] - translations[k]
        targets.append(pose9d_from_rt(delta_t, identity_R))
    if targets:
        target_tool_denoise_pose9d_k = torch.stack(targets, dim=0)
    else:
        target_tool_denoise_pose9d_k = torch.zeros(0, 9, dtype=tool_points_T.dtype, device=tool_points_T.device)

    return {
        "tool_points_E_k": tool_points_E_k,
        "object_points_E_k": object_points_E_k,
        "rel_tool_object_t_k": rel_tool_object_t_k,
        "tool_rotation_E_k": rotations,
        "tool_translation_E_k": translations,
        "target_tool_denoise_pose9d_k": target_tool_denoise_pose9d_k,
    }


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
    """Fully batched translation-only noise sampling with GPU rejection.

    All operations are vectorized on GPU — no Python loops over batch items.

    Guard: min tool-to-object NN distance >= pen_threshold.
    Fallback: rejected items get contact pose (t_idx=0).
    """
    B = contact_R.shape[0]
    device = contact_R.device
    T = num_steps
    do_guards = (tool_canonical is not None and obj_pc is not None)

    # Pre-allocate results — start as contact pose (safe fallback)
    result_t_idx = torch.zeros(B, dtype=torch.long, device=device)
    result_noised_R = contact_R.clone()
    result_noised_t = contact_t.clone()
    result_target_trans = torch.zeros(B, 3, device=device)
    result_target_rot = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

    # Mask of items that still need a valid sample
    pending = torch.ones(B, dtype=torch.bool, device=device)
    if T <= 0:
        return {
            "t_idx": result_t_idx,
            "noised_R": result_noised_R,
            "noised_t": result_noised_t,
            "target_trans": result_target_trans,
            "target_rot_mat": result_target_rot,
        }

    for _retry in range(max_retries):
        n_pending = pending.sum().item()
        if n_pending == 0:
            break

        # ── 1. Sample perturbations for pending items ─────────────────
        pert_t = (torch.rand(n_pending, 3, device=device) * 2 - 1) * max_trans

        # ── 2. Sample timesteps ───────────────────────────────────────
        if precise_prob:
            weights = torch.exp(-1.0 * torch.arange(T + 1, device=device, dtype=torch.float32))
            t_idx = torch.multinomial(weights.expand(n_pending, -1), 1).squeeze(-1)
        else:
            t_idx = torch.randint(0, T + 1, (n_pending,), device=device)

        # ── 3. Interpolate to sampled timestep ────────────────────────
        alpha = t_idx.float() / T  # (n,) in [0, 1]

        # Cumulative translation at t_idx: linear interp
        cum_t = pert_t * alpha.unsqueeze(-1)                   # (n, 3)

        # ── 4. Apply perturbation to contact pose ─────────────────────
        contact_R_p = contact_R[pending]  # (n, 3, 3)
        contact_t_p = contact_t[pending]  # (n, 3)

        noised_R = contact_R_p.clone()
        noised_t = contact_t_p + cum_t

        # ── 5. Compute incremental step (denoising target) ────────────
        # Step at t_idx: translation at (t_idx-1)/T vs t_idx/T
        alpha_prev = (t_idx.float() - 1).clamp(min=0) / T
        cum_t_prev = pert_t * alpha_prev.unsqueeze(-1)

        prev_t_world = contact_t_p + cum_t_prev

        target_rot = torch.eye(3, device=device).unsqueeze(0).expand(n_pending, -1, -1).clone()
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

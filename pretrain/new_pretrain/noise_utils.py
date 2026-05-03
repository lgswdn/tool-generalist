"""noise_utils.py — SE(3) noising for RPDiff-style iterative pose denoising.

Follows RPDiff's data pipeline exactly:
  - SLERP for rotation interpolation
  - Linear interpolation for translation
  - Incremental per-step targets (inverse of one-step noise)

Reuses rpdiff utilities where possible.
"""

from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# ── SE(3) Perturbation Sampling ─────────────────────────────────────────────

def sample_full_perturbation(
    batch_size: int,
    max_trans: float = 0.15,
    max_rot_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a large SE(3) perturbation for each sample in the batch.

    Args:
        batch_size: Number of perturbations to sample.
        max_trans:  Maximum translation magnitude (metres).
        max_rot_deg: Maximum rotation angle (degrees).

    Returns:
        rot_mats:  (B, 3, 3) rotation matrices.
        trans:     (B, 3) translation vectors.
    """
    # Random rotation: uniform random axis, uniform angle in [0, max_rot_deg]
    angles = np.random.uniform(0, np.deg2rad(max_rot_deg), size=batch_size)
    axes = np.random.randn(batch_size, 3)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    rotvecs = axes * angles[:, None]
    rot_mats = R.from_rotvec(rotvecs).as_matrix()  # (B, 3, 3)

    # Random translation: uniform in a ball of radius max_trans
    trans = np.random.uniform(-max_trans, max_trans, size=(batch_size, 3))

    return rot_mats.astype(np.float32), trans.astype(np.float32)


# ── Trajectory Interpolation (RPDiff-style) ─────────────────────────────────

def interpolate_se3_trajectory(
    R_pert: np.ndarray,     # (3, 3) single perturbation rotation
    t_pert: np.ndarray,     # (3,) single perturbation translation
    num_steps: int,
) -> dict:
    """Create a T+1 step interpolated trajectory from identity to (R_pert, t_pert).

    Follows RPDiff dataio_full_chunked.py L1372-L1405:
      - Translation: linear interpolation
      - Rotation: SLERP (scipy.spatial.transform.Slerp)

    Args:
        R_pert:    (3, 3) full perturbation rotation.
        t_pert:    (3,) full perturbation translation.
        num_steps: T — number of noising steps (trajectory has T+1 entries).

    Returns:
        dict with:
            "cumulative_R": list of (3, 3) — cumulative rotations at each step [0..T]
            "cumulative_t": list of (3,)   — cumulative translations at each step [0..T]
            "incremental_R": list of (3, 3) — per-step rotation deltas [0..T-1]
            "incremental_t": list of (3,)   — per-step translation deltas [0..T-1]
    """
    # Translation: linear interpolation
    trans_interp = np.linspace(np.zeros(3), t_pert, num_steps + 1)  # (T+1, 3)

    # Rotation: SLERP
    quat_identity = np.array([0, 0, 0, 1], dtype=np.float64)  # scipy uses (x, y, z, w)
    quat_pert = R.from_matrix(R_pert).as_quat()
    slerp = Slerp(np.arange(2), R.from_quat([quat_identity, quat_pert]))
    interp_rots = slerp(np.linspace(0, 1, num_steps + 1))
    rotmat_interp = interp_rots.as_matrix()  # (T+1, 3, 3)

    # Compute incremental steps
    incremental_R = []
    incremental_t = []
    for i in range(1, num_steps + 1):
        # step[i] = interp[i] @ inv(interp[i-1])
        delta_R = rotmat_interp[i] @ np.linalg.inv(rotmat_interp[i - 1])
        delta_t = trans_interp[i] - trans_interp[i - 1]
        incremental_R.append(delta_R.astype(np.float32))
        incremental_t.append(delta_t.astype(np.float32))

    return {
        "cumulative_R": [m.astype(np.float32) for m in rotmat_interp],
        "cumulative_t": [t.astype(np.float32) for t in trans_interp],
        "incremental_R": incremental_R,
        "incremental_t": incremental_t,
    }


# ── Batch noising for training ──────────────────────────────────────────────

def sample_noised_poses_batch(
    contact_R: torch.Tensor,     # (B, 3, 3) contact rotations
    contact_t: torch.Tensor,     # (B, 3) contact translations
    num_steps: int,
    max_trans: float,
    max_rot_deg: float,
    interp: bool = True,
    precise_prob: bool = False,
) -> dict:
    """Sample a random diffusion timestep and noised pose for each sample in a batch.

    For each sample:
      1. Sample a full SE(3) perturbation
      2. Interpolate T+1 trajectory via SLERP + linear
      3. Pick a random timestep t_idx
      4. Return the noised pose at t_idx and the one-step denoising target

    Args:
        contact_R: (B, 3, 3) GT contact rotations.
        contact_t: (B, 3) GT contact translations.
        num_steps: T — number of diffusion steps.
        max_trans: Max translation perturbation.
        max_rot_deg: Max rotation perturbation.
        interp: Use SLERP interpolation (True) or random walk (False).
        precise_prob: Bias toward smaller timesteps.

    Returns:
        dict with tensors on same device as inputs:
            "t_idx":        (B,) int — sampled timestep index (0 = contact)
            "noised_R":     (B, 3, 3) — tool rotation at noised pose
            "noised_t":     (B, 3) — tool translation at noised pose
            "target_trans":  (B, 3) — GT translation target (inverse of step noise)
            "target_rot_mat": (B, 3, 3) — GT rotation target (inverse of step noise)
    """
    B = contact_R.shape[0]
    device = contact_R.device

    # Sample full perturbations (numpy)
    pert_R, pert_t = sample_full_perturbation(B, max_trans, max_rot_deg)

    # For each sample, build trajectory and pick a timestep
    all_t_idx = []
    all_noised_R = []
    all_noised_t = []
    all_target_trans = []
    all_target_rot = []

    contact_R_np = contact_R.detach().cpu().numpy()
    contact_t_np = contact_t.detach().cpu().numpy()

    for b in range(B):
        traj = interpolate_se3_trajectory(pert_R[b], pert_t[b], num_steps)

        # Sample timestep (RPDiff-style)
        if precise_prob:
            diff_vals = np.exp(-1.0 * np.arange(num_steps + 1))
            probs = diff_vals / diff_vals.sum()
            t_idx = np.where(np.random.multinomial(1, probs))[0][0]
        else:
            t_idx = np.random.randint(0, num_steps + 1)

        all_t_idx.append(t_idx)

        if t_idx == 0:
            # At contact pose — noise target is identity (zero step)
            all_noised_R.append(contact_R_np[b])
            all_noised_t.append(contact_t_np[b])
            all_target_trans.append(np.zeros(3, dtype=np.float32))
            all_target_rot.append(np.eye(3, dtype=np.float32))
        else:
            # Apply cumulative perturbation to contact pose
            cum_R = traj["cumulative_R"][t_idx]
            cum_t = traj["cumulative_t"][t_idx]

            # Noised pose = perturbation applied to contact
            # Following RPDiff: rotate child around its centroid, then translate
            noised_R = cum_R @ contact_R_np[b]
            noised_t = cum_R @ contact_t_np[b] + cum_t

            all_noised_R.append(noised_R)
            all_noised_t.append(noised_t)

            # Target: inverse of the incremental step (RPDiff convention)
            step_R = traj["incremental_R"][t_idx - 1]
            step_t = traj["incremental_t"][t_idx - 1]
            all_target_trans.append(-1.0 * step_t)
            all_target_rot.append(np.linalg.inv(step_R).astype(np.float32))

    return {
        "t_idx":         torch.tensor(all_t_idx, dtype=torch.long, device=device),
        "noised_R":      torch.tensor(np.stack(all_noised_R), dtype=torch.float32, device=device),
        "noised_t":      torch.tensor(np.stack(all_noised_t), dtype=torch.float32, device=device),
        "target_trans":  torch.tensor(np.stack(all_target_trans), dtype=torch.float32, device=device),
        "target_rot_mat": torch.tensor(np.stack(all_target_rot), dtype=torch.float32, device=device),
    }


# ── On-the-fly SDF via NN distances ─────────────────────────────────────────

def compute_on_the_fly_sdf(
    tool_canonical: torch.Tensor,  # (B, P, 3) canonical tool points
    obj_pc: torch.Tensor,          # (B, Q, 3) object points (world frame)
    noised_R: torch.Tensor,        # (B, 3, 3)
    noised_t: torch.Tensor,        # (B, 3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute approximate unsigned mutual SDF using point-to-point NN distance.

    For each noised pose, transform canonical tool to world frame and compute
    nearest-neighbor distances between tool and object point clouds.

    Args:
        tool_canonical: (B, P, 3) canonical tool points (origin, R=I).
        obj_pc:         (B, Q, 3) object points in world frame.
        noised_R:       (B, 3, 3) noised rotation.
        noised_t:       (B, 3) noised translation.

    Returns:
        tool_sdf: (B, P) unsigned NN distance from each tool point to object.
        obj_sdf:  (B, Q) unsigned NN distance from each object point to tool.
    """
    # Transform canonical tool to world frame at noised pose
    tool_world = torch.bmm(tool_canonical, noised_R.transpose(1, 2)) + noised_t.unsqueeze(1)

    # Pairwise distances: (B, P, Q)
    dist_matrix = torch.cdist(tool_world, obj_pc, p=2)

    # Tool-to-object: min over object points
    tool_sdf = dist_matrix.min(dim=-1).values  # (B, P)

    # Object-to-tool: min over tool points
    obj_sdf = dist_matrix.min(dim=-2).values   # (B, Q)

    return tool_sdf, obj_sdf

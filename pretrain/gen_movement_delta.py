#!/usr/bin/env python3
"""
gen_movement_delta.py — Generate (ΔT, ΔO) movement-delta pairs.

Given existing contact configurations in .pt files (from contact_gen.py),
for each config:
  1. Pick a contact point P from the stored contact_pts_world.
  2. Apply a small penetrating transform ΔT to the tool (translation toward
     object interior + small random rotation).
  3. Optimise a full SE(3) ΔO for the object so that:
       - Penetration between tool-new and object-new is resolved.
       - The contact point P (which moved rigidly with the tool) stays on
         the object surface.
       - ΔO itself is regularised to be small (penalise unnecessary motion).

The result is appended to the same .pt file:
    delta_tool_translations   (N, 3)
    delta_tool_rotations      (N, 3, 3)
    delta_obj_translations    (N, 3)
    delta_obj_rotations       (N, 3, 3)
    movement_contact_pts      (N, 3)      the anchor contact point used

Usage:
    python gen_movement_delta.py --input contact_configs.pt
    python gen_movement_delta.py --input-dir tmp_data/ --device cuda:0
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import trimesh

try:
    import kaolin
    import kaolin.ops.mesh
    import kaolin.metrics.trianglemesh
except ImportError:
    sys.exit("kaolin is required:  pip install kaolin")

from contact_config import MOVEMENT_DELTA, MovementDeltaHyperparams

# ═══════════════════════════════════════════════════════════════════════════════
#  Rotation utilities  (shared with contact_gen.py)
# ═══════════════════════════════════════════════════════════════════════════════

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """6-D continuous representation → (*, 3, 3) rotation matrix."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(*, 3, 3) rotation matrix → (*, 6) 6-D representation."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix (Rodrigues).

    Args:
        axis:  (*, 3)  unit vectors
        angle: (*,)    radians

    Returns:
        R: (*, 3, 3)
    """
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] =  axis[..., 1]
    K[..., 1, 0] =  axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] =  axis[..., 0]
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    angle = angle[..., None, None]
    R = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


# ═══════════════════════════════════════════════════════════════════════════════
#  Mesh helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_mesh(path: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def compute_unsigned_distance(
    points: torch.Tensor,   # (B, P, 3)
    verts: torch.Tensor,    # (V, 3)
    faces: torch.Tensor,    # (F, 3)
) -> torch.Tensor:          # (B, P)
    B = points.shape[0]
    fv = kaolin.ops.mesh.index_vertices_by_faces(
        verts.unsqueeze(0), faces
    ).expand(B, -1, -1, -1)
    sq, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), fv
    )
    return torch.sqrt(sq.clamp(min=1e-12))


def compute_sign(
    points: torch.Tensor,   # (B, P, 3)
    verts: torch.Tensor,    # (V, 3)
    faces: torch.Tensor,    # (F, 3)
) -> torch.Tensor:          # (B, P) bool
    B = points.shape[0]
    return kaolin.ops.mesh.check_sign(
        verts.unsqueeze(0).expand(B, -1, -1), faces, points
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ΔT sampling  (small penetrating push)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_delta_T_down_only(
    contact_pts: torch.Tensor,   # (N, 3) one contact point per config
    obj_verts: torch.Tensor,     # (V, 3) object mesh vertices
    obj_faces: torch.Tensor,     # (F, 3) object mesh faces
    hp: MovementDeltaHyperparams,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a small rigid-body ΔT that pushes the tool straight down.

    The push direction is purely -z (downward), with no rotation.

    Returns:
        delta_t : (N, 3)    translation component (downward only)
        delta_R : (N, 3, 3) rotation component (identity)
    """
    N = contact_pts.shape[0]

    # ---- Translation: straight down (-z direction) ----
    magnitude = torch.empty(N, 1, device=device).uniform_(0.005, 0.01)  # 5-10mm
    down_direction = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).expand(N, -1)
    delta_t = down_direction * magnitude  # (N, 3)

    # ---- Rotation: identity (no rotation) ----
    delta_R = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()  # (N, 3, 3)

    return delta_t, delta_R


def sample_delta_T(
    contact_pts: torch.Tensor,   # (N, 3) one contact point per config
    obj_verts: torch.Tensor,     # (V, 3) object mesh vertices
    obj_faces: torch.Tensor,     # (F, 3) object mesh faces
    hp: MovementDeltaHyperparams,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a small rigid-body ΔT that pushes the tool toward the object.

    Uses rejection sampling:
    1. Start with inward face normal at each contact point
    2. Add large perturbation for diversity
    3. Check if push direction causes penetration (SDF decrease)
    4. Accept only if penetration is achieved

    Returns:
        delta_t : (N, 3)    translation component
        delta_R : (N, 3, 3) rotation component
    """
    N = contact_pts.shape[0]

    # ---- Compute inward surface normals at contact points ----
    fv = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    )  # (1, F, 3, 3)

    inward_normals_list = []
    for i in range(N):
        pt = contact_pts[i:i+1].unsqueeze(0)  # (1, 1, 3)
        sq_dist, face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            pt.contiguous(), fv
        )
        face_idx_i = face_idx.squeeze().item()
        face_verts = fv[0, face_idx_i]  # (3, 3)

        v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
        face_normal = F.normalize(torch.cross(v1 - v0, v2 - v0), dim=-1)

        obj_center = obj_verts.mean(dim=0)
        center_to_face = (face_verts.mean(dim=0) - obj_center)
        if (face_normal @ center_to_face).item() > 0:
            inward_normal = -face_normal
        else:
            inward_normal = face_normal

        inward_normals_list.append(inward_normal.unsqueeze(0))

    inward_normals = torch.cat(inward_normals_list, dim=0)  # (N, 3)

    # ---- Rejection sampling for push directions ----
    # Sample perturbed directions and check if they cause penetration
    push_directions = torch.zeros(N, 3, device=device)
    perturbation_scale = 1.0  # large perturbation for diversity
    max_attempts = 50
    test_distance = 0.005  # small step to test SDF change

    for i in range(N):
        inward_n = inward_normals[i]  # (3,)
        accepted = False

        for attempt in range(max_attempts):
            # Sample perturbed direction around inward normal
            perturbation = torch.randn(3, device=device) * perturbation_scale
            candidate_dir = F.normalize(inward_n + perturbation, dim=-1)

            # Test SDF change: move contact point slightly in candidate direction
            test_pt = contact_pts[i] + candidate_dir * test_distance

            # Check if this point is inside the object (penetration achieved)
            inside = kaolin.ops.mesh.check_sign(
                obj_verts.unsqueeze(0), obj_faces, test_pt.unsqueeze(0).unsqueeze(0)
            )

            if inside.squeeze().item():
                # This direction causes penetration - accept it
                push_directions[i] = candidate_dir
                accepted = True
                break

        if not accepted:
            # Fallback to inward normal if no valid direction found
            push_directions[i] = inward_n

    # ---- Translation: along accepted push direction ----
    magnitude = torch.empty(N, 1, device=device).uniform_(hp.delta_t_min, hp.delta_t_max)
    delta_t = push_directions * magnitude  # (N, 3)

    # ---- Rotation: small random axis-angle ----
    axis = F.normalize(torch.randn(N, 3, device=device), dim=-1)
    angle_min = math.radians(hp.delta_r_min_deg)
    angle_max = math.radians(hp.delta_r_max_deg)
    angle = torch.empty(N, device=device).uniform_(angle_min, angle_max)
    delta_R = axis_angle_to_matrix(axis, angle)  # (N, 3, 3)

    return delta_t, delta_R


# ═══════════════════════════════════════════════════════════════════════════════
#  ΔO Optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_delta_O(
    tool_pts_new: torch.Tensor,     # (N, P, 3) tool cloud after ΔT applied
    obj_verts: torch.Tensor,        # (V, 3) object mesh (original world frame)
    obj_faces: torch.Tensor,        # (F, 3)
    P_anchor_new: torch.Tensor,     # (N, 3) contact point after moving with tool
    contact_pts_original: torch.Tensor,  # (N, 3) original contact points on object surface
    hp: MovementDeltaHyperparams,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimise ΔO (rotation only) to resolve penetration from ΔT.

    The object transform is parameterised as rotation around the anchor:
        obj_verts_new = delta_R @ (obj_verts - P_anchor_new) + P_anchor_new + delta_trans

    where delta_trans is computed analytically from delta_R to guarantee
    the contact point lands exactly at P_anchor_new:
        delta_trans = -delta_R @ (contact_pts_original - P_anchor_new)

    This makes the anchor constraint a hard constraint (always satisfied exactly).

    Returns:
        delta_obj_t : (N, 3)     computed translation delta (from rotation)
        delta_obj_R : (N, 3, 3)  optimised object rotation delta
        final_pen   : (N,)       final max penetration per config
        final_floor : (N,)       final max floor violation per config
    """
    N = tool_pts_new.shape[0]

    # DEBUG: Print z-coordinates
    print(f"    DEBUG anchor z (after ΔT): {P_anchor_new[:, 2].mean().item():.4f}")
    print(f"    DEBUG contact_pts z (original): {contact_pts_original[:, 2].mean().item():.4f}")

    # Optimisation parameters: only rot6d (identity init)
    identity_6d = matrix_to_rot6d(torch.eye(3, device=device)).unsqueeze(0).expand(N, -1)
    delta_rot6d = identity_6d.clone().detach().requires_grad_(True)    # (N, 6)

    optimiser = torch.optim.Adam([delta_rot6d], lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=hp.opt_steps, eta_min=hp.lr * 0.01,
    )

    # ---- Print initial state (before optimization, step -1) ----
    with torch.no_grad():
        delta_R_init = rot6d_to_matrix(delta_rot6d)
        # Compute delta_trans analytically: for contact_pts_original to land at P_anchor_new
        delta_trans_init = -torch.einsum("nij, nj -> ni", delta_R_init, contact_pts_original - P_anchor_new)

        obj_expanded_init = obj_verts.unsqueeze(0).expand(N, -1, -1)
        pivot_init = P_anchor_new.unsqueeze(1)
        obj_new_init = torch.einsum(
            "nij, nvj -> nvi",
            delta_R_init,
            obj_expanded_init - pivot_init,
        ) + pivot_init + delta_trans_init.unsqueeze(1)

        L_pen_init = torch.tensor(0.0, device=device)
        L_contact_init = torch.tensor(0.0, device=device)
        L_floor_init = torch.tensor(0.0, device=device)

        for i in range(N):
            pts_i = tool_pts_new[i:i+1]
            fv_i = kaolin.ops.mesh.index_vertices_by_faces(obj_new_init[i:i+1], obj_faces)
            sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts_i.contiguous(), fv_i)
            dist = torch.sqrt(sq_dist.clamp(min=1e-12))
            inside = kaolin.ops.mesh.check_sign(obj_new_init[i:i+1], obj_faces, pts_i)

            pen_dist = torch.where(inside, dist, torch.zeros_like(dist))
            K_pen = min(4, pen_dist.shape[-1])
            topk_pen, _ = torch.topk(pen_dist, K_pen, dim=-1, largest=True)
            L_pen_init = L_pen_init + topk_pen.mean()

            masked_dist = torch.where(~inside, dist, torch.full_like(dist, float("inf")))
            K_c = min(hp.k_closest, masked_dist.shape[-1])
            topk_c, _ = torch.topk(masked_dist, K_c, dim=-1, largest=False)
            topk_c = torch.where(topk_c.isinf(), torch.zeros_like(topk_c), topk_c)
            L_contact_init = L_contact_init + topk_c.mean()

            # Object floor: penalize deepest violations (top-k vertices below z=0)
            obj_z_below = F.relu(-obj_new_init[i, :, 2])  # (V,)
            K_floor = min(4, obj_z_below.shape[-1])
            topk_floor, _ = torch.topk(obj_z_below, K_floor, dim=-1, largest=True)
            L_floor_init = L_floor_init + topk_floor.mean()

        L_pen_init = L_pen_init / N
        L_contact_init = L_contact_init / N
        L_floor_init = L_floor_init / N

        eye = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        L_reg_rot_init = ((delta_R_init - eye) ** 2).sum(dim=(-2, -1)).mean()

        total_init = (
            hp.w_pen * L_pen_init
            + hp.w_contact * L_contact_init
            + hp.w_obj_floor * L_floor_init
            + hp.w_reg_rot * L_reg_rot_init
        )

        print(
            f"    INIT (step -1)  |  "
            f"total {total_init.item():.5f}  "
            f"pen {L_pen_init.item():.5f}  "
            f"contact {L_contact_init.item():.5f}  "
            f"reg_r {L_reg_rot_init.item():.5f}  "
            f"floor {L_floor_init.item():.5f}"
        )

    for step in range(hp.opt_steps):
        optimiser.zero_grad()

        # ---- Apply ΔO to object mesh ----
        delta_R = rot6d_to_matrix(delta_rot6d)  # (N, 3, 3)
        # Compute delta_trans analytically: for contact_pts_original to land at P_anchor_new
        delta_trans = -torch.einsum("nij, nj -> ni", delta_R, contact_pts_original - P_anchor_new)  # (N, 3)

        # Pivot around each config's anchor point
        obj_expanded = obj_verts.unsqueeze(0).expand(N, -1, -1)  # (N, V, 3)
        pivot = P_anchor_new.unsqueeze(1)  # (N, 1, 3)
        obj_new = torch.einsum(
            "nij, nvj -> nvi",
            delta_R,
            obj_expanded - pivot,
        ) + pivot + delta_trans.unsqueeze(1)  # (N, V, 3)

        # ---- Losses ----
        # 1. Penetration: tool points inside moved object
        # We query each tool_pts_new[i] against obj_new[i].
        # Since per-config object meshes differ, we loop over N.
        L_pen_total = torch.tensor(0.0, device=device)
        L_contact_total = torch.tensor(0.0, device=device)
        L_obj_floor_total = torch.tensor(0.0, device=device)

        for i in range(N):
            # Point → moved-object distance
            pts_i = tool_pts_new[i:i+1]  # (1, P, 3)
            fv_i = kaolin.ops.mesh.index_vertices_by_faces(
                obj_new[i:i+1], obj_faces,
            )  # (1, F, 3, 3)
            sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                pts_i.contiguous(), fv_i,
            )
            dist = torch.sqrt(sq_dist.clamp(min=1e-12))  # (1, P)
            inside = kaolin.ops.mesh.check_sign(
                obj_new[i:i+1], obj_faces, pts_i,
            )  # (1, P) bool

            # Penetration
            pen_dist = torch.where(inside, dist, torch.zeros_like(dist))
            K_pen = min(4, pen_dist.shape[-1])
            topk_pen, _ = torch.topk(pen_dist, K_pen, dim=-1, largest=True)
            L_pen_total = L_pen_total + topk_pen.mean()

            # Contact attraction (outside points toward surface)
            masked_dist = torch.where(~inside, dist, torch.full_like(dist, float("inf")))
            K_c = min(hp.k_closest, masked_dist.shape[-1])
            topk_c, _ = torch.topk(masked_dist, K_c, dim=-1, largest=False)
            topk_c = torch.where(topk_c.isinf(), torch.zeros_like(topk_c), topk_c)
            L_contact_total = L_contact_total + topk_c.mean()

            # Object floor: penalize deepest violations (top-k vertices below z=0)
            obj_z_below = F.relu(-obj_new[i, :, 2])  # (V,)
            K_floor = min(4, obj_z_below.shape[-1])
            topk_floor, _ = torch.topk(obj_z_below, K_floor, dim=-1, largest=True)
            L_obj_floor_total = L_obj_floor_total + topk_floor.mean()

        L_pen = L_pen_total / N
        L_contact = L_contact_total / N
        L_floor = L_obj_floor_total / N

        # 2. Regularisation: penalise ΔO rotation magnitude
        # Rotation deviation from identity (Frobenius norm of R - I)
        eye = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        L_reg_rot = ((delta_R - eye) ** 2).sum(dim=(-2, -1)).mean()

        # ---- Total ----
        total = (
            hp.w_pen * L_pen
            + hp.w_contact * L_contact
            + hp.w_obj_floor * L_floor
            + hp.w_reg_rot * L_reg_rot
        )

        total.backward()
        optimiser.step()
        scheduler.step()

        if step % 20 == 0 or step == hp.opt_steps - 1:
            print(
                f"    step {step:4d}  |  "
                f"total {total.item():.5f}  "
                f"pen {L_pen.item():.5f}  "
                f"contact {L_contact.item():.5f}  "
                f"reg_r {L_reg_rot.item():.5f}  "
                f"floor {L_floor.item():.5f}"
            )

    # ---- Extract final ΔO and compute per-config losses ----
    with torch.no_grad():
        delta_obj_R = rot6d_to_matrix(delta_rot6d)  # (N, 3, 3)
        delta_obj_t = -torch.einsum("nij, nj -> ni", delta_obj_R, contact_pts_original - P_anchor_new)  # (N, 3)

        # Compute final per-config penetration and floor violations
        obj_expanded = obj_verts.unsqueeze(0).expand(N, -1, -1)
        pivot = P_anchor_new.unsqueeze(1)
        obj_new = torch.einsum(
            "nij, nvj -> nvi",
            delta_obj_R,
            obj_expanded - pivot,
        ) + pivot + delta_obj_t.unsqueeze(1)

        final_pen_list = []
        final_floor_list = []

        for i in range(N):
            pts_i = tool_pts_new[i:i+1]
            fv_i = kaolin.ops.mesh.index_vertices_by_faces(obj_new[i:i+1], obj_faces)
            sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts_i.contiguous(), fv_i)
            dist = torch.sqrt(sq_dist.clamp(min=1e-12))
            inside = kaolin.ops.mesh.check_sign(obj_new[i:i+1], obj_faces, pts_i)

            # Max penetration depth for this config
            pen_dist = torch.where(inside, dist, torch.zeros_like(dist))
            max_pen = pen_dist.max()
            final_pen_list.append(max_pen.unsqueeze(0))

            # Max floor violation for this config
            obj_z_below = F.relu(-obj_new[i, :, 2])
            max_floor = obj_z_below.max()
            final_floor_list.append(max_floor.unsqueeze(0))

        final_pen = torch.cat(final_pen_list, dim=0)    # (N,)
        final_floor = torch.cat(final_floor_list, dim=0)  # (N,)

    return delta_obj_t, delta_obj_R, final_pen, final_floor


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-file processing
# ═══════════════════════════════════════════════════════════════════════════════

def process_pt_file(
    pt_path: str,
    device: str,
    hp: MovementDeltaHyperparams,
    seed: int,
    force: bool = False,
) -> bool:
    """Generate (ΔT, ΔO) for every config in a .pt file.

    Modifies the .pt file in-place, adding the movement delta fields.
    Returns True on success.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    N = data["tool_translations"].shape[0]

    if "delta_tool_translations" in data and not force:
        print(f"  [SKIP] {pt_path} already has movement deltas")
        return True

    print(f"  Processing {pt_path}: {N} configs")

    # ---- Load meshes ----
    tool_path = data["tool_mesh_path"]
    obj_path = data["object_mesh_path"]

    tool_verts_raw, tool_faces = load_mesh(tool_path, device)
    obj_verts_raw, obj_faces = load_mesh(obj_path, device)

    tool_scale = data.get("tool_scale", 0.1)
    obj_scale = data.get("object_scale", 0.15)
    tool_verts = tool_verts_raw * tool_scale
    obj_verts_scaled = obj_verts_raw * obj_scale

    # ---- Ground object (same transform as contact_gen) ----
    R_obj = data["object_rotation"].to(device)  # (3, 3)
    z_shift = data["obj_z_shift"]
    if isinstance(z_shift, torch.Tensor):
        z_shift = z_shift.to(device)

    obj_verts = obj_verts_scaled @ R_obj.T
    obj_verts = obj_verts.clone()
    obj_verts[:, 2] -= z_shift

    # ---- Canonical tool cloud ----
    P_tool = data["tool_pts_canonical"].to(device)  # (P, 3)

    torch.manual_seed(seed)

    # ---- Process each config ----
    all_delta_tool_t = []
    all_delta_tool_R = []
    all_delta_obj_t = []
    all_delta_obj_R = []
    all_anchor_pts = []
    all_final_pen = []
    all_final_floor = []

    # Process configs in small batches to balance memory/speed
    batch_size = N
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        b = end - start
        print(f"    Configs {start}–{end-1} / {N}")

        # ---- 1. Get contact points & tool poses ----
        tool_t = data["tool_translations"][start:end].to(device)   # (b, 3)
        tool_R = data["tool_rotations"][start:end].to(device)      # (b, 3, 3)
        contact_pts = data["contact_pts_world"][start:end, 0].to(device)  # (b, 3) — use first contact pt

        # ---- 2. Sample ΔT ----
        delta_t, delta_R = sample_delta_T(contact_pts, obj_verts, obj_faces, hp, device)
        # delta_t: (b, 3), delta_R: (b, 3, 3)

        # ---- 3. Apply ΔT to tool ----
        # New tool pose:  R_new = ΔR @ R,  t_new = ΔR @ t + Δt
        tool_R_new = delta_R @ tool_R   # (b, 3, 3)
        tool_t_new = torch.einsum("nij, nj -> ni", delta_R, tool_t) + delta_t  # (b, 3)

        # Transform tool cloud to new world frame
        tool_pts_new = torch.einsum(
            "pi, nji -> npj", P_tool, tool_R_new,
        ) + tool_t_new.unsqueeze(1)  # (b, P, 3)

        # ---- 4. Move anchor point with tool ----
        # P_anchor is in world frame.  Under ΔT: P_new = ΔR @ P + Δt
        P_anchor_new = torch.einsum("nij, nj -> ni", delta_R, contact_pts) + delta_t  # (b, 3)

        # ---- 5. Optimise ΔO ----
        delta_obj_t, delta_obj_R, final_pen, final_floor = optimise_delta_O(
            tool_pts_new, obj_verts, obj_faces, P_anchor_new, contact_pts, hp, device,
        )

        all_delta_tool_t.append(delta_t.cpu())
        all_delta_tool_R.append(delta_R.cpu())
        all_delta_obj_t.append(delta_obj_t.cpu())
        all_delta_obj_R.append(delta_obj_R.cpu())
        all_anchor_pts.append(P_anchor_new.cpu())
        all_final_pen.append(final_pen.cpu())
        all_final_floor.append(final_floor.cpu())

    # ---- Concatenate ----
    delta_tool_t_all = torch.cat(all_delta_tool_t, dim=0)   # (N, 3)
    delta_tool_R_all = torch.cat(all_delta_tool_R, dim=0)   # (N, 3, 3)
    delta_obj_t_all = torch.cat(all_delta_obj_t, dim=0)    # (N, 3)
    delta_obj_R_all = torch.cat(all_delta_obj_R, dim=0)    # (N, 3, 3)
    anchor_pts_all = torch.cat(all_anchor_pts, dim=0)     # (N, 3)
    final_pen_all = torch.cat(all_final_pen, dim=0)       # (N,)
    final_floor_all = torch.cat(all_final_floor, dim=0)   # (N,)

    # ---- Filter by thresholds ----
    valid_mask = (final_pen_all < hp.pen_max_eps) & (final_floor_all < hp.floor_max_eps)
    n_valid = valid_mask.sum().item()
    n_invalid = N - n_valid

    print(f"    Valid: {n_valid}/{N} configs passed (pen<{hp.pen_max_eps*1000:.1f}mm, floor<{hp.floor_max_eps*1000:.1f}mm)")
    if n_invalid > 0:
        print(f"    {n_invalid} configs marked as invalid (movement_delta_valid=False)")

    # ---- Save all configs, mark invalid ones ----
    data["delta_tool_translations"] = delta_tool_t_all
    data["delta_tool_rotations"]    = delta_tool_R_all
    data["delta_obj_translations"]  = delta_obj_t_all
    data["delta_obj_rotations"]     = delta_obj_R_all
    data["movement_contact_pts"]    = anchor_pts_all
    data["movement_delta_valid"]    = valid_mask          # (N,) bool tensor
    data["movement_final_pen"]      = final_pen_all       # (N,)
    data["movement_final_floor"]    = final_floor_all     # (N,)

    torch.save(data, pt_path)
    print(f"    ✓ Saved movement deltas to {pt_path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker for multi-GPU
# ═══════════════════════════════════════════════════════════════════════════════

def worker(files, gpu, seed, force):
    """Process a subset of files on a specific GPU."""
    device = f"cuda:{gpu}"
    hp = MOVEMENT_DELTA

    ok = fail = skip = 0
    failed_files = []
    for f in files:
        try:
            success = process_pt_file(f, device, hp, seed, force)
            if success:
                ok += 1
            else:
                skip += 1
        except Exception as e:
            import traceback
            err_str = f"{f}: {e}\n" + traceback.format_exc()
            failed_files.append(err_str)
            fail += 1

    return ok, fail, skip, failed_files
            traceback.print_exc()
            fail += 1

    return ok, fail, skip


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate (ΔT, ΔO) movement-delta pairs for contact configs.",
    )
    parser.add_argument("--input", type=str, help="Single .pt file")
    parser.add_argument("--input-dir", type=str, help="Directory of .pt files")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="GPU IDs to use (e.g., --gpus 0 1 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if movement deltas already exist")
    args = parser.parse_args()

    hp = MOVEMENT_DELTA  # read from contact_config.py

    if args.input:
        files = [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    else:
        print("ERROR: Must provide --input or --input-dir")
        sys.exit(1)

    print(f"Movement Delta Generator")
    print(f"  Files     : {len(files)}")
    print(f"  GPUs      : {args.gpus}")
    print(f"  ΔT range  : {hp.delta_t_min*1000:.1f}–{hp.delta_t_max*1000:.1f} mm, "
          f"{hp.delta_r_min_deg:.1f}–{hp.delta_r_max_deg:.1f}°")
    print(f"  ΔO optim  : {hp.opt_steps} steps, lr={hp.lr}  (rotation only, trans computed from anchor)")
    print(f"  Weights   : pen={hp.w_pen} contact={hp.w_contact} floor={hp.w_obj_floor} reg_r={hp.w_reg_rot}")
    print()

    # Distribute files across GPUs (round-robin)
    n_gpus = len(args.gpus)
    subsets = [[] for _ in range(n_gpus)]
    for i, f in enumerate(files):
        subsets[i % n_gpus].append(f)

    if n_gpus == 1:
        # Single GPU: run inline
        ok, fail, skip = worker(subsets[0], args.gpus[0], args.seed, args.force)
    else:
        # Multi-GPU: one subprocess per GPU
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        with mp.Pool(n_gpus) as pool:
            results = pool.starmap(worker, [
                (subsets[i], args.gpus[i], args.seed, args.force)
                for i in range(n_gpus)
            ])
        ok   = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)
        skip = sum(r[2] for r in results)

    print(f"\nDone.  ✓ {ok}  ✗ {fail}  ⟳ {skip}")


if __name__ == "__main__":
    main()

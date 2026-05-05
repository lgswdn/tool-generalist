#!/usr/bin/env python3
"""
contact_gen.py  –  Batched, GPU-accelerated contact configuration generator.

Given a watertight Object mesh and a Tool mesh, this script:
  1. Randomises the object pose and grounds it (z_min = 0).
  2. Initialises N random tool poses satisfying floor & orientation constraints.
  3. Runs differentiable Adam optimisation (via Kaolin distance queries) to
     resolve penetrations and snap the tool to the object surface.
  4. Filters & saves the converged (object_pose, tool_pose) pairs.

Rotation representation: 6-D continuous representation (Zhou et al., 2019)
  to avoid singularities during gradient-based optimisation.

Dependencies:
  pip install torch kaolin trimesh numpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

# --------------- Kaolin imports ---------------
import kaolin
import kaolin.ops.mesh
import kaolin.metrics.trianglemesh

from contact_config import CONTACT_GEN

# ==============================================================================
#                          CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Per-run configuration: I/O paths, device, seed, and batch controls.

    All loss weights, thresholds, and learning-rate defaults come from
    contact_config.CONTACT_GEN so they live in ONE canonical place.
    """

    # ── I/O ────────────────────────────────────────────────────────────────
    object_mesh_path: str = "object.obj"
    tool_mesh_path:   str = "tool.obj"
    output_path:      str = "contact_configs.pt"
    save_init_path:   str = ""   # if set, dump initial poses (pre-optimisation)
    tools_json_path:  str = ""   # path to tools_adjusted.json (for head_area)

    # ── Batch & runtime ──────────────────────────────────────────────────────
    batch_size: int = 512
    device:     str = "cuda:0"
    seed:       int = 42

    # ── From contact_config.CONTACT_GEN (edit there to change defaults) ────────
    tool_scale:            float = CONTACT_GEN.tool_scale
    object_scale_range:    tuple  = CONTACT_GEN.object_scale_range
    num_tool_surface_pts:  int   = CONTACT_GEN.num_surface_pts
    contact_mode_prob:     float = CONTACT_GEN.contact_mode_prob
    opt_steps:             int   = CONTACT_GEN.opt_steps
    lr:                    float = CONTACT_GEN.lr
    w_pen:                 float = CONTACT_GEN.w_pen
    w_contact:             float = CONTACT_GEN.w_contact
    w_floor:               float = CONTACT_GEN.w_floor
    w_upright:             float = CONTACT_GEN.w_upright
    upright_threshold:     float = CONTACT_GEN.upright_threshold
    k_closest:             int   = CONTACT_GEN.k_closest
    pen_max_eps:           float = CONTACT_GEN.pen_max_eps
    contact_eps:           float = CONTACT_GEN.contact_eps


# ==============================================================================
#                     ROTATION UTILITIES  (6-D continuous)
# ==============================================================================

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6-D rotation representation to 3×3 rotation matrices.

    Args:
        rot6d: (*, 6)  –  first two columns of a rotation matrix, flattened.

    Returns:
        R: (*, 3, 3)
    """
    shape = rot6d.shape[:-1]
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    # Gram-Schmidt
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (*, 3, 3)  columns = basis


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Extract the 6-D representation from a rotation matrix.

    Args:
        R: (*, 3, 3)

    Returns:
        rot6d: (*, 6)
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def random_rotation_matrices(n: int, device: str) -> torch.Tensor:
    """Uniform random SO(3) matrices via QR decomposition.

    Returns:
        R: (n, 3, 3)
    """
    H = torch.randn(n, 3, 3, device=device)
    Q, R_ = torch.linalg.qr(H)
    # Ensure det(Q) = +1
    signs = torch.sign(torch.diagonal(R_, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0] *= -1
    return Q


# ==============================================================================
#                        MESH LOADING HELPERS
# ==============================================================================

def load_mesh(path: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a triangle mesh and return (vertices, faces) on *device*.

    Returns:
        verts: (V, 3)  float32
        faces: (F, 3)  int64
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def sample_surface_points(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Uniformly sample *num_points* on the mesh surface.

    Uses area-weighted face sampling + barycentric interpolation.
    Returns:
        pts: (num_points, 3)
    """
    device = verts.device
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Triangle areas (proportional)
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = torch.norm(cross, dim=-1)  # 2× area, but ratio is fine
    probs = areas / areas.sum()

    # Sample faces
    face_idx = torch.multinomial(probs, num_points, replacement=True)

    # Random barycentric coords
    r1 = torch.sqrt(torch.rand(num_points, device=device))
    r2 = torch.rand(num_points, device=device)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2

    pts = (
        w0.unsqueeze(-1) * v0[face_idx]
        + w1.unsqueeze(-1) * v1[face_idx]
        + w2.unsqueeze(-1) * v2[face_idx]
    )
    return pts  # (num_points, 3)


# ==============================================================================
#                     HEAD AREA  (tools_adjusted.json)
# ==============================================================================

def load_tool_head_area(
    tools_json_path: str,
    tool_mesh_path: str,
) -> Optional[Tuple[list, list]]:
    """Look up the head_area for a tool by matching its filename stem.

    Returns:
        (head_lo, head_hi) as lists of 3 floats  (normalised bbox ratios),
        or None if not found.
    """
    if not tools_json_path or not Path(tools_json_path).exists():
        return None

    tool_stem = Path(tool_mesh_path).stem  # e.g. '006_claw_gripper_end_effector_var_001'
    with open(tools_json_path, "r") as f:
        tools = json.load(f)

    for entry in tools:
        if entry["name"] == tool_stem:
            ha = entry["head_area"]
            return ha[0], ha[1]  # lo, hi

    print(f"  ⚠ Tool '{tool_stem}' not found in {tools_json_path}, using uniform sampling.")
    return None


def compute_region_bounds(
    verts: torch.Tensor,
    head_area: Optional[Tuple[list, list]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert normalised head_area ratios to world-space bounds.

    Returns:
        (head_world_min, head_world_max) each (3,), or None.
    """
    if head_area is None:
        return None
    device = verts.device
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    bbox_range = bbox_max - bbox_min
    lo = torch.tensor(head_area[0], device=device, dtype=torch.float32)
    hi = torch.tensor(head_area[1], device=device, dtype=torch.float32)
    return bbox_min + lo * bbox_range, bbox_min + hi * bbox_range


def sample_region_surface_points(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_points: int,
    region_min: Optional[torch.Tensor] = None,
    region_max: Optional[torch.Tensor] = None,
    inside: bool = True,
) -> torch.Tensor:
    """Sample surface points from a specific bbox region (or its complement).

    Args:
        inside: If True, keep points INSIDE [region_min, region_max].
                If False, keep points OUTSIDE that box (i.e. handle/body).

    Returns:
        pts: (num_points, 3)
    """
    device = verts.device
    if region_min is None or region_max is None:
        return sample_surface_points(verts, faces, num_points)

    n_oversample = num_points * 15
    all_pts = sample_surface_points(verts, faces, n_oversample)

    in_box = (
        (all_pts >= region_min.unsqueeze(0)) &
        (all_pts <= region_max.unsqueeze(0))
    ).all(dim=-1)

    mask = in_box if inside else ~in_box
    region_pts = all_pts[mask]

    if region_pts.shape[0] == 0:
        print(f"  ⚠ No {'head' if inside else 'body'}-region points found, using uniform.")
        return sample_surface_points(verts, faces, num_points)

    idx = torch.randint(region_pts.shape[0], (num_points,), device=device)
    return region_pts[idx]


# ==============================================================================
#                OBJECT POSE  (random rotation + grounding)
# ==============================================================================

def randomise_object_pose(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a random SO(3) rotation and ground the object (z_min = 0).

    Returns:
        new_verts : (V, 3)  rotated + grounded vertices
        R_obj     : (3, 3)  rotation matrix applied
        z_shift   : scalar tensor  z-translation applied for grounding
    """
    device = verts.device
    R_obj = random_rotation_matrices(1, device).squeeze(0)  # (3, 3)
    rotated = verts @ R_obj.T  # (V, 3)

    # Ground: shift so z_min = 0
    z_shift = rotated[:, 2].min()
    rotated[:, 2] -= z_shift

    return rotated, R_obj, z_shift


# ==============================================================================
#                TOOL POSE INITIALISATION (with constraint projection)
# ==============================================================================

def _project_orientation(R: torch.Tensor, upright_threshold: float = 0.0) -> torch.Tensor:
    """Hard orientation constraint at initialisation.

    Flips any pose where the tool's +Z has a positive world-Z component
    (i.e. any upward-pointing orientation), consistent with the L_upright
    loss which penalises relu(R[2,2] - upright_threshold).

    The flip is a 180° rotation around the tool's local X axis:
        R' = R @ diag(1, -1, -1)
    which maps +Z → -Z and +Y → -Y, reflecting the tool downward.

    Args:
        R:                  (N, 3, 3)
        upright_threshold:  same value as cfg.upright_threshold (default 0.0)

    Returns:
        R_proj: (N, 3, 3) with tool +Z having no upward component.
    """
    z_col = R[:, :, 2]                        # (N, 3) — image of local +Z in world frame
    bad   = z_col[:, 2] > upright_threshold   # any upward component

    if bad.any():
        flip = torch.eye(3, device=R.device).unsqueeze(0).expand(bad.sum(), -1, -1).clone()
        flip[:, 1, 1] = -1.0
        flip[:, 2, 2] = -1.0
        R[bad] = R[bad] @ flip

    return R


def initialise_tool_poses(
    P_uniform: torch.Tensor,
    P_head: torch.Tensor,
    P_body: torch.Tensor,
    n_head_batch: int,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate N initial tool poses with intent-based anchor selection.

    The batch is split: first n_head_batch items target the Head region,
    the rest target the Body/Handle region.  For each item, a random
    anchor point is sampled from its assigned region, rotated, and then
    translated so that the rotated anchor lands on the object surface.

    Args:
        P_uniform:     (P_u, 3)  full-tool uniform cloud (for floor guard)
        P_head:        (P_h, 3)  head-region surface points
        P_body:        (P_b, 3)  body-region surface points
        n_head_batch:  int       how many items target the head
        obj_verts:     (V, 3)
        obj_faces:     (F, 3)
        cfg:           Config

    Returns:
        trans_init: (N, 3)
        rot6d_init: (N, 6)
    """
    device = cfg.device
    N = cfg.batch_size
    n_body_batch = N - n_head_batch
    OFFSET_MAX = 0.02

    # --- 1. Random rotations, project orientation ---
    R = random_rotation_matrices(N, device)
    R = _project_orientation(R, upright_threshold=cfg.upright_threshold)

    # --- 2. Sample one anchor per item from the assigned region ---
    head_anchor_idx = torch.randint(P_head.shape[0], (n_head_batch,), device=device)
    body_anchor_idx = torch.randint(P_body.shape[0], (n_body_batch,), device=device)
    anchors = torch.cat([P_head[head_anchor_idx], P_body[body_anchor_idx]], dim=0)  # (N, 3)

    # --- 3. Rotate each anchor: p_rot[i] = R[i] @ anchor[i] ---
    p_rot = torch.einsum("nij, nj -> ni", R, anchors)  # (N, 3)

    # --- 4. Sample target surface points + outward offset ---
    surf_pts = sample_surface_points(obj_verts, obj_faces, N)
    obj_centre = obj_verts.mean(dim=0)
    normals = F.normalize(surf_pts - obj_centre.unsqueeze(0), dim=-1)
    offset = torch.rand(N, 1, device=device) * OFFSET_MAX
    target = surf_pts + normals * offset

    # --- 5. Translation: place rotated anchor at target ---
    t = target - p_rot

    # --- 6. Floor guard (using uniform cloud) ---
    transformed = torch.einsum("pi, nji -> npj", P_uniform, R) + t.unsqueeze(1)
    z_mins = transformed[:, :, 2].min(dim=1).values
    lift = torch.clamp(-z_mins, min=0.0)
    t[:, 2] += lift

    rot6d_init = matrix_to_rot6d(R)
    return t, rot6d_init


# ==============================================================================
#                    DISTANCE / SIGN QUERIES  (Kaolin)
# ==============================================================================

def compute_unsigned_distance(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> torch.Tensor:
    """Batched unsigned point-to-mesh distance.

    Args:
        points:    (B, P, 3)
        obj_verts: (V, 3)
        obj_faces: (F, 3)

    Returns:
        dist: (B, P)  unsigned squared distances by default; we sqrt them.
    """
    B = points.shape[0]
    # Kaolin expects face_vertices: (B, F, 3, 3) or (1, F, 3, 3) broadcastable
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    )  # (1, F, 3, 3)
    face_verts = face_verts.expand(B, -1, -1, -1)

    # point_to_mesh_distance returns (distance, face_idx, dist_type)
    # distance is *squared* unsigned distance  (B, P)
    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )
    return torch.sqrt(sq_dist.clamp(min=1e-12))  # (B, P)


def compute_sign(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> torch.Tensor:
    """Check which points are *inside* the mesh (requires watertight mesh).

    Args:
        points:    (B, P, 3)
        obj_verts: (V, 3)
        obj_faces: (F, 3)

    Returns:
        inside: (B, P) bool  –  True if inside.
    """
    B = points.shape[0]
    verts_batch = obj_verts.unsqueeze(0).expand(B, -1, -1)
    sign = kaolin.ops.mesh.check_sign(verts_batch, obj_faces, points)
    return sign  # True = inside


# ==============================================================================
#                        LOSS FUNCTIONS
# ==============================================================================

def compute_losses(
    pts_world: torch.Tensor,         # (B, P, 3)
    obj_verts: torch.Tensor,         # (V, 3)
    obj_faces: torch.Tensor,         # (F, 3)
    R_batch: torch.Tensor,           # (B, 3, 3)
    cfg: Config,
) -> dict:
    """Compute the full loss landscape on a single uniform cloud.

    L_pen:     penalises points INSIDE the object.
    L_contact: attracts the K-closest OUTSIDE points toward the surface.
    L_floor:   penalises points below z=0.
    L_upright: penalises the tool's local +Z axis from pointing upward.
               Matches the hard orientation constraint in _project_orientation,
               but as a soft, differentiable penalty during optimisation.
    """
    B, P, _ = pts_world.shape

    dist = compute_unsigned_distance(pts_world, obj_verts, obj_faces)  # (B, P)
    inside = compute_sign(pts_world, obj_verts, obj_faces)             # (B, P)

    # ====== L_pen (Top-K worst penetrators) ======
    pen_dist = torch.where(inside, dist, torch.zeros_like(dist))
    K_pen = min(4, P)
    topk_pen, _ = torch.topk(pen_dist, K_pen, dim=1, largest=True)
    L_pen = topk_pen.mean()
    # Keep hard max for filtering (no gradient needed)
    L_pen_max_per_sample = pen_dist.max(dim=1).values

    # ====== L_contact (Inf-Masked Attraction) ======
    masked_dist = torch.where(~inside, dist, torch.full_like(dist, float("inf")))
    K = min(cfg.k_closest, P)
    topk_dist, _ = torch.topk(masked_dist, K, dim=1, largest=False)
    # Clamp surviving infs to 0 (when fewer than K points are outside)
    topk_dist = torch.where(topk_dist.isinf(), torch.zeros_like(topk_dist), topk_dist)
    L_contact_per_sample = topk_dist.mean(dim=1)
    L_contact = L_contact_per_sample.mean()

    # ====== L_floor ======
    z_vals = pts_world[:, :, 2]
    below = F.relu(-z_vals)
    L_floor_per_sample = below.mean(dim=1)
    L_floor = L_floor_per_sample.mean()

    # ====== L_upright — penalise any upward component of tool +Z ======
    # R_batch[:, 2, 2] = dot(tool +Z, world +Z).
    # With threshold=0: zero cost when pointing horizontal or downward,
    # linearly penalised as soon as the tool has any upward component.
    tool_z_world = R_batch[:, 2, 2]                              # (B,)
    L_upright = F.relu(tool_z_world - cfg.upright_threshold).mean()

    # ====== Total ======
    total = (
        cfg.w_pen     * L_pen
        + cfg.w_contact * L_contact
        + cfg.w_floor   * L_floor
        + cfg.w_upright * L_upright
    )

    return {
        "total":    total,
        "pen":      L_pen,
        "contact":  L_contact,
        "floor":    L_floor,
        "upright":  L_upright,
        "pen_max_ps":  L_pen_max_per_sample.detach(),
        "contact_ps":  L_contact_per_sample.detach(),
    }


# ==============================================================================
#                          OPTIMISATION LOOP
# ==============================================================================

def transform_points(
    pts: torch.Tensor,
    rot6d: torch.Tensor,
    trans: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rigid transform to tool point cloud.

    Args:
        pts:   (P, 3)   canonical tool points
        rot6d: (B, 6)   6-D rotation parameters
        trans: (B, 3)   translations

    Returns:
        pts_world: (B, P, 3)
        R:         (B, 3, 3)
    """
    R = rot6d_to_matrix(rot6d)  # (B, 3, 3)
    # pts_world = pts @ R^T + t
    pts_world = torch.einsum("pi, bji -> bpj", pts, R) + trans.unsqueeze(1)
    return pts_world, R


@torch.no_grad()
def _log_step(step: int, losses: dict) -> None:
    if step % 10 == 0 or step == 0:
        print(
            f"  step {step:4d}  |  total {losses['total'].item():.5f}  "
            f"pen {losses['pen'].item():.5f}  "
            f"contact {losses['contact'].item():.5f}  "
            f"floor {losses['floor'].item():.5f}  "
            f"upright {losses['upright'].item():.4f}"
        )


def optimise(
    P_uniform: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
    trans: torch.Tensor,
    rot6d: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Run Adam optimisation on the tool transforms.

    Uses a single uniform cloud for all losses (L_pen, L_contact, L_floor).
    """
    trans_opt = trans.clone().detach().requires_grad_(True)
    rot6d_opt = rot6d.clone().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([trans_opt, rot6d_opt], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.opt_steps, eta_min=cfg.lr * 0.01
    )

    for step in range(cfg.opt_steps):
        optimiser.zero_grad()
        R = rot6d_to_matrix(rot6d_opt)
        pts_world = torch.einsum("pi, bji -> bpj", P_uniform, R) + trans_opt.unsqueeze(1)
        losses = compute_losses(pts_world, obj_verts, obj_faces, R, cfg)
        losses["total"].backward()
        optimiser.step()
        scheduler.step()
        _log_step(step, losses)

    with torch.no_grad():
        R = rot6d_to_matrix(rot6d_opt)
        pts_world = torch.einsum("pi, bji -> bpj", P_uniform, R) + trans_opt.unsqueeze(1)
        final_losses = compute_losses(pts_world, obj_verts, obj_faces, R, cfg)

    return trans_opt.detach(), rot6d_opt.detach(), final_losses


# ==============================================================================
#                     CONTACT INFO EXTRACTION
# ==============================================================================

@torch.no_grad()
def compute_contact_info(
    P_uniform: torch.Tensor,   # (P, 3)  canonical tool cloud
    trans: torch.Tensor,       # (N, 3)  tool translations (valid subset)
    rot6d: torch.Tensor,       # (N, 6)  tool rotations   (valid subset)
    obj_verts: torch.Tensor,   # (V, 3)
    obj_faces: torch.Tensor,   # (F, 3)
    cfg: Config,
    n_contact: int = 5,
    sdf_threshold: float = 5e-3,
) -> dict:
    """Sample contact points from tool surface points with SDF < sdf_threshold.

    For each converged configuration:
      1. Transforms the full tool surface cloud to world frame.
      2. Computes unsigned SDF (distance to object surface) for all exterior points.
      3. Keeps only points where SDF < sdf_threshold  (the contact band).
      4. Randomly samples n_contact points from that band; if fewer than n_contact
         points pass the threshold, all passing points are used and the rest are
         filled by the closest exterior point (repeated) to keep a fixed shape.
      5. For each kept point, looks up the nearest object face and returns its
         outward unit normal.

    Args:
        n_contact:     number of contact points per config to keep (default 5).
        sdf_threshold: max distance to object surface to be considered in contact
                       (default 5e-3, same scale as contact_eps).

    Returns dict with keys:
        contact_pts_world  : (N, n_contact, 3)  – points in world frame
        contact_pts_tool_frame : (N, n_contact, 3)  – same points in tool frame
        contact_normals        : (N, n_contact, 3)  – outward face normals
        contact_sdfs           : (N, n_contact)     – SDF (distance) at each point
    """
    N = trans.shape[0]
    P = P_uniform.shape[0]
    device = cfg.device

    # ---- 1. Transform tool cloud to world frame ----
    R = rot6d_to_matrix(rot6d)                                              # (N, 3, 3)
    pts_world = torch.einsum("pi, bji -> bpj", P_uniform, R) + trans.unsqueeze(1)  # (N, P, 3)

    # ---- 2. SDF: unsigned distance, masked to exterior only ----
    dist   = compute_unsigned_distance(pts_world, obj_verts, obj_faces)  # (N, P)
    inside = compute_sign(pts_world, obj_verts, obj_faces)               # (N, P) bool
    # Inside points get inf so they are never selected
    sdf = torch.where(~inside, dist, torch.full_like(dist, float("inf")))  # (N, P)

    # ---- 3 & 4. Per-config: sample n_contact from sdf < threshold ----
    sel_pts_world = torch.zeros(N, n_contact, 3, device=device)
    sel_sdf       = torch.zeros(N, n_contact,    device=device)

    for b in range(N):
        sdf_b     = sdf[b]                         # (P,)
        in_band   = (sdf_b < sdf_threshold).nonzero(as_tuple=False).squeeze(1)  # (M,)

        if in_band.numel() == 0:
            # Fallback: use the single closest exterior point, repeated
            closest = sdf_b.argmin().unsqueeze(0).expand(n_contact)
            sel_pts_world[b] = pts_world[b][closest]
            sel_sdf[b]       = sdf_b[closest]
        elif in_band.numel() <= n_contact:
            # Fewer candidates than requested – use all, pad with closest
            chosen = in_band
            pad    = in_band[0:1].expand(n_contact - in_band.numel())
            chosen = torch.cat([chosen, pad], dim=0)
            sel_pts_world[b] = pts_world[b][chosen]
            sel_sdf[b]       = sdf_b[chosen]
        else:
            # Random subsample without replacement
            perm   = torch.randperm(in_band.numel(), device=device)[:n_contact]
            chosen = in_band[perm]
            sel_pts_world[b] = pts_world[b][chosen]
            sel_sdf[b]       = sdf_b[chosen]

    # ---- 5a. Map back to tool canonical frame: p_tool = R^T @ (p_world - t) ----
    p_centered       = sel_pts_world - trans.unsqueeze(1)          # (N, n_contact, 3)
    R_T              = R.permute(0, 2, 1)                          # (N, 3, 3)
    sel_pts_tool     = torch.einsum("bij, bkj -> bki", R_T, p_centered)  # (N, n_contact, 3)

    # ---- 5b. Nearest object face & outward face normal ----
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    ).expand(N, -1, -1, -1)   # (N, F, 3, 3)

    _sq_dist, face_idx, _dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        sel_pts_world.contiguous(), face_verts
    )  # face_idx: (N, n_contact)

    v0 = obj_verts[obj_faces[:, 0]]
    v1 = obj_verts[obj_faces[:, 1]]
    v2 = obj_verts[obj_faces[:, 2]]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # (F, 3)
    contact_normals = face_normals[face_idx]   # (N, n_contact, 3)

    return {
        "contact_pts_world":      sel_pts_world.cpu(),   # (N, n_contact, 3) world frame
        "contact_pts_tool_frame": sel_pts_tool.cpu(),    # (N, n_contact, 3)
        "contact_normals":        contact_normals.cpu(), # (N, n_contact, 3)
    }


# ==============================================================================
#          SIGNED SDF  —  tool canonical pts → object,  object canonical pts → tool
# ==============================================================================

@torch.no_grad()
def compute_tool_pts_sdf(
    P_tool: torch.Tensor,      # (P, 3)  canonical tool cloud
    trans: torch.Tensor,       # (N, 3)  tool translations
    rot6d: torch.Tensor,       # (N, 6)  tool rotations
    obj_verts: torch.Tensor,   # (V, 3)  grounded world frame
    obj_faces: torch.Tensor,   # (F, 3)
) -> torch.Tensor:             # (N, P)  signed SDF
    """Compute signed SDF from canonical tool points to the object, for each config.

    For each config n, transforms P_tool to world frame and queries the object mesh.
    SDF convention: positive = outside object, negative = inside object.

    Returns:
        sdf : (N, P)  signed distances
    """
    R = rot6d_to_matrix(rot6d)                                             # (N, 3, 3)
    pts_world = torch.einsum("pi, bji -> bpj", P_tool, R) + trans.unsqueeze(1)  # (N, P, 3)
    dist   = compute_unsigned_distance(pts_world, obj_verts, obj_faces)   # (N, P)
    inside = compute_sign(pts_world, obj_verts, obj_faces)                 # (N, P) bool
    sdf = torch.where(inside, -dist, dist)
    return sdf.cpu()                                                       # (N, P)


@torch.no_grad()
def compute_obj_pts_sdf(
    P_obj: torch.Tensor,       # (Q, 3)  canonical object pts (before R_obj)
    R_obj: torch.Tensor,       # (3, 3)
    z_shift: torch.Tensor,     # scalar  grounding z-offset
    tool_verts: torch.Tensor,  # (T, 3)  canonical tool frame
    tool_faces: torch.Tensor,  # (G, 3)
    trans: torch.Tensor,       # (N, 3)  tool translations
    rot6d: torch.Tensor,       # (N, 6)  tool rotations
) -> torch.Tensor:             # (N, Q)  signed SDF
    """Compute signed SDF from canonical object points to the tool, for each config.

    Applies R_obj + z_shift to get world-frame object points, then transforms them
    into each config's tool canonical frame and queries the canonical tool mesh.
    SDF convention: positive = outside tool, negative = inside tool.

    Returns:
        sdf : (N, Q)  signed distances
    """
    device = trans.device

    # Object canonical → world frame  (same for all configs)
    p_world = P_obj @ R_obj.T                          # (Q, 3)
    p_world = p_world.clone()
    p_world[:, 2] -= z_shift                           # apply grounding

    # World → tool canonical frame per config:  p_tool = R^T @ (p_world - t)
    R   = rot6d_to_matrix(rot6d)                       # (N, 3, 3)
    R_T = R.permute(0, 2, 1)                           # (N, 3, 3)
    p_centered     = p_world.unsqueeze(0) - trans.unsqueeze(1)          # (N, Q, 3)
    pts_tool_frame = torch.einsum("nij, nkj -> nki", R_T, p_centered)  # (N, Q, 3)

    N = trans.shape[0]
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        tool_verts.unsqueeze(0), tool_faces
    ).expand(N, -1, -1, -1)                                            # (N, G, 3, 3)

    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        pts_tool_frame.contiguous(), face_verts
    )
    dist = torch.sqrt(sq_dist.clamp(min=1e-12))                        # (N, Q)

    tool_verts_batch = tool_verts.unsqueeze(0).expand(N, -1, -1)
    inside = kaolin.ops.mesh.check_sign(
        tool_verts_batch, tool_faces, pts_tool_frame
    )                                                                  # (N, Q) bool
    sdf = torch.where(inside, -dist, dist)
    return sdf.cpu()                                                   # (N, Q)


# ==============================================================================
#                          FILTERING & SAVING
# ==============================================================================

def filter_and_save(
    trans: torch.Tensor,
    rot6d: torch.Tensor,
    R_obj: torch.Tensor,
    z_shift: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
    tool_verts: torch.Tensor,
    tool_faces: torch.Tensor,
    P_uniform: torch.Tensor,      # (P, 3)  canonical tool pts  (sampled once)
    P_obj_canonical: torch.Tensor,  # (Q, 3)  canonical object pts (sampled once)
    losses: dict,
    cfg: Config,
    obj_scale: float,
) -> int:
    """Keep only converged configurations and save to disk.

    Convergence criteria:
      • per-sample MAX penetration depth  < pen_max_eps
      • per-sample contact loss           < contact_eps

    Saved .pt layout
    ----------------
    Stored ONCE per file (shared across all configs):
      - tool_pts_canonical  (P, 3)  : canonical tool surface points
      - obj_pts_canonical   (Q, 3)  : canonical object surface points (before R_obj)
      - object_rotation     (3, 3)  : R_obj applied to object
      - obj_z_shift         scalar  : z grounding offset applied to object

    Stored PER CONFIG (N entries):
      - tool_translations   (N, 3)
      - tool_rotations      (N, 3, 3)
      - pen_loss            (N,)
      - contact_loss        (N,)
      - tool_pts_sdf        (N, P)  : signed SDF tool canonical pts → object
      - obj_pts_sdf         (N, Q)  : signed SDF object canonical pts → tool
      - contact_pts_world       (N, 5, 3)  : contact points in world frame
      - contact_pts_tool_frame  (N, 5, 3)
      - contact_normals         (N, 5, 3)

    SDF sign convention: positive = outside, negative = inside.

    Returns:
        n_saved: int
    """
    pen_ok = losses["pen_max_ps"] < cfg.pen_max_eps
    contact_ok = losses["contact_ps"] < cfg.contact_eps
    valid = pen_ok & contact_ok
    n_valid = valid.sum().item()

    print(f"\n✓ Converged: {n_valid} / {cfg.batch_size}")
    print(f"  Penetration pass (max < {cfg.pen_max_eps}): {pen_ok.sum().item()}")
    print(f"  Contact pass:     {contact_ok.sum().item()}")

    if n_valid == 0:
        print("⚠  No valid configurations found. Try increasing batch_size or opt_steps.")
        return 0

    # Reconstruct rotation matrices
    R_tool = rot6d_to_matrix(rot6d[valid])

    # ---- Contact info (5 pts + normals) ----
    print("  Computing contact info …")
    contact_info = compute_contact_info(
        P_uniform, trans[valid], rot6d[valid], obj_verts, obj_faces, cfg,
    )
    C = contact_info["contact_pts_world"].shape[1]
    print(f"  Contact info: {C} pts/config (with face normals)")

    # ---- Tool canonical pts → object SDF ----
    print(f"  Computing tool-side SDF ({P_uniform.shape[0]} pts × {n_valid} configs) …")
    tool_pts_sdf = compute_tool_pts_sdf(
        P_uniform, trans[valid], rot6d[valid], obj_verts, obj_faces,
    )  # (N, P)

    # ---- Object canonical pts → tool SDF ----
    print(f"  Computing object-side SDF ({P_obj_canonical.shape[0]} pts × {n_valid} configs) …")
    obj_pts_sdf = compute_obj_pts_sdf(
        P_obj_canonical, R_obj, z_shift, tool_verts, tool_faces,
        trans[valid], rot6d[valid],
    )  # (N, Q)
    print(f"  Done.")

    # ── Bake coordinate transforms so consumers need zero frame math ─────────

    # 1. Center tool canonical cloud at (0,0,0)
    tool_centroid = P_uniform.mean(dim=0)                          # (3,) mesh frame
    P_tool_c      = (P_uniform - tool_centroid).cpu()              # (P, 3) centered

    # 2. tool_translations → world-frame pos of tool CENTROID
    #    t_adj = R @ centroid + t_origin  (gives identical world pts)
    t_adj = torch.einsum("nij,j->ni", R_tool, tool_centroid) + trans[valid]  # (N, 3)

    # 3. World-frame object cloud → centered
    obj_world    = P_obj_canonical @ R_obj.T                       # (Q, 3)
    obj_world    = obj_world.clone()
    obj_world[:, 2] -= z_shift                                     # ground to z_min≈0
    obj_centroid = obj_world.mean(dim=0)                           # (3,) world centroid
    P_obj_c      = (obj_world - obj_centroid).cpu()                # (Q, 3) centered

    # 4. contact_pts_tool_frame → centered tool frame
    contact_pts_tool_c = (
        contact_info["contact_pts_tool_frame"]
        - tool_centroid.cpu().unsqueeze(0).unsqueeze(0)
    )

    result = {
        "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
        "tool_mesh_path":   str(Path(cfg.tool_mesh_path).resolve()),
        # ── Scales ──────────────────────────────────────────────────────────
        "tool_scale":   cfg.tool_scale,
        "object_scale": obj_scale,
        # ── Canonical clouds — load and use directly, no transforms needed ──
        #   tool_pts_canonical : (P,3) centered at (0,0,0), R=I
        #   obj_pts_canonical  : (Q,3) centered at (0,0,0), world frame
        #   obj_centroid       : (3,)  world-frame centroid, z>0
        "tool_pts_canonical": P_tool_c,
        "obj_pts_canonical":  P_obj_c,
        "obj_centroid":       obj_centroid.cpu(),
        # ── Per-config poses ─────────────────────────────────────────────────
        #   tool_translations : (N,3) world-frame position of tool CENTROID
        #   tool_rotations    : (N,3,3) unchanged
        "tool_translations":  t_adj.cpu(),
        "tool_rotations":     R_tool.cpu(),
        "pen_loss":    losses["pen_max_ps"][valid].cpu(),
        "contact_loss": losses["contact_ps"][valid].cpu(),
        # ── Per-config SDF ──────────────────────────────────────────────────
        "tool_pts_sdf": tool_pts_sdf,   # (N,P) signed: +outside object
        "obj_pts_sdf":  obj_pts_sdf,    # (N,Q) signed: +outside tool
        # ── Sparse contact geometry ──────────────────────────────────────────
        "contact_pts_world":      contact_info["contact_pts_world"],  # (N,C,3) world
        "contact_pts_tool_frame": contact_pts_tool_c,                 # (N,C,3) centered tool
        "contact_normals":        contact_info["contact_normals"],    # (N,C,3)
        # ── Private metadata — ONLY for gen_movement_delta mesh SDF ─────────
        #   Not used by dataset.py or any training code.
        "_object_rotation": R_obj.cpu(),
        "_obj_z_shift":     z_shift.cpu() if isinstance(z_shift, torch.Tensor)
                            else torch.tensor(float(z_shift)),
    }

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    torch.save(result, cfg.output_path)
    print(f"  Saved to {cfg.output_path}")
    return n_valid


# ==============================================================================
#                                MAIN
# ==============================================================================

def main(cfg: Config) -> None:
    device = cfg.device
    torch.manual_seed(cfg.seed)

    # ---- 1. Load meshes ----
    print(f"Loading object mesh: {cfg.object_mesh_path}")
    obj_verts, obj_faces = load_mesh(cfg.object_mesh_path, device)

    print(f"Loading tool mesh:   {cfg.tool_mesh_path}")
    tool_verts, tool_faces = load_mesh(cfg.tool_mesh_path, device)

    # ---- 1b. Apply scales (matches RL) ----
    # Tool: fixed scale
    print(f"  Applying tool scale: {cfg.tool_scale}")
    tool_verts = tool_verts * cfg.tool_scale

    # Object: random scale in range
    obj_scale = torch.empty(1, device=device).uniform_(
        cfg.object_scale_range[0], cfg.object_scale_range[1]
    ).item()
    print(f"  Applying object scale: {obj_scale:.4f} (range {cfg.object_scale_range})")
    obj_verts = obj_verts * obj_scale

    # ---- 2. Sample canonical object cloud (before pose randomisation) ----
    P_obj_canonical = sample_surface_points(
        obj_verts, obj_faces, cfg.num_tool_surface_pts
    )
    print(f"  P_obj_canonical: {P_obj_canonical.shape[0]} points (canonical object frame)")

    # ---- 3. Randomise object pose & ground ----
    print("Randomising object pose & grounding …")
    obj_verts, R_obj, z_shift = randomise_object_pose(obj_verts, obj_faces)

    # ---- 4. Sample canonical tool cloud (used for ALL losses + SDF output) ----
    P_uniform = sample_surface_points(
        tool_verts, tool_faces, cfg.num_tool_surface_pts
    )
    print(f"  P_uniform (tool canonical): {P_uniform.shape[0]} points")

    # ---- 5. Compute head/body subsets for init anchor selection ----
    head_area = load_tool_head_area(cfg.tools_json_path, cfg.tool_mesh_path)
    bounds = compute_region_bounds(tool_verts, head_area)
    if bounds is not None:
        head_min, head_max = bounds
        print(f"  Head bounds: {head_min.tolist()} → {head_max.tolist()}")
        # Extract head/body subsets from the uniform cloud for anchor picking
        in_head = (
            (P_uniform >= head_min.unsqueeze(0)) &
            (P_uniform <= head_max.unsqueeze(0))
        ).all(dim=-1)
        P_head = P_uniform[in_head]
        P_body = P_uniform[~in_head]
        # Fallbacks if a region is empty
        if P_head.shape[0] == 0:
            print("  ⚠ No head points in uniform cloud, using all as head.")
            P_head = P_uniform
        if P_body.shape[0] == 0:
            print("  ⚠ No body points in uniform cloud, using all as body.")
            P_body = P_uniform
    else:
        P_head = P_uniform
        P_body = P_uniform
    print(f"  Init anchors: {P_head.shape[0]} head / {P_body.shape[0]} body")

    # ---- 6. Compute intent split ----
    n_head_batch = int(cfg.batch_size * cfg.contact_mode_prob)
    n_body_batch = cfg.batch_size - n_head_batch
    print(f"  Batch split: {n_head_batch} head / {n_body_batch} body")

    # ---- 7. Initialise tool poses (anchor → surface) ----
    print(f"Initialising {cfg.batch_size} tool poses …")
    trans_init, rot6d_init = initialise_tool_poses(
        P_uniform, P_head, P_body, n_head_batch,
        obj_verts, obj_faces, cfg,
    )

    # ---- 7b. Optionally save initial poses for debugging ----
    if cfg.save_init_path:
        R_init = rot6d_to_matrix(rot6d_init)
        init_result = {
            "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
            "tool_mesh_path": str(Path(cfg.tool_mesh_path).resolve()),
            "object_rotation": R_obj.cpu(),
            "object_vertices_grounded": obj_verts.cpu(),
            "tool_translations": trans_init.detach().cpu(),
            "tool_rotations": R_init.detach().cpu(),
        }
        torch.save(init_result, cfg.save_init_path)
        print(f"  ✓ Initial poses saved to {cfg.save_init_path}")

    # ---- 8. Optimise ----
    print(f"Running {cfg.opt_steps} Adam steps …")
    trans_opt, rot6d_opt, final_losses = optimise(
        P_uniform, obj_verts, obj_faces, trans_init, rot6d_init, cfg,
    )

    # ---- 9. Filter & save ----
    n_saved = filter_and_save(
        trans_opt, rot6d_opt, R_obj, z_shift,
        obj_verts, obj_faces, tool_verts, tool_faces,
        P_uniform, P_obj_canonical, final_losses, cfg, obj_scale,
    )
    print(f"\nDone.  {n_saved} valid contact configurations saved.")


# ==============================================================================
#                           CLI ENTRY POINT
# ==============================================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Generate collision-free tool–object contact configurations."
    )
    # ── I/O ──────────────────────────────────────────────────────────────────────
    p.add_argument("--object",     required=True, help="Path to watertight object .obj")
    p.add_argument("--tool",       required=True, help="Path to tool .obj")
    p.add_argument("--output",     default="contact_configs.pt", help="Output .pt file")
    p.add_argument("--save-init",  default="",
                   help="Save initial poses (pre-opt) here for debugging")
    p.add_argument("--tools-json", default="",
                   help="Path to tools_adjusted.json for head_area lookup")
    # ── Batch & runtime ──────────────────────────────────────────────────────────
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--device",     type=str,   default="cuda:0")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed (change for different object poses)")
    # ── All loss weights / thresholds live in contact_config.py ────────────────

    args = p.parse_args()
    return Config(
        object_mesh_path=args.object,
        tool_mesh_path=args.tool,
        output_path=args.output,
        save_init_path=args.save_init,
        tools_json_path=args.tools_json,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

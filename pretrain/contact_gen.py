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

# ==============================================================================
#                          CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """All tuneable knobs in one place."""

    # ----- I/O -----
    object_mesh_path: str = "object.obj"
    tool_mesh_path: str = "tool.obj"
    output_path: str = "contact_configs.pt"
    save_init_path: str = ""        # if set, dump initial poses (pre-optimisation)
    tools_json_path: str = ""       # path to tools_adjusted.json (for head_area)

    # ----- Batch & sampling -----
    batch_size: int = 512           # number of random tool poses
    num_tool_surface_pts: int = 2048  # uniform surface cloud (all losses)
    contact_mode_prob: float = 0.7   # prob of targeting head (vs handle/body)
    device: str = "cuda:0"

    # ----- Optimisation -----
    opt_steps: int = 80
    lr: float = 5e-3
    # Loss weights
    w_pen: float = 800.0            # penetration penalty
    w_contact: float = 1.0          # attraction loss
    w_floor: float = 20.0           # below-floor penalty
    k_closest: int = 24             # how many closest points for attraction

    # ----- Convergence thresholds -----
    pen_eps: float = 1e-4           # max allowed penetration loss
    contact_eps: float = 3e-3        # max avg distance for "in contact"


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a random SO(3) rotation and ground the object (z_min = 0).

    Returns:
        new_verts: (V, 3)  on same device
        R_obj: (3, 3)      the rotation applied
    """
    device = verts.device
    R_obj = random_rotation_matrices(1, device).squeeze(0)  # (3, 3)
    rotated = verts @ R_obj.T  # (V, 3)

    # Ground: shift so z_min = 0
    z_min = rotated[:, 2].min()
    rotated[:, 2] -= z_min

    return rotated, R_obj


# ==============================================================================
#                TOOL POSE INITIALISATION (with constraint projection)
# ==============================================================================

def _project_orientation(R: torch.Tensor) -> torch.Tensor:
    """Ensure the tool's local +Z maps to a direction with global z <= 0.

    Strategy: if the rotated +Z has positive global-z, flip it by
    composing with a 180° rotation about the local X axis.

    Args:
        R: (N, 3, 3)

    Returns:
        R_proj: (N, 3, 3) with the constraint enforced.
    """
    z_col = R[:, :, 2]       # (N, 3) – image of local +Z
    bad = z_col[:, 2] > 0    # violating samples

    if bad.any():
        # Rotation by π about X:  diag(1, -1, -1)
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
    R = _project_orientation(R)

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
               Inside points get 0 contact loss (handled by L_pen).
    L_floor:   penalises points below z=0.
    """
    B, P, _ = pts_world.shape

    dist = compute_unsigned_distance(pts_world, obj_verts, obj_faces)  # (B, P)
    inside = compute_sign(pts_world, obj_verts, obj_faces)             # (B, P)

    # ====== L_pen ======
    pen_dist = torch.where(inside, dist, torch.zeros_like(dist))
    L_pen_per_sample = pen_dist.mean(dim=1)
    L_pen = L_pen_per_sample.mean()

    # ====== L_contact (outside-only) ======
    outside_dist = torch.where(~inside, dist, torch.zeros_like(dist))
    K = min(cfg.k_closest, P)
    topk_dist, _ = torch.topk(outside_dist, K, dim=1, largest=False)
    L_contact_per_sample = topk_dist.mean(dim=1)
    L_contact = L_contact_per_sample.mean()

    # ====== L_floor ======
    z_vals = pts_world[:, :, 2]
    below = F.relu(-z_vals)
    L_floor_per_sample = below.mean(dim=1)
    L_floor = L_floor_per_sample.mean()

    # ====== Total ======
    total = (
        cfg.w_pen * L_pen
        + cfg.w_contact * L_contact
        + cfg.w_floor * L_floor
    )

    return {
        "total": total,
        "pen": L_pen,
        "contact": L_contact,
        "floor": L_floor,
        "pen_ps": L_pen_per_sample.detach(),
        "contact_ps": L_contact_per_sample.detach(),
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

    for step in range(cfg.opt_steps):
        optimiser.zero_grad()
        R = rot6d_to_matrix(rot6d_opt)
        pts_world = torch.einsum("pi, bji -> bpj", P_uniform, R) + trans_opt.unsqueeze(1)
        losses = compute_losses(pts_world, obj_verts, obj_faces, R, cfg)
        losses["total"].backward()
        optimiser.step()
        _log_step(step, losses)

    with torch.no_grad():
        R = rot6d_to_matrix(rot6d_opt)
        pts_world = torch.einsum("pi, bji -> bpj", P_uniform, R) + trans_opt.unsqueeze(1)
        final_losses = compute_losses(pts_world, obj_verts, obj_faces, R, cfg)

    return trans_opt.detach(), rot6d_opt.detach(), final_losses


# ==============================================================================
#                          FILTERING & SAVING
# ==============================================================================

def filter_and_save(
    trans: torch.Tensor,
    rot6d: torch.Tensor,
    R_obj: torch.Tensor,
    obj_verts: torch.Tensor,
    losses: dict,
    cfg: Config,
) -> int:
    """Keep only converged configurations and save to disk.

    Convergence criteria:
      • per-sample penetration loss  < pen_eps
      • per-sample contact loss      < contact_eps

    Returns:
        n_saved: int
    """
    pen_ok = losses["pen_ps"] < cfg.pen_eps
    contact_ok = losses["contact_ps"] < cfg.contact_eps
    valid = pen_ok & contact_ok
    n_valid = valid.sum().item()

    print(f"\n✓ Converged: {n_valid} / {cfg.batch_size}")
    print(f"  Penetration pass: {pen_ok.sum().item()}")
    print(f"  Contact pass:     {contact_ok.sum().item()}")

    if n_valid == 0:
        print("⚠  No valid configurations found. Try increasing batch_size or opt_steps.")
        return 0

    # Reconstruct rotation matrices
    R_tool = rot6d_to_matrix(rot6d[valid])

    result = {
        "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
        "tool_mesh_path": str(Path(cfg.tool_mesh_path).resolve()),
        "object_rotation": R_obj.cpu(),
        "object_vertices_grounded": obj_verts.cpu(),
        "tool_translations": trans[valid].cpu(),
        "tool_rotations": R_tool.cpu(),
        "pen_loss": losses["pen_ps"][valid].cpu(),
        "contact_loss": losses["contact_ps"][valid].cpu(),
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
    torch.manual_seed(42)

    # ---- 1. Load meshes ----
    print(f"Loading object mesh: {cfg.object_mesh_path}")
    obj_verts, obj_faces = load_mesh(cfg.object_mesh_path, device)

    print(f"Loading tool mesh:   {cfg.tool_mesh_path}")
    tool_verts, tool_faces = load_mesh(cfg.tool_mesh_path, device)

    # ---- 2. Randomise object pose & ground ----
    print("Randomising object pose & grounding …")
    obj_verts, R_obj = randomise_object_pose(obj_verts, obj_faces)

    # ---- 3. Sample uniform cloud (used for ALL losses) ----
    P_uniform = sample_surface_points(
        tool_verts, tool_faces, cfg.num_tool_surface_pts
    )
    print(f"  P_uniform: {P_uniform.shape[0]} points")

    # ---- 4. Compute head/body subsets for init anchor selection ----
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

    # ---- 5. Compute intent split ----
    n_head_batch = int(cfg.batch_size * cfg.contact_mode_prob)
    n_body_batch = cfg.batch_size - n_head_batch
    print(f"  Batch split: {n_head_batch} head / {n_body_batch} body")

    # ---- 6. Initialise tool poses (anchor → surface) ----
    print(f"Initialising {cfg.batch_size} tool poses …")
    trans_init, rot6d_init = initialise_tool_poses(
        P_uniform, P_head, P_body, n_head_batch,
        obj_verts, obj_faces, cfg,
    )

    # ---- 6b. Optionally save initial poses for debugging ----
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

    # ---- 7. Optimise ----
    print(f"Running {cfg.opt_steps} Adam steps …")
    trans_opt, rot6d_opt, final_losses = optimise(
        P_uniform, obj_verts, obj_faces, trans_init, rot6d_init, cfg,
    )

    # ---- 8. Filter & save ----
    n_saved = filter_and_save(
        trans_opt, rot6d_opt, R_obj, obj_verts, final_losses, cfg,
    )
    print(f"\nDone.  {n_saved} valid contact configurations saved.")


# ==============================================================================
#                           CLI ENTRY POINT
# ==============================================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Generate collision-free tool–object contact configurations."
    )
    p.add_argument("--object", type=str, required=True, help="Path to watertight object .obj")
    p.add_argument("--tool", type=str, required=True, help="Path to tool .obj")
    p.add_argument("--output", type=str, default="contact_configs.pt", help="Output .pt file")
    p.add_argument("--save-init", type=str, default="",
                   help="If set, save initial poses (pre-opt) to this .pt for debugging")
    p.add_argument("--tools-json", type=str, default="",
                   help="Path to tools_adjusted.json for head_area lookup")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-pts", type=int, default=1024,
                   help="Uniform surface cloud points (all losses)")
    p.add_argument("--contact-mode-prob", type=float, default=0.7,
                   help="Fraction of batch targeting head vs body for init")
    p.add_argument("--opt-steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    # Loss weights
    p.add_argument("--w-pen", type=float, default=50.0)
    p.add_argument("--w-contact", type=float, default=1.0)
    p.add_argument("--w-floor", type=float, default=20.0)
    p.add_argument("--k-closest", type=int, default=32)
    # Thresholds
    p.add_argument("--pen-eps", type=float, default=1e-3)
    p.add_argument("--contact-eps", type=float, default=5e-3)

    args = p.parse_args()
    return Config(
        object_mesh_path=args.object,
        tool_mesh_path=args.tool,
        output_path=args.output,
        save_init_path=args.save_init,
        tools_json_path=args.tools_json,
        batch_size=args.batch_size,
        num_tool_surface_pts=args.num_pts,
        contact_mode_prob=args.contact_mode_prob,
        device=args.device,
        opt_steps=args.opt_steps,
        lr=args.lr,
        w_pen=args.w_pen,
        w_contact=args.w_contact,
        w_floor=args.w_floor,
        k_closest=args.k_closest,
        pen_eps=args.pen_eps,
        contact_eps=args.contact_eps,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

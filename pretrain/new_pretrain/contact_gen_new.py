#!/usr/bin/env python3
"""
contact_gen_new.py — Ultra-fast contact configuration generator.

Drop-in replacement for contact_gen.py.  Replaces the Kaolin gradient-descent
optimisation core with a single-pass GPU rejection sampler:

  1. Build a 128³ voxel SDF of the (scaled, grounded) object mesh using trimesh.
  2. For each of B=4096 (tool_pt, obj_pt) contact pairs, test M=1024 candidate
     rotation matrices in parallel by querying the SDF grid with F.grid_sample.
  3. Keep poses where every tool point is outside the object (SDF > -epsilon).
  4. Apply upright and floor constraints as pre/post filters (no gradient needed).
  5. Compute final per-config SDF outputs using Kaolin (only on N_valid survivors).

Output .pt format is identical to contact_gen.py EXCEPT contact_pts_* fields are
omitted (not needed by current training pipeline).

Interface:
    from new_pretrain.contact_gen_new import Config, main as optimize_gen_main
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Kaolin: used only for final per-config SDF outputs (GPU-accelerated mesh queries)
import kaolin
import kaolin.ops.mesh
import kaolin.metrics.trianglemesh

# Contact config lives in the same directory
from contact_config import CONTACT_GEN


# ==============================================================================
#                          CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Per-run configuration. Drop-in identical interface to contact_gen.Config."""

    # ── I/O ────────────────────────────────────────────────────────────────
    object_mesh_path: str = "object.obj"
    tool_mesh_path:   str = "tool.obj"
    output_path:      str = "contact_configs.pt"
    tools_json_path:  str = ""   # unused by new sampler, kept for API compat

    # ── Device / seed ────────────────────────────────────────────────────
    device: str = "cuda:0"
    seed:   int = 42

    # ── Rejection sampler hyperparameters (from CONTACT_GEN) ──────────────
    B:            int   = CONTACT_GEN.B              # 4096 contact pairs per call
    M:            int   = CONTACT_GEN.M              # 1024 candidate rotations per pair
    K:            int   = CONTACT_GEN.num_surface_pts  # 512 tool surface pts
    sdf_grid_res: int   = CONTACT_GEN.sdf_grid_res   # 128
    chunk_B:      int   = CONTACT_GEN.chunk_B        # 512

    # ── Scales (must match RL / contact_gen.py exactly) ───────────────────
    tool_scale:           float = CONTACT_GEN.tool_scale
    object_scale_range:   tuple = CONTACT_GEN.object_scale_range
    num_tool_surface_pts: int   = CONTACT_GEN.num_surface_pts

    # ── Geometric constraints ─────────────────────────────────────────────
    upright_threshold: float = CONTACT_GEN.upright_threshold
    epsilon:           float = CONTACT_GEN.epsilon
    floor_eps:         float = CONTACT_GEN.floor_eps

    # ── Head-area bias (tool contact point sampling) ───────────────────────
    contact_mode_prob: float = CONTACT_GEN.contact_mode_prob   # 0.7 → 70% head

    # ── Kept from CONTACT_GEN for compat (not used by sampler) ────────────
    pen_max_eps: float = CONTACT_GEN.pen_max_eps
    contact_eps: float = CONTACT_GEN.contact_eps


# ==============================================================================
#                     ROTATION UTILITIES  (6-D continuous)
# ==============================================================================

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """(*, 6) → (*, 3, 3)  via Gram-Schmidt."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # columns = basis vectors


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(*, 3, 3) → (*, 6)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def random_rotation_matrices(n: int, device: str) -> torch.Tensor:
    """Uniform SO(3) sample via QR.  Returns (n, 3, 3)."""
    H = torch.randn(n, 3, 3, device=device)
    Q, R_ = torch.linalg.qr(H)
    signs = torch.sign(torch.diagonal(R_, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0] *= -1
    return Q


# ==============================================================================
#                        MESH LOADING HELPERS
# ==============================================================================

def load_mesh(path: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load .obj → (verts (V,3) float32, faces (F,3) int64) on device."""
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces,    dtype=torch.int64,   device=device)
    return verts, faces


def sample_surface_points(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Area-weighted surface sampling.  Returns (num_points, 3)."""
    device = verts.device
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    probs = areas / areas.sum()
    face_idx = torch.multinomial(probs, num_points, replacement=True)
    r1 = torch.sqrt(torch.rand(num_points, device=device))
    r2 = torch.rand(num_points, device=device)
    pts = ((1 - r1).unsqueeze(-1) * v0[face_idx]
           + (r1 * (1 - r2)).unsqueeze(-1) * v1[face_idx]
           + (r1 * r2).unsqueeze(-1) * v2[face_idx])
    return pts


# ==============================================================================
#          HEAD-AREA BIAS  (tool contact point split, ported from contact_gen.py)
# ==============================================================================

def load_tool_head_area(
    tools_json_path: str,
    tool_mesh_path:  str,
) -> Optional[Tuple[list, list]]:
    """Return (head_lo, head_hi) normalised bbox ratios from tools_adjusted.json,
    or None if not found / json not provided."""
    if not tools_json_path or not Path(tools_json_path).exists():
        return None
    tool_stem = Path(tool_mesh_path).stem
    with open(tools_json_path) as f:
        tools = json.load(f)
    for entry in tools:
        if entry.get("name") == tool_stem:
            ha = entry["head_area"]
            return ha[0], ha[1]
    print(f"  ⚠ Tool '{tool_stem}' not in {tools_json_path}; using uniform sampling.")
    return None


def compute_head_bounds(
    verts: torch.Tensor,
    head_area: Optional[Tuple[list, list]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert normalised head_area ratios → world-space (head_min, head_max)."""
    if head_area is None:
        return None
    device = verts.device
    bbox_min = verts.min(dim=0).values
    bbox_range = verts.max(dim=0).values - bbox_min
    lo = torch.tensor(head_area[0], device=device, dtype=torch.float32)
    hi = torch.tensor(head_area[1], device=device, dtype=torch.float32)
    return bbox_min + lo * bbox_range, bbox_min + hi * bbox_range


def split_head_body(
    P: torch.Tensor,                               # (K, 3) tool surface pts
    bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:            # (P_head, 3), (P_body, 3)
    """Split a tool surface cloud into head and body subsets.
    If bounds is None both tensors are the full cloud.
    Falls back to full cloud if either region is empty.
    """
    if bounds is None:
        return P, P
    head_min, head_max = bounds
    in_head = ((P >= head_min.unsqueeze(0)) & (P <= head_max.unsqueeze(0))).all(dim=-1)
    P_head = P[in_head]
    P_body = P[~in_head]
    if P_head.shape[0] == 0:
        print("  ⚠ No head-region pts; using full cloud as head.")
        P_head = P
    if P_body.shape[0] == 0:
        print("  ⚠ No body-region pts; using full cloud as body.")
        P_body = P
    return P_head, P_body


# ==============================================================================
#                OBJECT POSE  (random rotation + grounding)
# ==============================================================================

def randomise_object_pose(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random SO(3) rotation + ground (z_min = 0).
    Returns (new_verts (V,3), R_obj (3,3), z_shift scalar)."""
    device = verts.device
    R_obj   = random_rotation_matrices(1, device).squeeze(0)
    rotated = verts @ R_obj.T
    z_shift = rotated[:, 2].min()
    rotated[:, 2] -= z_shift
    return rotated, R_obj, z_shift


# ==============================================================================
#              SDF GRID CONSTRUCTION  (GPU — Kaolin batched distance)
# ==============================================================================

@torch.no_grad()
def build_sdf_grid(
    obj_verts: torch.Tensor,   # (V, 3) scaled + grounded, world frame, on device
    obj_faces: torch.Tensor,   # (F, 3) int64, on device
    grid_res:  int   = 128,
    padding:   float = 0.05,
    device:    str   = "cuda:0",
    chunk:     int   = 65536,  # grid pts per Kaolin call (64k → ~1.5 GB VRAM)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a grid_res³ voxel SDF entirely on GPU using Kaolin.

    Replaces the CPU trimesh BVH approach (~30-60 s) with batched GPU queries
    (~1-3 s on L40).  Same sign convention: positive = outside, negative = inside.

    Returns:
        sdf_grid : (1, 1, R, R, R) float32 on device  — axes: [D=z, H=y, W=x]
        bbox_min : (3,) float32
        bbox_max : (3,) float32
    """
    R = grid_res
    bbox_min = obj_verts.min(dim=0).values - padding   # (3,) on device
    bbox_max = obj_verts.max(dim=0).values + padding   # (3,) on device

    # Build R³ query grid on GPU
    xs = torch.linspace(bbox_min[0].item(), bbox_max[0].item(), R, device=device)
    ys = torch.linspace(bbox_min[1].item(), bbox_max[1].item(), R, device=device)
    zs = torch.linspace(bbox_min[2].item(), bbox_max[2].item(), R, device=device)
    xg, yg, zg = torch.meshgrid(xs, ys, zs, indexing="ij")   # (R,R,R) each
    pts = torch.stack([xg, yg, zg], dim=-1).reshape(-1, 3)    # (R³, 3)
    N_pts = pts.shape[0]

    # Pre-expand face_verts once — reused with batch=1 across chunks
    face_verts_1 = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces   # (1, V, 3) → (1, F, 3, 3)
    )

    dist_vals   = torch.empty(N_pts, dtype=torch.float32, device=device)
    inside_vals = torch.empty(N_pts, dtype=torch.bool,    device=device)

    for s in range(0, N_pts, chunk):
        e      = min(s + chunk, N_pts)
        q_pts  = pts[s:e].unsqueeze(0)                          # (1, n, 3)

        # Unsigned distance
        sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            q_pts.contiguous(), face_verts_1
        )                                                        # (1, n)
        dist_vals[s:e] = sq_dist.squeeze(0).clamp(min=0).sqrt()

        # Sign: True = inside mesh
        inside_vals[s:e] = kaolin.ops.mesh.check_sign(
            obj_verts.unsqueeze(0), obj_faces, q_pts
        ).squeeze(0)                                             # (n,) bool

    # Signed SDF: negative inside, positive outside
    sdf_vals = torch.where(inside_vals, -dist_vals, dist_vals)  # (R³,)

    # Reshape: xyz order → permute to [z,y,x] = [D,H,W] for grid_sample
    sdf_dhw  = sdf_vals.reshape(R, R, R).permute(2, 1, 0)       # (D, H, W)
    sdf_grid = sdf_dhw.unsqueeze(0).unsqueeze(0)                 # (1, 1, D, H, W)

    return sdf_grid, bbox_min, bbox_max


def _query_sdf_grid(
    pts_world: torch.Tensor,     # (N_pts, 3)  world-frame points
    sdf_grid:  torch.Tensor,     # (1, 1, D, H, W)
    bbox_min:  torch.Tensor,     # (3,)
    bbox_max:  torch.Tensor,     # (3,)
) -> torch.Tensor:               # (N_pts,)
    """Trilinear interpolation into the SDF grid for arbitrary world pts.

    Coordinates outside the grid are clamped (border padding mode).
    """
    # Normalise to [-1, 1]:  x→W, y→H, z→D  (matches [D,H,W] storage)
    span   = bbox_max - bbox_min                             # (3,)
    norm   = 2.0 * (pts_world - bbox_min) / span - 1.0      # (N,3)  [-1,1]
    # grid_sample grid shape: [N_batch, D_out, H_out, W_out, 3] coords (x,y,z)
    grid   = norm.view(1, 1, 1, -1, 3)                      # (1,1,1,N,3)
    out    = F.grid_sample(
        sdf_grid,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )                                                        # (1,1,1,1,N)
    return out.view(-1)                                      # (N,)


# ==============================================================================
#              BATCHED SDF REJECTION SAMPLER  (core algorithm)
# ==============================================================================

@torch.no_grad()
def rejection_sample(
    P_tool:   torch.Tensor,     # (K, 3)  canonical tool surface pts (centroid-shifted)
    obj_surf: torch.Tensor,     # (S, 3)  object surface pts (world frame, grounded)
    sdf_grid: torch.Tensor,     # (1,1,D,H,W)
    bbox_min: torch.Tensor,     # (3,)
    bbox_max: torch.Tensor,     # (3,)
    cfg:      Config,
    P_head:   Optional[torch.Tensor] = None,  # (P_h, 3)  head-region pts (or None→uniform)
    P_body:   Optional[torch.Tensor] = None,  # (P_b, 3)  body-region pts (or None→uniform)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched rejection sampling: returns valid (R, t) pairs.

    Algorithm (all operations loop-free over B and M):

      For each chunk of B pairs:
        Step 1: sample tool contact pts p_B (B,3) and obj contact pts p_A (B,3).
        Step 2: shift tool cloud so p_B is at origin → (B, K, 3).
        Step 3: generate M upright-filtered rotations R_cands (M, 3, 3).
        Step 4: apply R_cands → (B, M, K, 3), then translate to p_A.
        Step 5: normalise coords and query SDF grid via grid_sample → (B, M, K).
        Step 6: floor filter  (tool z_min ≥ -floor_eps).
        Step 7: penetration filter  (min SDF per pose > -epsilon).
        Step 8: per-pair pick the rotation with maximum min-SDF clearance.

    Returns:
        R_valid : (N_valid, 3, 3)
        t_valid : (N_valid, 3)   world-frame tool MESH-ORIGIN translation
                                  (contact point placement, not centroid-adjusted yet)
    """
    device   = cfg.device
    B        = cfg.B
    M        = cfg.M
    K        = P_tool.shape[0]
    chunk_B  = cfg.chunk_B
    epsilon  = cfg.epsilon
    floor_eps = cfg.floor_eps

    # Pre-generate M candidate rotations filtered by upright constraint
    # We over-sample and keep only those with R[2,2] <= upright_threshold
    def _sample_upright_rotations(n: int) -> torch.Tensor:
        """Return n rotation matrices with tool +Z ≤ upright_threshold (pointing down)."""
        collected = []
        needed    = n
        while needed > 0:
            cands = random_rotation_matrices(needed * 4, device)         # over-sample
            good  = cands[:, 2, 2] <= cfg.upright_threshold
            cands = cands[good]
            if cands.shape[0] > 0:
                take = min(cands.shape[0], needed)
                collected.append(cands[:take])
                needed -= take
        return torch.cat(collected, dim=0)[:n]   # (n, 3, 3)

    R_cands = _sample_upright_rotations(M)   # (M, 3, 3)

    # ── Pre-compute head/body flag ────────────────────────────────────────────
    use_head = (P_head is not None) and (P_body is not None)

    R_list, t_list = [], []
    n_total_pairs  = 0

    for b_start in range(0, B, chunk_B):
        b_end  = min(b_start + chunk_B, B)
        cb     = b_end - b_start           # actual chunk size

        # Step 1: sample contact pairs with head-area bias
        if use_head:
            n_head = int(cb * cfg.contact_mode_prob)
            n_body = cb - n_head
            idx_h = torch.randint(P_head.shape[0], (n_head,), device=device)
            idx_b = torch.randint(P_body.shape[0], (n_body,), device=device)
            idx_tool_pts = torch.cat([P_head[idx_h], P_body[idx_b]], dim=0)  # (cb, 3)
        else:
            idx_tool_pts = P_tool[torch.randint(K, (cb,), device=device)]    # (cb, 3)

        idx_obj  = torch.randint(obj_surf.shape[0], (cb,), device=device)
        p_B = idx_tool_pts      # (cb, 3)  tool contact pt (canonical frame)
        p_A = obj_surf[idx_obj] # (cb, 3)  obj contact pt  (world frame)

        # Step 2: shift tool cloud so p_B → origin  →  (cb, K, 3)
        tool_shifted = P_tool.unsqueeze(0) - p_B.unsqueeze(1)   # (cb, K, 3)

        # Step 4: rotate (cb, K, 3) by M rotations → (cb, M, K, 3)
        # einsum: 'mij, bkj -> bmki'
        pts_rot = torch.einsum("mij, bkj -> bmki", R_cands, tool_shifted)  # (cb,M,K,3)

        # Step 3 (placed after rotation): translate rotated clouds to p_A
        pts_world = pts_rot + p_A[:, None, None, :]              # (cb, M, K, 3)

        # Step 6: floor filter — z_min ≥ -floor_eps
        z_min_per_pose = pts_world[..., 2].min(dim=-1).values   # (cb, M)
        floor_ok = z_min_per_pose >= -floor_eps

        # Step 5: query SDF grid for all (cb*M*K) points
        pts_flat = pts_world.reshape(-1, 3)                      # (cb*M*K, 3)
        sdf_flat = _query_sdf_grid(pts_flat, sdf_grid, bbox_min, bbox_max)
        sdf_bmk  = sdf_flat.reshape(cb, M, K)                   # (cb, M, K)

        # Step 7: penetration filter — min SDF (deepest pt) > -epsilon
        min_sdf  = sdf_bmk.min(dim=-1).values                   # (cb, M)
        pen_ok   = min_sdf > -epsilon

        # Combined validity mask
        valid_mask = floor_ok & pen_ok                           # (cb, M)

        # Step 8: per-pair select rotation with MAXIMUM min-SDF clearance
        # Mask invalid entries with large negative sentinel
        score = min_sdf.clone()
        score[~valid_mask] = -1e9
        best_m = score.argmax(dim=-1)                           # (cb,)
        pair_valid = valid_mask[torch.arange(cb, device=device), best_m]  # (cb,)

        n_total_pairs += cb
        n_valid_chunk  = pair_valid.sum().item()
        if n_valid_chunk == 0:
            continue

        # Gather valid configs
        vi   = pair_valid.nonzero(as_tuple=False).squeeze(1)    # (n_valid,)
        bm_i = best_m[vi]                                        # (n_valid,)
        R_sel = R_cands[bm_i]                                   # (n_valid, 3, 3)

        # Translation: place tool so that p_B (rotated) lands at p_A
        # t = p_A - R @ p_B
        p_B_valid = p_B[vi]                                     # (n_valid, 3)
        p_A_valid = p_A[vi]                                     # (n_valid, 3)
        Rp_B = torch.einsum("nij, nj -> ni", R_sel, p_B_valid) # (n_valid, 3)
        t_sel = p_A_valid - Rp_B                                # (n_valid, 3)

        R_list.append(R_sel.cpu())
        t_list.append(t_sel.cpu())

    if not R_list:
        return torch.zeros(0, 3, 3), torch.zeros(0, 3)

    R_valid = torch.cat(R_list, dim=0)   # (N_valid, 3, 3)
    t_valid = torch.cat(t_list, dim=0)   # (N_valid, 3)
    print(f"  Rejection sampler: {R_valid.shape[0]} valid poses "
          f"from {n_total_pairs} contact pairs "
          f"({100*R_valid.shape[0]/max(n_total_pairs,1):.1f}% pair-success rate, "
          f"{M} rotations tested per pair)")
    return R_valid, t_valid


# ==============================================================================
#          FINAL PER-CONFIG SDF  (Kaolin — called only on N_valid survivors)
# ==============================================================================

def _kaolin_unsigned_distance(
    points: torch.Tensor,    # (B, P, 3)
    obj_verts: torch.Tensor, # (V, 3)
    obj_faces: torch.Tensor, # (F, 3)
) -> torch.Tensor:           # (B, P)
    B = points.shape[0]
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    ).expand(B, -1, -1, -1)
    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )
    return torch.sqrt(sq_dist.clamp(min=1e-12))


def _kaolin_inside(
    points: torch.Tensor,    # (B, P, 3)
    obj_verts: torch.Tensor, # (V, 3)
    obj_faces: torch.Tensor, # (F, 3)
) -> torch.Tensor:           # (B, P) bool  True = inside
    B = points.shape[0]
    verts_b = obj_verts.unsqueeze(0).expand(B, -1, -1)
    return kaolin.ops.mesh.check_sign(verts_b, obj_faces, points)


@torch.no_grad()
def compute_tool_pts_sdf(
    P_tool:    torch.Tensor,   # (P, 3) canonical tool pts
    R_tool:    torch.Tensor,   # (N, 3, 3)
    t_tool:    torch.Tensor,   # (N, 3)  mesh-origin translation
    obj_verts: torch.Tensor,   # (V, 3) world frame
    obj_faces: torch.Tensor,   # (F, 3)
) -> torch.Tensor:             # (N, P)
    """Signed SDF: canonical tool pts → object surface.  +outside / -inside."""
    N  = R_tool.shape[0]
    # pts_world = P_tool @ R^T + t
    pts_world = torch.einsum("pi, nij -> npj", P_tool, R_tool) + t_tool.unsqueeze(1)
    dist   = _kaolin_unsigned_distance(pts_world, obj_verts, obj_faces)  # (N, P)
    inside = _kaolin_inside(pts_world, obj_verts, obj_faces)             # (N, P)
    return torch.where(inside, -dist, dist).cpu()


@torch.no_grad()
def compute_obj_pts_sdf(
    P_obj_canonical: torch.Tensor,  # (Q, 3) canonical object pts (before R_obj)
    R_obj:    torch.Tensor,         # (3, 3)
    z_shift:  torch.Tensor,         # scalar
    tool_verts: torch.Tensor,       # (T, 3) canonical tool frame
    tool_faces: torch.Tensor,       # (G, 3)
    R_tool:   torch.Tensor,         # (N, 3, 3)
    t_tool:   torch.Tensor,         # (N, 3)
) -> torch.Tensor:                  # (N, Q)
    """Signed SDF: canonical object pts → tool surface.  +outside / -inside."""
    device = R_tool.device
    N  = R_tool.shape[0]

    # Object canonical → world frame (identical for all configs)
    p_world = P_obj_canonical @ R_obj.T                          # (Q, 3)
    p_world = p_world.clone()
    p_world[:, 2] -= z_shift

    # World → tool canonical frame: p_tool = R^T @ (p_world - t)
    R_T        = R_tool.permute(0, 2, 1)                         # (N, 3, 3)
    p_centered = p_world.unsqueeze(0) - t_tool.unsqueeze(1)      # (N, Q, 3)
    pts_tool   = torch.einsum("nij, nkj -> nki", R_T, p_centered)  # (N, Q, 3)

    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        tool_verts.unsqueeze(0), tool_faces
    ).expand(N, -1, -1, -1)
    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        pts_tool.contiguous(), face_verts
    )
    dist = torch.sqrt(sq_dist.clamp(min=1e-12))
    tool_verts_b = tool_verts.unsqueeze(0).expand(N, -1, -1)
    inside = kaolin.ops.mesh.check_sign(tool_verts_b, tool_faces, pts_tool)
    return torch.where(inside, -dist, dist).cpu()


# ==============================================================================
#                          SAVE
# ==============================================================================

def save_results(
    R_tool:          torch.Tensor,   # (N, 3, 3)  on any device
    t_tool_origin:   torch.Tensor,   # (N, 3)  mesh-origin trans (before centroid adj)
    P_uniform:       torch.Tensor,   # (K, 3)  canonical tool surface pts
    P_obj_canonical: torch.Tensor,   # (Q, 3)  canonical object pts (pre R_obj)
    obj_verts:       torch.Tensor,   # (V, 3)  world frame
    obj_faces:       torch.Tensor,   # (F, 3)
    tool_verts:      torch.Tensor,   # (T, 3)  canonical tool frame
    tool_faces:      torch.Tensor,   # (G, 3)
    R_obj:           torch.Tensor,   # (3, 3)
    z_shift:         torch.Tensor,   # scalar
    obj_scale:       float,
    cfg:             Config,
) -> int:
    """Compute final SDF fields, bake centroid-shifted coordinates, save to disk.

    .pt layout (identical to contact_gen.py minus contact_pts_* fields):

    Shared:
      tool_pts_canonical  (P,3)  centroid-subtracted
      obj_pts_canonical   (Q,3)  centroid-subtracted world frame
      obj_centroid        (3,)
      object_rotation / _object_rotation  (3,3)   R_obj
      obj_z_shift / _obj_z_shift          scalar
      tool_scale, object_scale, mesh paths

    Per-config (N entries):
      tool_translations (N,3)  world centroid position
      tool_rotations    (N,3,3)
      tool_pts_sdf      (N,P)  signed SDF tool canonical → object
      obj_pts_sdf       (N,Q)  signed SDF object canonical → tool
      pen_loss          (N,)   pseudo-metric (min SDF value per config)
      contact_loss      (N,)   placeholder zeros
    """
    N = R_tool.shape[0]
    device = cfg.device

    R_tool = R_tool.to(device)
    t_tool_origin = t_tool_origin.to(device)

    print(f"  Computing tool→object SDF ({P_uniform.shape[0]} pts × {N} configs) …")
    tool_pts_sdf = compute_tool_pts_sdf(
        P_uniform.to(device), R_tool, t_tool_origin,
        obj_verts, obj_faces,
    )   # (N, P) on cpu

    print(f"  Computing object→tool SDF ({P_obj_canonical.shape[0]} pts × {N} configs) …")
    obj_pts_sdf = compute_obj_pts_sdf(
        P_obj_canonical.to(device), R_obj, z_shift,
        tool_verts, tool_faces,
        R_tool, t_tool_origin,
    )   # (N, Q) on cpu

    # ── Bake centroid-shifted coordinates (identical logic to contact_gen.py) ──

    # 1. Center tool canonical cloud
    tool_centroid = P_uniform.mean(dim=0)           # (3,) mesh frame (device)
    P_tool_c      = (P_uniform - tool_centroid).cpu()  # (P,3) centered

    # 2. Adjust translation: world pos of tool CENTROID
    #    t_centroid = R @ centroid + t_origin
    t_adj = torch.einsum("nij, j -> ni", R_tool, tool_centroid) + t_tool_origin  # (N,3)

    # 3. Object canonical → world, subtract centroid
    obj_world    = P_obj_canonical.to(device) @ R_obj.T      # (Q,3)
    obj_world    = obj_world.clone()
    obj_world[:, 2] -= z_shift
    obj_centroid = obj_world.mean(dim=0)                     # (3,) world
    P_obj_c      = (obj_world - obj_centroid).cpu()          # (Q,3) centered

    # pen_loss proxy: per-config min SDF (higher = more clearance)
    pen_proxy = tool_pts_sdf.min(dim=1).values   # (N,) — already on cpu

    result = {
        # ── Mesh paths ─────────────────────────────────────────────────────
        "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
        "tool_mesh_path":   str(Path(cfg.tool_mesh_path).resolve()),
        # ── Scales ─────────────────────────────────────────────────────────
        "tool_scale":   cfg.tool_scale,
        "object_scale": obj_scale,
        # ── Canonical clouds ────────────────────────────────────────────────
        "tool_pts_canonical": P_tool_c,                   # (P,3)
        "obj_pts_canonical":  P_obj_c,                    # (Q,3)
        "obj_centroid":       obj_centroid.cpu(),          # (3,)
        # exact surface centroid used to define t_adj — needed for on-the-fly SDF
        "tool_centroid_raw":  tool_centroid.cpu(),         # (3,)
        # ── Object pose (stored with both public and private keys) ──────────
        "object_rotation":  R_obj.cpu(),
        "_object_rotation": R_obj.cpu(),
        "obj_z_shift":   z_shift.cpu() if isinstance(z_shift, torch.Tensor)
                         else torch.tensor(float(z_shift)),
        "_obj_z_shift":  z_shift.cpu() if isinstance(z_shift, torch.Tensor)
                         else torch.tensor(float(z_shift)),
        # ── Per-config ──────────────────────────────────────────────────────
        "tool_translations": t_adj.cpu(),        # (N,3)
        "tool_rotations":    R_tool.cpu(),        # (N,3,3)
        "pen_loss":          pen_proxy,           # (N,)  min SDF per config
        "contact_loss":      torch.zeros(N),      # (N,)  placeholder
        # ── SDF arrays ──────────────────────────────────────────────────────
        "tool_pts_sdf": tool_pts_sdf,             # (N,P)
        "obj_pts_sdf":  obj_pts_sdf,              # (N,Q)
    }

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    torch.save(result, cfg.output_path)
    print(f"  ✓ Saved {N} configs → {cfg.output_path}")
    return N


# ==============================================================================
#                                MAIN
# ==============================================================================

def main(cfg: Config) -> None:
    device = cfg.device
    torch.manual_seed(cfg.seed)

    # ── 1. Load meshes ────────────────────────────────────────────────────────
    print(f"Loading object mesh: {cfg.object_mesh_path}")
    obj_verts, obj_faces = load_mesh(cfg.object_mesh_path, device)

    print(f"Loading tool mesh:   {cfg.tool_mesh_path}")
    tool_verts, tool_faces = load_mesh(cfg.tool_mesh_path, device)

    # ── 2. Apply scales (must match contact_gen.py exactly) ──────────────────
    print(f"  Tool scale: {cfg.tool_scale}")
    tool_verts = tool_verts * cfg.tool_scale

    obj_scale = torch.empty(1, device=device).uniform_(
        cfg.object_scale_range[0], cfg.object_scale_range[1]
    ).item()
    print(f"  Object scale: {obj_scale:.4f}  (range {cfg.object_scale_range})")
    obj_verts = obj_verts * obj_scale

    # ── 3. Sample canonical object cloud (before pose randomisation) ──────────
    P_obj_canonical = sample_surface_points(obj_verts, obj_faces, cfg.num_tool_surface_pts)
    print(f"  P_obj_canonical: {P_obj_canonical.shape[0]} pts")

    # ── 4. Randomise object pose & ground ────────────────────────────────────
    print("Randomising object pose & grounding …")
    obj_verts, R_obj, z_shift = randomise_object_pose(obj_verts, obj_faces)

    # ── 5. Sample tool surface cloud ─────────────────────────────────────────
    P_uniform = sample_surface_points(tool_verts, tool_faces, cfg.K)
    print(f"  Tool surface pts (K={cfg.K}): {P_uniform.shape}")

    # ── 6. Build object SDF grid ──────────────────────────────────────────────
    print(f"Building {cfg.sdf_grid_res}³ SDF grid …")
    sdf_grid, bbox_min, bbox_max = build_sdf_grid(
        obj_verts, obj_faces, cfg.sdf_grid_res, device=device
    )
    print(f"  SDF grid: {sdf_grid.shape}  bbox {bbox_min.tolist()} → {bbox_max.tolist()}")

    # ── 7. Sample object surface points for contact-pair anchors ─────────────
    obj_surf = sample_surface_points(obj_verts, obj_faces, max(cfg.B * 4, 16384))
    print(f"  Object contact anchors: {obj_surf.shape[0]} pts")

    # ── 8. Head-area split for biased tool contact sampling ──────────────────
    head_area = load_tool_head_area(cfg.tools_json_path, cfg.tool_mesh_path)
    bounds    = compute_head_bounds(tool_verts, head_area)
    P_head, P_body = split_head_body(P_uniform, bounds)
    print(f"  Tool contact pts: {P_head.shape[0]} head / {P_body.shape[0]} body "
          f"(bias {cfg.contact_mode_prob:.0%} head)")

    # ── 9. Rejection sampling ─────────────────────────────────────────────────
    print(f"Running rejection sampler  (B={cfg.B}, M={cfg.M}, chunk_B={cfg.chunk_B}) …")
    R_valid, t_valid = rejection_sample(
        P_uniform, obj_surf, sdf_grid, bbox_min, bbox_max, cfg,
        P_head=P_head, P_body=P_body,
    )
    N_valid = R_valid.shape[0]
    print(f"\n✓ Valid contact poses found: {N_valid}")

    if N_valid == 0:
        print("⚠  No valid poses found. Try increasing B or M.")
        return

    # ── 9. Compute final SDF outputs & save ───────────────────────────────────
    n_saved = save_results(
        R_valid, t_valid,
        P_uniform, P_obj_canonical,
        obj_verts, obj_faces,
        tool_verts, tool_faces,
        R_obj, z_shift, obj_scale, cfg,
    )
    print(f"\nDone.  {n_saved} valid contact configurations saved.")


# ==============================================================================
#                           CLI ENTRY POINT
# ==============================================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Fast contact configuration generator (rejection sampling)."
    )
    p.add_argument("--object",     required=True)
    p.add_argument("--tool",       required=True)
    p.add_argument("--output",     default="contact_configs.pt")
    p.add_argument("--tools-json", default="")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--B",          type=int, default=CONTACT_GEN.B)
    p.add_argument("--M",          type=int, default=CONTACT_GEN.M)
    p.add_argument("--chunk-B",    type=int, default=CONTACT_GEN.chunk_B)
    args = p.parse_args()
    return Config(
        object_mesh_path=args.object,
        tool_mesh_path=args.tool,
        output_path=args.output,
        tools_json_path=args.tools_json,
        device=args.device,
        seed=args.seed,
        B=args.B,
        M=args.M,
        chunk_B=args.chunk_B,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

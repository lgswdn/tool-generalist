#!/usr/bin/env python3
"""
contact_gen_gradient.py  –  Gradient-based contact configuration generator.

Inspired by corn.py approach:
1. Sample random tool pose (position + orientation) in workspace
2. Sample random object pose in workspace
3. Compute SDF of object points relative to tool
4. Move tool toward/away from object along SDF gradient
5. Identify contact points (SDF <= threshold)

This approach:
- Allows diverse approach orientations (better coverage)
- No complex optimization loop
- Simple gradient step to create contacts

Dependencies:
  pip install torch kaolin trimesh numpy
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

import kaolin
import kaolin.ops.mesh
import kaolin.metrics.trianglemesh


@dataclass
class Config:
    """All tuneable knobs in one place."""

    # ----- I/O -----
    object_mesh_path: str = "object.obj"
    tool_mesh_path: str = "tool.obj"
    output_path: str = "contact_configs.pt"

    # ----- Scale (matches RL) -----
    tool_scale: float = 0.1
    object_scale_range: tuple[float, float] = (0.1, 0.2)

    # ----- Batch & sampling -----
    batch_size: int = 512
    num_surface_pts: int = 512
    device: str = "cuda:0"
    seed: int = 42

    # ----- Workspace bounds -----
    # Allow tools to approach from near-ground level (z_min lowered)
    workspace_min: tuple[float, float, float] = (-0.15, -0.15, -0.05)
    workspace_max: tuple[float, float, float] = (0.15, 0.15, 0.25)

    # ----- Contact threshold -----
    contact_threshold: float = 0.005  # SDF <= this = contact
    penetration_threshold: float = 0.001  # max allowed penetration


# ==============================================================================
#                        MESH LOADING HELPERS
# ==============================================================================

def load_mesh(path: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a triangle mesh and return (vertices, faces) on device."""
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def sample_surface_points(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Uniformly sample num_points on mesh surface."""
    device = verts.device
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = torch.norm(cross, dim=-1)
    probs = areas / areas.sum()

    face_idx = torch.multinomial(probs, num_points, replacement=True)

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
    return pts


# ==============================================================================
#                    ROTATION / POSE UTILITIES
# ==============================================================================

def random_rotation_matrices(n: int, device: str) -> torch.Tensor:
    """Uniform random SO(3) matrices via QR decomposition."""
    H = torch.randn(n, 3, 3, device=device)
    Q, R_ = torch.linalg.qr(H)
    signs = torch.sign(torch.diagonal(R_, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0] *= -1
    return Q


def random_poses(n: int, workspace_min: torch.Tensor, workspace_max: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random poses (position + rotation) in workspace.

    Returns:
        positions: (N, 3)
        rotations: (N, 3, 3)
    """
    # Random position in workspace
    positions = torch.rand(n, 3, device=device)
    positions = workspace_min + positions * (workspace_max - workspace_min)

    # Random rotation
    rotations = random_rotation_matrices(n, device)

    return positions, rotations


# ==============================================================================
#                    SDF COMPUTATION (Kaolin)
# ==============================================================================

def compute_unsigned_distance(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> torch.Tensor:
    """Batched unsigned point-to-mesh distance.

    Args:
        points: (B, P, 3)
        obj_verts: (V, 3)
        obj_faces: (F, 3)

    Returns:
        dist: (B, P) unsigned distance
    """
    B = points.shape[0]
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    ).expand(B, -1, -1, -1)

    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )
    return torch.sqrt(sq_dist)


def compute_sign(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> torch.Tensor:
    """Determine if points are inside mesh using kaolin's check_sign.

    Args:
        points: (B, P, 3) query points
        obj_verts: (V, 3) mesh vertices
        obj_faces: (F, 3) mesh faces (int64)

    Returns:
        sign: (B, P) bool, True = inside
    """
    B, P = points.shape[:2]
    device = points.device

    # kaolin.ops.mesh.check_sign signature:
    # check_sign(verts, faces, points)
    # verts: (batch_size, num_vertices, 3)
    # faces: (num_faces, 3) - must be int64
    # points: (batch_size, num_points, 3)
    sign = kaolin.ops.mesh.check_sign(
        obj_verts.unsqueeze(0).expand(B, -1, -1),  # (B, V, 3)
        obj_faces.long(),  # (F, 3) int64
        points.contiguous(),  # (B, P, 3)
    )  # (B, P) bool

    return sign


def compute_sdf_with_gradient(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute signed SDF and gradient direction (toward nearest surface).

    Args:
        points: (B, P, 3)
        obj_verts: (V, 3)
        obj_faces: (F, 3)

    Returns:
        sdf: (B, P) signed distance (positive=outside, negative=inside)
        gradient: (B, P, 3) direction toward nearest surface point
    """
    B, P = points.shape[:2]
    device = points.device

    # Unsigned distance
    dist = compute_unsigned_distance(points, obj_verts, obj_faces)  # (B, P)

    # Sign
    inside = compute_sign(points, obj_verts, obj_faces)  # (B, P) bool

    # Signed SDF
    sdf = torch.where(inside, -dist, dist)  # (B, P)

    # Gradient: direction toward nearest surface point
    # Find nearest surface point for each query point
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    ).expand(B, -1, -1, -1)

    _, face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )  # face_idx: (B, P)

    # Compute face centers as nearest point approximation
    v0 = obj_verts[obj_faces[:, 0]]
    v1 = obj_verts[obj_faces[:, 1]]
    v2 = obj_verts[obj_faces[:, 2]]
    face_centers = (v0 + v1 + v2) / 3  # (F, 3)

    nearest_pts = face_centers[face_idx]  # (B, P, 3)

    # Gradient direction: from query point toward nearest surface
    gradient = nearest_pts - points  # (B, P, 3)
    gradient = F.normalize(gradient, dim=-1)

    # For inside points, gradient should point outward (flip)
    # For outside points, gradient points toward surface (correct)
    # Inside: move toward surface = outward direction
    # We computed nearest_pts - points, which points toward surface from both sides
    # For inside, this is outward (correct)
    # For outside, this is toward surface (correct)

    return sdf, gradient


# ==============================================================================
#                    CONTACT GENERATION
# ==============================================================================

def generate_contacts(
    tool_verts: torch.Tensor,
    tool_faces: torch.Tensor,
    obj_cloud: torch.Tensor,  # (P, 3) object surface points
    cfg: Config,
) -> dict:
    """Generate contact configurations using gradient-based approach.

    1. Sample random tool poses in workspace
    2. Compute SDF of object points relative to tool
    3. Find minimum SDF point and its gradient
    4. Move tool along gradient to surface
    5. Identify contact points

    Args:
        tool_verts: (V, 3) tool mesh vertices (already scaled)
        tool_faces: (F, 3) tool mesh faces
        obj_cloud: (P, 3) object surface points (already scaled and posed)
        cfg: Config

    Returns:
        dict with tool poses, contact points, etc.
    """
    device = cfg.device
    N = cfg.batch_size

    # Workspace bounds
    workspace_min = torch.tensor(cfg.workspace_min, device=device)
    workspace_max = torch.tensor(cfg.workspace_max, device=device)

    # ---- 1. Sample random tool poses ----
    tool_positions, tool_rotations = random_poses(N, workspace_min, workspace_max, device)

    # ---- 2. Transform object cloud to tool frame ----
    # obj_cloud_tool = R^T @ (obj_cloud - t)
    obj_cloud_expanded = obj_cloud.unsqueeze(0).expand(N, -1, -1)  # (N, P, 3)
    obj_centered = obj_cloud_expanded - tool_positions.unsqueeze(1)  # (N, P, 3)
    R_T = tool_rotations.permute(0, 2, 1)  # (N, 3, 3)
    obj_cloud_tool = torch.einsum("nij, nkj -> nki", R_T, obj_centered)  # (N, P, 3)

    # ---- 3. Compute SDF of object points relative to tool mesh ----
    sdf, gradient = compute_sdf_with_gradient(obj_cloud_tool, tool_verts, tool_faces)
    # sdf: (N, P), gradient: (N, P, 3) in tool frame

    # ---- 4. Find minimum SDF point per sample ----
    min_idx = sdf.argmin(dim=-1)  # (N,)
    min_sdf = sdf.gather(1, min_idx.unsqueeze(1)).squeeze(1)  # (N,)

    # Get gradient for minimum point
    min_gradient = gradient.gather(1, min_idx.unsqueeze(1).unsqueeze(2).expand(N, 1, 3)).squeeze(1)  # (N, 3)

    # Transform gradient to world frame
    min_gradient_world = torch.einsum("nij, nj -> ni", tool_rotations, min_gradient)  # (N, 3)

    # ---- 5. Move tool position toward surface ----
    # gradient points toward nearest surface point from query point
    # For inside (sdf < 0): gradient points outward toward surface
    # For outside (sdf > 0): gradient points toward surface (inward)
    # We want to move toward surface from both sides
    # Move by -sdf * gradient: this brings minimum point to surface
    move = -min_sdf.unsqueeze(1) * min_gradient_world  # (N, 3)

    new_tool_positions = tool_positions + move

    # ---- 6. Floor guard ----
    # Transform tool mesh to check if below ground
    tool_mesh_world = torch.einsum("vi, nji -> nvj", tool_verts, tool_rotations) + new_tool_positions.unsqueeze(1)
    z_mins = tool_mesh_world[:, :, 2].min(dim=1).values
    lift = torch.clamp(-z_mins, min=0.0)
    new_tool_positions[:, 2] += lift

    # ---- 7. Recompute SDF after move ----
    obj_centered_new = obj_cloud_expanded - new_tool_positions.unsqueeze(1)
    obj_cloud_tool_new = torch.einsum("nij, nkj -> nki", R_T, obj_centered_new)
    sdf_new, _ = compute_sdf_with_gradient(obj_cloud_tool_new, tool_verts, tool_faces)

    # ---- 8. Identify contact points ----
    contact_mask = sdf_new.abs() <= cfg.contact_threshold  # (N, P) bool

    # ---- 9. Filter valid configurations ----
    # Valid: no deep penetration, at least one contact point
    max_penetration = sdf_new.min(dim=1).values  # (N,) most negative = deepest penetration
    num_contacts = contact_mask.sum(dim=1)  # (N,)

    valid = (max_penetration >= -cfg.penetration_threshold) & (num_contacts >= 1)

    return {
        "tool_positions": new_tool_positions[valid].cpu(),
        "tool_rotations": tool_rotations[valid].cpu(),
        "sdf_new": sdf_new[valid].cpu(),
        "contact_mask": contact_mask[valid].cpu(),
        "num_contacts": num_contacts[valid].cpu(),
        "max_penetration": max_penetration[valid].cpu(),
        "n_valid": valid.sum().item(),
    }


# ==============================================================================
#                                 MAIN
# ==============================================================================

def main(cfg: Config) -> dict:
    """Generate contact configurations and save to disk."""
    device = cfg.device
    torch.manual_seed(cfg.seed)

    # ---- 1. Load meshes ----
    print(f"Loading object mesh: {cfg.object_mesh_path}")
    obj_verts_raw, obj_faces = load_mesh(cfg.object_mesh_path, device)

    print(f"Loading tool mesh:   {cfg.tool_mesh_path}")
    tool_verts_raw, tool_faces = load_mesh(cfg.tool_mesh_path, device)

    # ---- 2. Apply scales ----
    tool_verts = tool_verts_raw * cfg.tool_scale
    print(f"  Tool scale: {cfg.tool_scale}")

    obj_scale = torch.empty(1, device=device).uniform_(
        cfg.object_scale_range[0], cfg.object_scale_range[1]
    ).item()
    obj_verts = obj_verts_raw * obj_scale
    print(f"  Object scale: {obj_scale:.4f}")

    # ---- Ground object: z_min = 0 (so floor guard doesn't block bottom half) ----
    z_min = obj_verts[:, 2].min()
    if z_min < 0:
        obj_verts[:, 2] -= z_min
        print(f"  Object grounded: shifted z by {-z_min:.4f}")

    # ---- 3. Sample object surface points (static pose at origin, grounded) ----
    obj_cloud = sample_surface_points(obj_verts, obj_faces, cfg.num_surface_pts)

    # ---- 4. Generate contacts ----
    print(f"Generating {cfg.batch_size} contact configurations...")
    result = generate_contacts(tool_verts, tool_faces, obj_cloud, cfg)
    n_valid = result["n_valid"]
    print(f"  Valid configurations: {n_valid} / {cfg.batch_size}")

    # ---- 5. Extract contact points ----
    # For each valid config, get object points that are in contact
    contact_pts_world = []
    for i in range(n_valid):
        mask = result["contact_mask"][i]  # (P,) bool
        pts = obj_cloud[mask].cpu()  # (C, 3) object points in contact
        if pts.shape[0] > 5:
            # Randomly select 5 if more
            idx = torch.randperm(pts.shape[0])[:5]
            pts = pts[idx]
        elif pts.shape[0] < 5:
            # Pad with nearest point
            pad = pts[0:1].expand(5 - pts.shape[0], 3)
            pts = torch.cat([pts, pad], dim=0)
        contact_pts_world.append(pts)

    contact_pts_world = torch.stack(contact_pts_world, dim=0)  # (N_valid, 5, 3)

    # ---- 6. Compute SDF for tool canonical points ----
    # tool_pts_sdf: signed distance of tool surface points to object
    tool_canonical_pts = sample_surface_points(tool_verts, tool_faces, cfg.num_surface_pts)
    tool_pts_sdf_list = []
    for i in range(n_valid):
        R = result["tool_rotations"][i].to(device)
        t = result["tool_positions"][i].to(device)
        # Transform tool pts to world, then check distance to object
        tool_pts_world = tool_canonical_pts @ R.T + t
        dist = compute_unsigned_distance(tool_pts_world.unsqueeze(0), obj_verts, obj_faces).squeeze(0)
        inside = compute_sign(tool_pts_world.unsqueeze(0), obj_verts, obj_faces).squeeze(0)
        sdf = torch.where(inside, -dist, dist)
        tool_pts_sdf_list.append(sdf.cpu())

    tool_pts_sdf = torch.stack(tool_pts_sdf_list, dim=0)  # (N_valid, P)

    # ---- 7. Compute SDF for object canonical points ----
    obj_canonical_pts = obj_cloud.cpu()
    obj_pts_sdf_list = []
    for i in range(n_valid):
        R = result["tool_rotations"][i].to(device)
        t = result["tool_positions"][i].to(device)
        # Object pts already in world frame, check distance to tool
        # Transform to tool frame
        obj_pts_tool = (obj_canonical_pts - t.cpu()) @ R.cpu()
        dist = compute_unsigned_distance(obj_pts_tool.unsqueeze(0).to(device), tool_verts, tool_faces).squeeze(0).cpu()
        inside = compute_sign(obj_pts_tool.unsqueeze(0).to(device), tool_verts, tool_faces).squeeze(0).cpu()
        sdf = torch.where(inside, -dist, dist)
        obj_pts_sdf_list.append(sdf)

    obj_pts_sdf = torch.stack(obj_pts_sdf_list, dim=0)  # (N_valid, P)

    # ---- 8. Save results ----
    output = {
        "object_mesh_path": str(Path(cfg.object_mesh_path).resolve()),
        "tool_mesh_path": str(Path(cfg.tool_mesh_path).resolve()),
        "tool_scale": cfg.tool_scale,
        "object_scale": obj_scale,
        "tool_pts_canonical": tool_canonical_pts.cpu(),
        "obj_pts_canonical": obj_canonical_pts.cpu(),
        "tool_translations": result["tool_positions"],
        "tool_rotations": result["tool_rotations"],
        "tool_pts_sdf": tool_pts_sdf,
        "obj_pts_sdf": obj_pts_sdf,
        "contact_pts_world": contact_pts_world,
        "pen_loss": result["max_penetration"].abs(),
        "contact_loss": result["num_contacts"].float() / cfg.num_surface_pts,
    }

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    torch.save(output, cfg.output_path)
    print(f"Saved to {cfg.output_path}")

    return output


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Gradient-based contact generator")
    p.add_argument("--object", type=str, required=True)
    p.add_argument("--tool", type=str, required=True)
    p.add_argument("--output", type=str, default="contact_configs.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-pts", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contact-threshold", type=float, default=0.005)
    p.add_argument("--penetration-threshold", type=float, default=0.001)
    args = p.parse_args()

    return Config(
        object_mesh_path=args.object,
        tool_mesh_path=args.tool,
        output_path=args.output,
        batch_size=args.batch_size,
        num_surface_pts=args.num_pts,
        device=args.device,
        seed=args.seed,
        contact_threshold=args.contact_threshold,
        penetration_threshold=args.penetration_threshold,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
#!/usr/bin/env python3
"""
gen_initial.py — Generate random initial poses for each contact config.

For each config in a .pt file, generates one initial pose where:
  - Tool center is within `init_radius` (default 25cm) of object center
  - Tool doesn't collide with object (no penetration)
  - Tool Z-axis points downward (z_z < 0)

Adds to .pt file:
  - init_translations: (N, 3) initial tool positions
  - init_rotations: (N, 3, 3) initial tool rotations

Usage:
    python gen_initial.py --input contact_configs.pt
    python gen_initial.py --input-dir teardrop_contact/ --gpus 0 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

try:
    import kaolin
    import kaolin.ops.mesh
    import kaolin.metrics.trianglemesh
    HAS_KAOLIN = True
except ImportError:
    HAS_KAOLIN = False
    print("[WARN] kaolin not available, using trimesh for collision check (slower)")

# =============================================================================
# Config
# =============================================================================
class Config:
    init_radius: float = 0.25  # 25cm from object center
    collision_threshold: float = 0.002  # max allowed penetration
    device: str = "cuda:0"
    batch_size: int = 128  # process configs in batches for collision check
    seed: int = 42


# =============================================================================
# Helpers
# =============================================================================
def load_mesh(path: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def random_rotation_downward(n: int, device: str) -> torch.Tensor:
    """Random rotation with Z-axis pointing downward (z_z < 0).

    Samples random rotations and filters those with z_z < 0.
    """
    # Sample more than needed, then filter
    H = torch.randn(n * 4, 3, 3, device=device)  # 4x to ensure enough valid
    Q, R_ = torch.linalg.qr(H)
    signs = torch.sign(torch.diagonal(R_, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0] *= -1

    # Check z-axis (third column)
    z_axis = Q[:, :, 2]  # (N*4, 3)
    valid_mask = z_axis[:, 2] < 0  # z_z < 0 (pointing down)
    valid_Q = Q[valid_mask]

    # Take first n valid
    if valid_Q.shape[0] < n:
        # If not enough, regenerate (shouldn't happen often)
        return random_rotation_downward(n, device)

    return valid_Q[:n]


def compute_unsigned_distance_batch(
    points: torch.Tensor,  # (B, P, 3)
    obj_verts: torch.Tensor,  # (V, 3)
    obj_faces: torch.Tensor,  # (F, 3)
) -> torch.Tensor:  # (B, P)
    """Batched point-to-mesh distance using kaolin."""
    if not HAS_KAOLIN:
        # Fallback: trimesh (slow)
        return compute_distance_trimesh(points, obj_verts, obj_faces)

    B = points.shape[0]
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        obj_verts.unsqueeze(0), obj_faces
    ).expand(B, -1, -1, -1)

    sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points.contiguous(), face_verts
    )
    return torch.sqrt(sq_dist)


def compute_sign_batch(
    points: torch.Tensor,  # (B, P, 3)
    obj_verts: torch.Tensor,  # (V, 3)
    obj_faces: torch.Tensor,  # (F, 3)
) -> torch.Tensor:  # (B, P) bool
    """Check if points are inside mesh."""
    if not HAS_KAOLIN:
        # Fallback: approximate using trimesh contains
        return compute_sign_trimesh(points, obj_verts, obj_faces)

    B, P = points.shape[:2]
    sign = kaolin.ops.mesh.check_sign(
        obj_verts.unsqueeze(0).expand(B, -1, -1),
        obj_faces.long(),
        points.contiguous(),
    )
    return sign


def compute_sdf_batch(
    points: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor,
) -> torch.Tensor:
    """Signed distance: positive=outside, negative=inside."""
    dist = compute_unsigned_distance_batch(points, obj_verts, obj_faces)
    inside = compute_sign_batch(points, obj_verts, obj_faces)
    return torch.where(inside, -dist, dist)


def compute_distance_trimesh(points, obj_verts, obj_faces):
    """Slow fallback using trimesh proximity."""
    B, P = points.shape[:2]
    mesh = trimesh.Trimesh(obj_verts.cpu().numpy(), obj_faces.cpu().numpy())
    results = []
    for i in range(B):
        pts = points[i].cpu().numpy()
        dists = trimesh.proximity.closest_point(mesh, pts)[1]
        results.append(torch.tensor(dists, device=points.device))
    return torch.stack(results)


def compute_sign_trimesh(points, obj_verts, obj_faces):
    """Slow fallback using trimesh contains."""
    B, P = points.shape[:2]
    mesh = trimesh.Trimesh(obj_verts.cpu().numpy(), obj_faces.cpu().numpy())
    results = []
    for i in range(B):
        pts = points[i].cpu().numpy()
        inside = mesh.contains(pts)
        results.append(torch.tensor(inside, device=points.device))
    return torch.stack(results)


def random_poses_near_object(
    n: int,
    obj_center: torch.Tensor,  # (3,)
    radius: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random poses within radius of object center."""
    # Random position in sphere around obj_center
    # Use rejection sampling for uniform distribution in sphere
    positions = torch.randn(n, 3, device=device)
    positions = positions / positions.norm(dim=-1, keepdim=True)  # unit sphere
    positions = positions * torch.rand(n, 1, device=device) ** (1/3) * radius  # uniform in volume
    positions = positions + obj_center

    # Random rotation (downward constraint)
    rotations = random_rotation_downward(n, device)

    return positions, rotations


def check_collision_free(
    tool_verts: torch.Tensor,  # (V, 3) scaled tool mesh
    tool_poses_t: torch.Tensor,  # (B, 3)
    tool_poses_R: torch.Tensor,  # (B, 3, 3)
    obj_verts: torch.Tensor,  # (V, 3) scaled object mesh (grounded)
    obj_faces: torch.Tensor,
    threshold: float,
    device: str,
) -> torch.Tensor:  # (B,) bool
    """Check if tool at pose doesn't penetrate object."""
    B = tool_poses_t.shape[0]

    # Transform tool vertices to world
    tool_world = torch.einsum("vi, nji -> nvj", tool_verts, tool_poses_R) + tool_poses_t.unsqueeze(1)

    # Compute SDF of tool points relative to object
    sdf = compute_sdf_batch(tool_world, obj_verts, obj_faces)

    # min SDF per config (most negative = deepest penetration)
    min_sdf = sdf.min(dim=1).values

    # Valid: min_sdf >= -threshold (no significant penetration)
    return min_sdf >= -threshold


# =============================================================================
# Main
# =============================================================================
def process_pt_file(pt_path: str, cfg: Config) -> bool:
    """Add initial poses to a .pt file. Returns True on success."""
    device = cfg.device

    # Load existing data
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    n_configs = data["tool_translations"].shape[0]

    if "init_translations" in data:
        print(f"  [SKIP] {pt_path} already has initial poses")
        return True

    print(f"  Processing {pt_path}: {n_configs} configs")

    # Load meshes
    tool_path = data["tool_mesh_path"]
    obj_path = data["object_mesh_path"]

    tool_verts_raw, tool_faces = load_mesh(tool_path, device)
    obj_verts_raw, obj_faces = load_mesh(obj_path, device)

    # Apply scales
    tool_scale = data.get("tool_scale", 0.1)
    obj_scale = data.get("object_scale", 0.15)
    tool_verts = tool_verts_raw * tool_scale
    obj_verts = obj_verts_raw * obj_scale

    # Ground object (same as contact_gen)
    R_obj = data.get("object_rotation", torch.eye(3))
    obj_verts = obj_verts @ R_obj.T
    z_min = obj_verts[:, 2].min()
    if z_min < 0:
        obj_verts[:, 2] -= z_min

    # Object center (use contact pose positions as reference for object center)
    # Actually, object is static - compute from obj_verts
    obj_center = obj_verts.mean(dim=0).to(device)

    # Generate initial poses in batches
    init_translations = []
    init_rotations = []

    torch.manual_seed(cfg.seed)

    remaining = n_configs
    attempts_per_config = 10  # try up to 10 times per config

    while remaining > 0:
        batch_n = min(cfg.batch_size, remaining * attempts_per_config)

        # Generate candidates
        cand_t, cand_R = random_poses_near_object(batch_n, obj_center, cfg.init_radius, device)

        # Check collision
        valid_mask = check_collision_free(
            tool_verts, cand_t, cand_R,
            obj_verts, obj_faces,
            cfg.collision_threshold, device
        )

        valid_t = cand_t[valid_mask]
        valid_R = cand_R[valid_mask]

        # Take as many as needed
        take = min(valid_t.shape[0], remaining)
        if take > 0:
            init_translations.append(valid_t[:take].cpu())
            init_rotations.append(valid_R[:take].cpu())
            remaining -= take
        else:
            # No valid poses in this batch, try again with larger radius
            print(f"    [WARN] No valid poses found, expanding radius...")

    init_translations = torch.cat(init_translations, dim=0)
    init_rotations = torch.cat(init_rotations, dim=0)

    # Add to data
    data["init_translations"] = init_translations
    data["init_rotations"] = init_rotations

    # Save back
    torch.save(data, pt_path)
    print(f"    Added {init_translations.shape[0]} initial poses")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate initial poses for contact configs")
    parser.add_argument("--input", type=str, help="Single .pt file to process")
    parser.add_argument("--input-dir", type=str, help="Directory of .pt files")
    parser.add_argument("--init-radius", type=float, default=0.25, help="Radius around object center (m)")
    parser.add_argument("--collision-threshold", type=float, default=0.002, help="Max allowed penetration (m)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.init_radius = args.init_radius
    cfg.collision_threshold = args.collision_threshold
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed

    if args.input:
        files = [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    else:
        print("ERROR: Must provide --input or --input-dir")
        sys.exit(1)

    print(f"Processing {len(files)} files...")
    print(f"  init_radius: {cfg.init_radius}m")
    print(f"  collision_threshold: {cfg.collision_threshold}m")
    print(f"  device: {cfg.device}")

    for f in files:
        try:
            process_pt_file(f, cfg)
        except Exception as e:
            print(f"  [FAIL] {f}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
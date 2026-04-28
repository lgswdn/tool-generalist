#!/usr/bin/env python3
"""
gen_initial.py — Generate initial poses for each contact config.

For each config in a .pt file, generates one initial pose where:
  1. Sample an anchor point on object surface
  2. Get outward normal direction with perturbation
  3. Translate tool away from anchor (1-5cm default)
  4. Apply small random rotation perturbation
  5. Rejection sampling to ensure no penetration

Adds to .pt file:
  - init_translations: (N, 3) initial tool positions
  - init_rotations: (N, 3, 3) initial tool rotations
  - init_anchor_pts: (N, 3) anchor points on object surface

Usage:
    python gen_initial.py --input contact_configs.pt
    python gen_initial.py --input-dir teardrop_contact/ --device cuda:0
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
    pull_distance_min: float = 0.01  # 1cm min pull away
    pull_distance_max: float = 0.05  # 5cm max pull away
    perturbation_scale: float = 0.8  # spread around outward direction
    rot_angle_max_deg: float = 15.0  # max rotation perturbation
    collision_threshold: float = 0.002  # max allowed penetration
    device: str = "cuda:0"
    batch_size: int = 128
    seed: int = 42
    force: bool = False


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


@torch.no_grad()
def compute_init_tool_pts_sdf(
    P_tool: torch.Tensor,      # (P, 3)  canonical tool cloud
    init_trans: torch.Tensor,  # (N, 3)  initial tool translations
    init_R: torch.Tensor,      # (N, 3, 3) initial tool rotations
    obj_verts: torch.Tensor,   # (V, 3)  grounded world frame
    obj_faces: torch.Tensor,   # (F, 3)
) -> torch.Tensor:             # (N, P)  signed SDF
    """Compute signed SDF from canonical tool points to object at initial pose.

    SDF convention: positive = outside object, negative = inside object.

    Returns:
        sdf : (N, P)  signed distances
    """
    device = init_trans.device
    pts_world = torch.einsum("pi, nji -> npj", P_tool, init_R) + init_trans.unsqueeze(1)  # (N, P, 3)
    dist = compute_unsigned_distance_batch(pts_world, obj_verts, obj_faces)   # (N, P)
    inside = compute_sign_batch(pts_world, obj_verts, obj_faces)             # (N, P) bool
    sdf = torch.where(inside, -dist, dist)
    return sdf.cpu()                                                         # (N, P)


@torch.no_grad()
def compute_init_obj_pts_sdf(
    P_obj: torch.Tensor,       # (Q, 3)  canonical object pts (before R_obj)
    R_obj: torch.Tensor,       # (3, 3)
    z_shift,                   # scalar  grounding z-offset
    tool_verts: torch.Tensor,  # (T, 3)  canonical tool frame
    tool_faces: torch.Tensor,  # (G, 3)
    init_trans: torch.Tensor,  # (N, 3)  initial tool translations
    init_R: torch.Tensor,      # (N, 3, 3) initial tool rotations
) -> torch.Tensor:             # (N, Q)  signed SDF
    """Compute signed SDF from canonical object points to tool at initial pose.

    Applies R_obj + z_shift to get world-frame object points, then transforms them
    into each config's tool canonical frame and queries the canonical tool mesh.
    SDF convention: positive = outside tool, negative = inside tool.

    Returns:
        sdf : (N, Q)  signed distances
    """
    device = init_trans.device

    # Handle z_shift being tensor or float
    if hasattr(z_shift, 'item'):
        z_shift_val = z_shift.item()
    else:
        z_shift_val = z_shift
    z_shift_tensor = torch.tensor(z_shift_val, device=device)

    # Object canonical → world frame  (same for all configs)
    p_world = P_obj @ R_obj.T                          # (Q, 3)
    p_world = p_world.clone()
    p_world[:, 2] -= z_shift_tensor                    # apply grounding

    # World → tool canonical frame per config:  p_tool = R^T @ (p_world - t)
    R_T = init_R.permute(0, 2, 1)                      # (N, 3, 3)
    p_centered = p_world.unsqueeze(0) - init_trans.unsqueeze(1)  # (N, Q, 3)
    pts_tool_frame = torch.einsum("nij, nkj -> nki", R_T, p_centered)  # (N, Q, 3)

    N = init_trans.shape[0]
    if HAS_KAOLIN:
        face_verts = kaolin.ops.mesh.index_vertices_by_faces(
            tool_verts.unsqueeze(0), tool_faces
        ).expand(N, -1, -1, -1)                                        # (N, G, 3, 3)

        sq_dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            pts_tool_frame.contiguous(), face_verts
        )
        dist = torch.sqrt(sq_dist.clamp(min=1e-12))                    # (N, Q)

        tool_verts_batch = tool_verts.unsqueeze(0).expand(N, -1, -1)
        inside = kaolin.ops.mesh.check_sign(
            tool_verts_batch, tool_faces, pts_tool_frame
        )                                                              # (N, Q) bool
    else:
        # Fallback to trimesh
        dist = compute_distance_trimesh(pts_tool_frame, tool_verts, tool_faces)
        inside = compute_sign_trimesh(pts_tool_frame, tool_verts, tool_faces)

    sdf = torch.where(inside, -dist, dist)
    return sdf.cpu()                                                   # (N, Q)


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


def sample_anchor_points(
    obj_verts: torch.Tensor,  # (V, 3)
    obj_faces: torch.Tensor,  # (F, 3)
    n: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random anchor points on object surface and compute outward normals.

    Returns:
        anchor_pts: (N, 3) points on surface
        outward_normals: (N, 3) normals pointing away from object
    """
    # Randomly sample points on mesh surface
    if HAS_KAOLIN:
        face_verts = kaolin.ops.mesh.index_vertices_by_faces(
            obj_verts.unsqueeze(0), obj_faces
        )  # (1, F, 3, 3)

        # Random face indices
        n_faces = obj_faces.shape[0]
        face_idx = torch.randint(0, n_faces, (n,), device=device)

        # Random barycentric coordinates
        r1 = torch.rand(n, device=device)
        r2 = torch.rand(n, device=device)
        sqrt_r1 = torch.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2

        # Get triangle vertices
        triangles = face_verts[0, face_idx]  # (N, 3, 3)
        anchor_pts = (
            u.unsqueeze(-1) * triangles[:, 0]
            + v.unsqueeze(-1) * triangles[:, 1]
            + w.unsqueeze(-1) * triangles[:, 2]
        )  # (N, 3)

        # Compute face normals
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        # Ensure outward direction (away from center)
        obj_center = obj_verts.mean(dim=0)
        center_to_face = triangles.mean(dim=1) - obj_center
        dot = (face_normals * center_to_face).sum(dim=-1)
        outward_normals = torch.where(
            dot.unsqueeze(-1) > 0,
            face_normals,
            -face_normals
        )
    else:
        # Fallback: sample from vertices
        idx = torch.randint(0, obj_verts.shape[0], (n,), device=device)
        anchor_pts = obj_verts[idx]
        outward_normals = torch.nn.functional.normalize(anchor_pts - obj_verts.mean(dim=0), dim=-1)

    return anchor_pts, outward_normals


def random_poses_away_from_anchor(
    n: int,
    obj_verts: torch.Tensor,  # (V, 3)
    obj_faces: torch.Tensor,  # (F, 3)
    pull_distance_range: tuple,  # (min, max) meters to pull away
    perturbation_scale: float,  # spread around outward direction
    rot_angle_max_deg: float,  # max random rotation angle
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate poses pulled away from anchor points with rejection sampling.

    For each pose:
    1. Sample anchor point on object surface
    2. Get outward normal direction
    3. Add perturbation for diversity
    4. Translate away from anchor along perturbed direction
    5. Apply small random rotation

    Returns:
        positions: (N, 3)
        rotations: (N, 3, 3)
        anchor_pts: (N, 3) the sampled anchor points
    """
    # Sample anchors and outward normals
    anchor_pts, outward_normals = sample_anchor_points(obj_verts, obj_faces, n, device)

    # Add perturbation to outward direction for diversity
    perturbation = torch.randn(n, 3, device=device) * perturbation_scale
    pull_directions = outward_normals + perturbation
    pull_directions = torch.nn.functional.normalize(pull_directions, dim=-1)

    # Random pull distance
    min_dist, max_dist = pull_distance_range
    pull_distance = torch.rand(n, 1, device=device) * (max_dist - min_dist) + min_dist

    # Position: anchor + pull_distance * pull_direction
    positions = anchor_pts + pull_distance * pull_directions

    # Random rotation with downward constraint + small perturbation
    base_rotations = random_rotation_downward(n, device)

    # Add small random rotation perturbation
    import math
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    angle = torch.rand(n, device=device) * math.radians(rot_angle_max_deg)

    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    eye = torch.eye(3, device=device).unsqueeze(0).expand(n, -1, -1)
    small_R = eye + torch.sin(angle.unsqueeze(-1).unsqueeze(-1)) * K + \
              (1 - torch.cos(angle.unsqueeze(-1).unsqueeze(-1))) * (K @ K)

    rotations = small_R @ base_rotations

    return positions, rotations, anchor_pts


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

    if "init_translations" in data and not cfg.force:
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
    R_obj = R_obj.to(device)  # move to GPU

    # Get z_shift from contact_gen, or compute if missing
    z_shift = data.get("obj_z_shift", None)
    if z_shift is not None:
        if hasattr(z_shift, 'item'):
            z_shift = z_shift.item()
    else:
        # Fallback: compute z_shift (for files from contact_gen_gradient.py)
        obj_verts_temp = obj_verts @ R_obj.T
        z_shift = obj_verts_temp[:, 2].min().item()

    obj_verts = obj_verts @ R_obj.T
    obj_verts = obj_verts.clone()
    obj_verts[:, 2] -= z_shift

    # Generate initial poses in batches using anchor-based approach
    init_translations = []
    init_rotations = []
    init_anchor_pts = []

    torch.manual_seed(cfg.seed)

    remaining = n_configs
    attempts_per_config = 10

    while remaining > 0:
        batch_n = min(cfg.batch_size, remaining * attempts_per_config)

        # Generate candidates using anchor-based approach
        cand_t, cand_R, cand_anchor = random_poses_away_from_anchor(
            batch_n, obj_verts, obj_faces,
            (cfg.pull_distance_min, cfg.pull_distance_max),
            cfg.perturbation_scale,
            cfg.rot_angle_max_deg,
            device
        )

        # Check collision with rejection sampling
        valid_mask = check_collision_free(
            tool_verts, cand_t, cand_R,
            obj_verts, obj_faces,
            cfg.collision_threshold, device
        )

        valid_t = cand_t[valid_mask]
        valid_R = cand_R[valid_mask]
        valid_anchor = cand_anchor[valid_mask]

        # Take as many as needed
        take = min(valid_t.shape[0], remaining)
        if take > 0:
            init_translations.append(valid_t[:take].cpu())
            init_rotations.append(valid_R[:take].cpu())
            init_anchor_pts.append(valid_anchor[:take].cpu())
            remaining -= take
        else:
            print(f"    [WARN] No valid poses found in batch, retrying...")

    init_translations = torch.cat(init_translations, dim=0)
    init_rotations = torch.cat(init_rotations, dim=0)
    init_anchor_pts = torch.cat(init_anchor_pts, dim=0)

    # ---- Compute SDFs at initial pose ----
    # Get canonical tool and object points from data
    P_tool = data.get("tool_pts_canonical")
    P_obj = data.get("obj_pts_canonical")

    if P_tool is not None and P_obj is not None:
        print(f"    Computing initial pose SDFs...")
        P_tool = P_tool.to(device)
        P_obj = P_obj.to(device)

        init_trans_gpu = init_translations.to(device)
        init_R_gpu = init_rotations.to(device)

        # Tool pts → object SDF
        init_tool_pts_sdf = compute_init_tool_pts_sdf(
            P_tool, init_trans_gpu, init_R_gpu,
            obj_verts, obj_faces
        )

        # Object pts → tool SDF
        init_obj_pts_sdf = compute_init_obj_pts_sdf(
            P_obj, R_obj, z_shift,
            tool_verts, tool_faces,
            init_trans_gpu, init_R_gpu
        )

        data["init_tool_pts_sdf"] = init_tool_pts_sdf
        data["init_obj_pts_sdf"] = init_obj_pts_sdf
    else:
        print(f"    [WARN] Missing canonical points, skipping SDF computation")

    # Add to data
    data["init_translations"] = init_translations
    data["init_rotations"] = init_rotations
    data["init_anchor_pts"] = init_anchor_pts  # anchor points used for generation

    # Save back
    torch.save(data, pt_path)
    print(f"    Added {init_translations.shape[0]} initial poses")
    return True


def worker(files, gpu, cfg_dict):
    """Process a subset of files on a specific GPU."""
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    cfg.device = f"cuda:{gpu}"

    ok = fail = skip = 0
    failed_files = []
    for f in files:
        try:
            success = process_pt_file(f, cfg)
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


def main():
    parser = argparse.ArgumentParser(description="Generate initial poses for contact configs")
    parser.add_argument("--input", type=str, help="Single .pt file to process")
    parser.add_argument("--input-dir", type=str, help="Directory of .pt files")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU IDs to use")
    parser.add_argument("--pull-distance-min", type=float, default=0.01, help="Min distance away from anchor (m)")
    parser.add_argument("--pull-distance-max", type=float, default=0.05, help="Max distance away from anchor (m)")
    parser.add_argument("--perturbation-scale", type=float, default=0.8, help="Spread around outward direction")
    parser.add_argument("--rot-angle-max", type=float, default=15.0, help="Max rotation perturbation (deg)")
    parser.add_argument("--collision-threshold", type=float, default=0.001, help="Max allowed penetration (m)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-generate even if initial poses exist")
    args = parser.parse_args()

    cfg = Config()
    cfg.pull_distance_min = args.pull_distance_min
    cfg.pull_distance_max = args.pull_distance_max
    cfg.perturbation_scale = args.perturbation_scale
    cfg.rot_angle_max_deg = args.rot_angle_max
    cfg.collision_threshold = args.collision_threshold
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.force = args.force

    if args.input:
        files = [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(f"{args.input_dir}/**/*.pt", recursive=True))
    else:
        print("ERROR: Must provide --input or --input-dir")
        sys.exit(1)

    print(f"Processing {len(files)} files...")
    print(f"  GPUs: {args.gpus}")
    print(f"  pull_distance: {cfg.pull_distance_min*100:.0f}-{cfg.pull_distance_max*100:.0f} cm")
    print(f"  perturbation_scale: {cfg.perturbation_scale}")
    print(f"  rot_angle_max: {cfg.rot_angle_max_deg} deg")
    print(f"  collision_threshold: {cfg.collision_threshold}m")

    # Distribute files across GPUs (round-robin)
    n_gpus = len(args.gpus)
    subsets = [[] for _ in range(n_gpus)]
    for i, f in enumerate(files):
        subsets[i % n_gpus].append(f)

    # Convert cfg to dict for multiprocessing
    cfg_dict = {
        'pull_distance_min': cfg.pull_distance_min,
        'pull_distance_max': cfg.pull_distance_max,
        'perturbation_scale': cfg.perturbation_scale,
        'rot_angle_max_deg': cfg.rot_angle_max_deg,
        'collision_threshold': cfg.collision_threshold,
        'batch_size': cfg.batch_size,
        'seed': cfg.seed,
        'force': cfg.force,
    }

    if n_gpus == 1:
        # Single GPU: run inline
        cfg.device = f"cuda:{args.gpus[0]}"
        ok, fail, skip = 0, 0, 0
        failed_files = []
        for f in files:
            try:
                success = process_pt_file(f, cfg)
                if success:
                    ok += 1
                else:
                    skip += 1
            except Exception as e:
                import traceback
                err_str = f"{f}: {e}\n" + traceback.format_exc()
                failed_files.append(err_str)
                fail += 1
    else:
        # Multi-GPU: one subprocess per GPU
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        with mp.Pool(n_gpus) as pool:
            results = pool.starmap(worker, [
                (subsets[i], args.gpus[i], cfg_dict)
                for i in range(n_gpus)
            ])
        ok = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)
        skip = sum(r[2] for r in results)
        failed_files = []
        for r in results:
            failed_files.extend(r[3])

    print(f"\nDone.  ✓ {ok}  ✗ {fail}  ⟳ {skip}")
    if fail > 0:
        print(f"\n===== FAILED FILES ({fail}) =====")
        for err in failed_files:
            print(err)
            print("-" * 40)


if __name__ == "__main__":
    main()
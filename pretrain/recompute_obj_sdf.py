"""recompute_obj_sdf.py — Recompute obj_pts_sdf with multiple methods and compare.

Methods:
  1. stored         — values from the .pt file (kaolin during gen)
  2. kaolin         — re-run kaolin point_to_mesh_distance
  3. trimesh        — trimesh.proximity (CPU, exact point-to-face)
  4. nn_verts       — brute-force NN to mesh vertices
  5. nn_cloud       — brute-force NN to 512 canonical tool pts

Usage:  python recompute_obj_sdf.py --data-dir tmp_data/
"""
import argparse, torch, trimesh, kaolin
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tmp_data")
    parser.add_argument("--cfg-idx", type=int, default=0)
    parser.add_argument("--n-pts", type=int, default=10)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.pt"))
    d = torch.load(str(files[0]), map_location="cpu", weights_only=False)
    ci = args.cfg_idx
    device = "cuda"

    # Load tool mesh
    mesh = trimesh.load(d["tool_mesh_path"], force="mesh", process=False)
    tool_verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    tool_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    print(f"Tool mesh: {d['tool_mesh_path']}")
    print(f"  {tool_verts.shape[0]} verts, {tool_faces.shape[0]} faces")

    # Reconstruct obj points in tool frame
    P_obj   = d["obj_pts_canonical"].to(device)
    R_obj   = d["object_rotation"].to(device)
    z_shift = d["obj_z_shift"].to(device)
    R_tool  = d["tool_rotations"][ci].to(device)
    t_tool  = d["tool_translations"][ci].to(device)

    p_world = P_obj @ R_obj.T
    p_world = p_world.clone()
    p_world[:, 2] -= z_shift
    obj_in_tool = (p_world - t_tool) @ R_tool   # (Q, 3)

    print(f"\nobj_in_tool range: [{obj_in_tool.min():.4f}, {obj_in_tool.max():.4f}]")
    print(f"obj_in_tool centroid: {obj_in_tool.mean(0).cpu().numpy()}")

    # --- Method 1: stored ---
    stored = d["obj_pts_sdf"][ci]  # (Q,) CPU

    # --- Method 2: kaolin point_to_mesh_distance ---
    pts = obj_in_tool.unsqueeze(0).contiguous()
    fv = kaolin.ops.mesh.index_vertices_by_faces(
        tool_verts.unsqueeze(0), tool_faces
    )
    print("Running kaolin point_to_mesh_distance...")
    sq_dist_kaolin, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts, fv)
    dist_kaolin = torch.sqrt(sq_dist_kaolin.clamp(min=1e-12)).squeeze(0).cpu()

    # --- Method 3: trimesh proximity (exact, CPU) ---
    print("Running trimesh proximity query...")
    query_pts = obj_in_tool.cpu().numpy()
    prox = trimesh.proximity.ProximityQuery(mesh)
    closest_pts, dist_trimesh_np, face_ids = prox.on_surface(query_pts)
    dist_trimesh = torch.tensor(dist_trimesh_np, dtype=torch.float32)

    # --- Method 4: NN to mesh vertices ---
    dists_nn = torch.cdist(obj_in_tool.unsqueeze(0), tool_verts.unsqueeze(0)).squeeze(0)
    dist_nn = dists_nn.min(dim=1).values.cpu()

    # --- Method 5: NN to 512 cloud pts ---
    P_tool_canon = d["tool_pts_canonical"].to(device)
    dists_cloud = torch.cdist(obj_in_tool.unsqueeze(0), P_tool_canon.unsqueeze(0)).squeeze(0)
    dist_cloud = dists_cloud.min(dim=1).values.cpu()

    # Summary
    print(f"\n{'Method':<25} {'mean':>10} {'max':>10} {'min':>10}")
    print(f"{'stored obj_sdf':<25} {stored.abs().mean():.5f} {stored.abs().max():.5f} {stored.abs().min():.5f}")
    print(f"{'kaolin recomputed':<25} {dist_kaolin.mean():.5f} {dist_kaolin.max():.5f} {dist_kaolin.min():.5f}")
    print(f"{'trimesh (exact)':<25} {dist_trimesh.mean():.5f} {dist_trimesh.max():.5f} {dist_trimesh.min():.5f}")
    print(f"{'NN to mesh verts':<25} {dist_nn.mean():.5f} {dist_nn.max():.5f} {dist_nn.min():.5f}")
    print(f"{'NN to 512 cloud pts':<25} {dist_cloud.mean():.5f} {dist_cloud.max():.5f} {dist_cloud.min():.5f}")

    # Per-point comparison
    order = stored.abs().argsort()
    n = args.n_pts
    print(f"\n{'idx':>5}  {'stored':>10}  {'kaolin':>10}  {'trimesh':>10}  {'nn_verts':>10}  {'nn_cloud':>10}")
    print("--- Closest ---")
    for i in range(n):
        pi = order[i].item()
        print(f"{pi:5d}  {stored[pi].item():+.6f}  {dist_kaolin[pi].item():.6f}  "
              f"{dist_trimesh[pi].item():.6f}  {dist_nn[pi].item():.6f}  {dist_cloud[pi].item():.6f}")
    print("--- Farthest ---")
    for i in range(n):
        pi = order[-(n-i)].item()
        print(f"{pi:5d}  {stored[pi].item():+.6f}  {dist_kaolin[pi].item():.6f}  "
              f"{dist_trimesh[pi].item():.6f}  {dist_nn[pi].item():.6f}  {dist_cloud[pi].item():.6f}")

    # Agreement checks
    diff_kaolin  = (stored.abs() - dist_kaolin).abs()
    diff_trimesh_nn = (dist_trimesh - dist_nn).abs()
    print(f"\n|stored| vs kaolin:   max_diff={diff_kaolin.max():.8f}")
    print(f"trimesh vs nn_verts:  max_diff={diff_trimesh_nn.max():.8f}  mean_diff={diff_trimesh_nn.mean():.8f}")
    print(f"trimesh vs kaolin:    max_diff={(dist_trimesh - dist_kaolin).abs().max():.8f}")

if __name__ == "__main__":
    main()

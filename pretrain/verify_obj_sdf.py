"""verify_obj_sdf.py — Cross-check stored obj_sdf against simple cloud-to-cloud distance.

Computes:
  1. obj→tool nearest-neighbor distance in WORLD frame (simple, no mesh)
  2. Stored obj_pts_sdf values
  3. Side-by-side comparison for each object point

If the stored obj_sdf is much smaller than the NN distance, something is wrong
in the coordinate transform or mesh query.

Usage:  python verify_obj_sdf.py --data-dir tmp_data/
"""
import argparse, torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tmp_data")
    parser.add_argument("--cfg-idx", type=int, default=0)
    parser.add_argument("--n-pts", type=int, default=15)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.pt"))
    if not files:
        print("No .pt files found"); return

    d = torch.load(str(files[0]), map_location="cpu", weights_only=False)
    ci = args.cfg_idx

    # Reconstruct world-frame clouds
    P_tool = d["tool_pts_canonical"]
    P_obj  = d["obj_pts_canonical"]
    R_tool = d["tool_rotations"][ci]
    t_tool = d["tool_translations"][ci]
    R_obj  = d["object_rotation"]
    z_shift = d["obj_z_shift"]

    tool_world = P_tool @ R_tool.T + t_tool
    obj_world  = P_obj @ R_obj.T
    obj_world  = obj_world.clone()
    obj_world[:, 2] -= z_shift

    # Stored SDF
    obj_sdf_stored = d["obj_pts_sdf"][ci]   # (Q,)

    # Simple NN distance: for each obj point, distance to nearest tool point (WORLD frame)
    # This is cloud-to-cloud, not mesh, so it's an upper bound on true SDF
    dists = torch.cdist(obj_world.unsqueeze(0), tool_world.unsqueeze(0)).squeeze(0)  # (Q, P)
    nn_dist_world = dists.min(dim=1).values  # (Q,)

    # Also compute: transform obj points to tool canonical and check NN to canonical tool cloud
    R_mat = R_tool  # (3,3) — tool rotation
    obj_in_tool_frame = (obj_world - t_tool) @ R_mat  # world → tool canonical
    dists_canon = torch.cdist(obj_in_tool_frame.unsqueeze(0), P_tool.unsqueeze(0)).squeeze(0)
    nn_dist_canon = dists_canon.min(dim=1).values  # (Q,)

    # Print comparison sorted by stored SDF
    order = obj_sdf_stored.argsort()

    print(f"File: {files[0].name}  config: {ci}")
    print(f"Tool world centroid: {tool_world.mean(0).numpy()}")
    print(f"Obj  world centroid: {obj_world.mean(0).numpy()}")
    print()
    print(f"{'idx':>5}  {'obj_world xyz':>30}  {'obj_in_tool_canon':>30}  "
          f"{'stored_sdf':>10}  {'nn_world':>10}  {'nn_canon':>10}  {'ratio':>6}")

    n = args.n_pts

    print(f"\n--- Closest {n} (by stored SDF) ---")
    for i in range(n):
        pi = order[i].item()
        w = obj_world[pi].numpy()
        c = obj_in_tool_frame[pi].numpy()
        s = obj_sdf_stored[pi].item()
        nw = nn_dist_world[pi].item()
        nc = nn_dist_canon[pi].item()
        r = s / nw if nw > 1e-6 else float('nan')
        print(f"{pi:5d}  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  "
              f"({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  "
              f"{s:+.5f}  {nw:+.5f}  {nc:+.5f}  {r:.2f}")

    print(f"\n--- Farthest {n} (by stored SDF) ---")
    for i in range(n):
        pi = order[-(n-i)].item()
        w = obj_world[pi].numpy()
        c = obj_in_tool_frame[pi].numpy()
        s = obj_sdf_stored[pi].item()
        nw = nn_dist_world[pi].item()
        nc = nn_dist_canon[pi].item()
        r = s / nw if nw > 1e-6 else float('nan')
        print(f"{pi:5d}  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  "
              f"({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  "
              f"{s:+.5f}  {nw:+.5f}  {nc:+.5f}  {r:.2f}")

    print(f"\n--- Global stats ---")
    print(f"  stored obj_sdf:  mean={obj_sdf_stored.mean():.5f}  max={obj_sdf_stored.max():.5f}")
    print(f"  nn_world:        mean={nn_dist_world.mean():.5f}  max={nn_dist_world.max():.5f}")
    print(f"  nn_canon:        mean={nn_dist_canon.mean():.5f}  max={nn_dist_canon.max():.5f}")
    print(f"  ratio stored/nn_world:  mean={( obj_sdf_stored / nn_dist_world.clamp(min=1e-6)).mean():.3f}")
    print(f"  ratio stored/nn_canon:  mean={(obj_sdf_stored / nn_dist_canon.clamp(min=1e-6)).mean():.3f}")

if __name__ == "__main__":
    main()

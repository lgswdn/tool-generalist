"""dump_sdf_samples.py — Print raw SDF values for a few points from one .pt file.

Usage:  python dump_sdf_samples.py --data-dir tmp_data/
"""
import argparse, torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tmp_data")
    parser.add_argument("--cfg-idx", type=int, default=0, help="config index to inspect")
    parser.add_argument("--n-pts", type=int, default=10, help="number of points to print")
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.pt"))
    if not files:
        print("No .pt files found"); return

    path = files[0]
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    ci = args.cfg_idx
    n  = args.n_pts

    P_tool = d["tool_pts_canonical"]          # (P, 3)
    P_obj  = d["obj_pts_canonical"]           # (Q, 3)
    R_tool = d["tool_rotations"][ci]          # (3,3)
    t_tool = d["tool_translations"][ci]       # (3,)
    R_obj  = d["object_rotation"]             # (3,3)
    z_shift = d["obj_z_shift"]

    tool_world = P_tool @ R_tool.T + t_tool   # (P, 3)
    obj_world  = P_obj @ R_obj.T
    obj_world  = obj_world.clone()
    obj_world[:, 2] -= z_shift

    tool_sdf = d["tool_pts_sdf"][ci]          # (P,)
    obj_sdf  = d["obj_pts_sdf"][ci]           # (Q,)

    print(f"File: {path.name}  config: {ci}")
    print(f"Tool canonical range: {P_tool.min():.3f} to {P_tool.max():.3f}")
    print(f"Obj  canonical range: {P_obj.min():.3f} to {P_obj.max():.3f}")
    print(f"Tool world centroid:  {tool_world.mean(0).numpy()}")
    print(f"Obj  world centroid:  {obj_world.mean(0).numpy()}")
    print()

    # Sort tool points by SDF (show closest to object first)
    tool_order = tool_sdf.argsort()
    print(f"{'=== TOOL points (sorted by SDF, closest to obj first) ===':^70}")
    print(f"{'idx':>5}  {'canonical xyz':>30}  {'world xyz':>30}  {'SDF':>8}")
    for i in range(n):
        pi = tool_order[i].item()
        c = P_tool[pi].numpy()
        w = tool_world[pi].numpy()
        s = tool_sdf[pi].item()
        print(f"{pi:5d}  ({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  {s:+.5f}")
    print("  ...")
    for i in range(n):
        pi = tool_order[-(n-i)].item()
        c = P_tool[pi].numpy()
        w = tool_world[pi].numpy()
        s = tool_sdf[pi].item()
        print(f"{pi:5d}  ({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  {s:+.5f}")

    print()
    # Sort obj points by SDF (show closest to tool first)
    obj_order = obj_sdf.argsort()
    print(f"{'=== OBJ points (sorted by SDF, closest to tool first) ===':^70}")
    print(f"{'idx':>5}  {'canonical xyz':>30}  {'world xyz':>30}  {'SDF':>8}")
    for i in range(n):
        pi = obj_order[i].item()
        c = P_obj[pi].numpy()
        w = obj_world[pi].numpy()
        s = obj_sdf[pi].item()
        print(f"{pi:5d}  ({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  {s:+.5f}")
    print("  ...")
    for i in range(n):
        pi = obj_order[-(n-i)].item()
        c = P_obj[pi].numpy()
        w = obj_world[pi].numpy()
        s = obj_sdf[pi].item()
        print(f"{pi:5d}  ({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  ({w[0]:+.4f},{w[1]:+.4f},{w[2]:+.4f})  {s:+.5f}")

    # Also compute pairwise distance between centroids
    d_centroid = (tool_world.mean(0) - obj_world.mean(0)).norm().item()
    print(f"\nCentroid distance (tool↔obj): {d_centroid:.4f}")

if __name__ == "__main__":
    main()

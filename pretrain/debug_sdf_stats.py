"""debug_sdf_stats.py — Inspect GT SDF distributions and model predictions.

Run with:
    python debug_sdf_stats.py --data-dir tmp_data/

Prints per-field statistics (mean, std, min, max, %positive, %negative, %near-zero)
for both tool_pts_sdf and obj_pts_sdf across all loaded .pt files and configs.
Also checks for common bugs (NaN, Inf, all-zero fields).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def stats(t: torch.Tensor, label: str, near_zero_thresh: float = 0.01):
    t = t.float().flatten()
    n = t.numel()
    nan_cnt = t.isnan().sum().item()
    inf_cnt = t.isinf().sum().item()
    t_clean = t[~(t.isnan() | t.isinf())]

    if t_clean.numel() == 0:
        print(f"  {label}: ALL NaN/Inf!")
        return

    mean  = t_clean.mean().item()
    std   = t_clean.std().item()
    mn    = t_clean.min().item()
    mx    = t_clean.max().item()
    pct_pos  = (t_clean > 0).float().mean().item() * 100
    pct_neg  = (t_clean < 0).float().mean().item() * 100
    pct_zero = (t_clean.abs() < near_zero_thresh).float().mean().item() * 100

    print(f"  {label}:")
    print(f"    n={n}  NaN={nan_cnt}  Inf={inf_cnt}")
    print(f"    mean={mean:+.5f}  std={std:.5f}  min={mn:+.5f}  max={mx:+.5f}")
    print(f"    %pos={pct_pos:.1f}%  %neg={pct_neg:.1f}%  |val|<{near_zero_thresh}: {pct_zero:.1f}%")
    if t_clean.abs().max().item() == 0:
        print(f"    *** WARNING: all values are exactly zero! Possible bug. ***")


def inspect_pt_file(path: str, max_configs: int = 20):
    data = torch.load(path, map_location="cpu", weights_only=False)
    n_configs = data["tool_translations"].shape[0]
    print(f"\n{'='*60}")
    print(f"File: {Path(path).name}  ({n_configs} configs)")
    print(f"{'='*60}")

    # ---- Keys present -------------------------------------------------------
    print(f"  Keys: {sorted(data.keys())}")

    # ---- Canonical clouds ---------------------------------------------------
    P_tool = data.get("tool_pts_canonical")
    P_obj  = data.get("obj_pts_canonical")
    if P_tool is not None:
        print(f"\n  tool_pts_canonical: {tuple(P_tool.shape)}"
              f"  range=[{P_tool.min():.4f}, {P_tool.max():.4f}]")
    if P_obj is not None:
        print(f"  obj_pts_canonical : {tuple(P_obj.shape)}"
              f"  range=[{P_obj.min():.4f}, {P_obj.max():.4f}]")

    # ---- SDF arrays (all configs) ------------------------------------------
    n_use = min(n_configs, max_configs)
    print(f"\n  Inspecting first {n_use}/{n_configs} configs:")

    if "tool_pts_sdf" in data:
        t_sdf = data["tool_pts_sdf"][:n_use]   # (n_use, P)
        stats(t_sdf, "tool_pts_sdf")
    else:
        print("  *** tool_pts_sdf NOT FOUND in file! ***")

    if "obj_pts_sdf" in data:
        o_sdf = data["obj_pts_sdf"][:n_use]    # (n_use, Q)
        stats(o_sdf, "obj_pts_sdf")
    else:
        print("  *** obj_pts_sdf NOT FOUND in file! ***")

    # ---- Cross-check: world-frame tool cloud --------------------------------
    if P_tool is not None and "tool_rotations" in data and "tool_translations" in data:
        R = data["tool_rotations"][0]    # (3, 3)
        t = data["tool_translations"][0] # (3,)
        tool_world = P_tool @ R.T + t   # (P, 3)
        print(f"\n  tool_pc world-frame (config 0):"
              f"  range=[{tool_world.min():.4f}, {tool_world.max():.4f}]"
              f"  centroid={tool_world.mean(0).numpy()}")

    if P_obj is not None and "object_rotation" in data:
        R_obj   = data["object_rotation"]      # (3, 3)
        z_shift = data.get("obj_z_shift", torch.tensor(0.0))
        obj_w   = P_obj @ R_obj.T
        obj_w   = obj_w.clone()
        obj_w[:, 2] -= z_shift
        print(f"  obj_pc  world-frame (shared  ):"
              f"  range=[{obj_w.min():.4f}, {obj_w.max():.4f}]"
              f"  centroid={obj_w.mean(0).numpy()}")


def inspect_all(data_dir: str, max_files: int = 5, max_configs: int = 20):
    files = sorted(Path(data_dir).rglob("*.pt"))
    if not files:
        print(f"No .pt files found under {data_dir}")
        return

    print(f"Found {len(files)} .pt files; inspecting first {min(len(files), max_files)}.")

    # Aggregate across all inspected files
    all_tool_sdf, all_obj_sdf = [], []

    for path in files[:max_files]:
        inspect_pt_file(str(path), max_configs=max_configs)
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if "tool_pts_sdf" in data:
            all_tool_sdf.append(data["tool_pts_sdf"].float().flatten())
        if "obj_pts_sdf" in data:
            all_obj_sdf.append(data["obj_pts_sdf"].float().flatten())

    # ---- Global aggregate ---------------------------------------------------
    print(f"\n{'='*60}")
    print("GLOBAL AGGREGATED STATS (across all inspected files × configs)")
    print(f"{'='*60}")
    if all_tool_sdf:
        stats(torch.cat(all_tool_sdf), "tool_pts_sdf  [GLOBAL]")
    if all_obj_sdf:
        stats(torch.cat(all_obj_sdf),  "obj_pts_sdf   [GLOBAL]")

    # ---- Huber loss estimate from raw GT values (lower-bound, pred=0) ------
    print("\n  Lower-bound Huber loss if model predicts zero:")
    if all_tool_sdf:
        t = torch.cat(all_tool_sdf)
        huber = torch.nn.functional.smooth_l1_loss(torch.zeros_like(t), t).item()
        print(f"    tool  zero-pred Huber = {huber:.6f}")
    if all_obj_sdf:
        o = torch.cat(all_obj_sdf)
        huber = torch.nn.functional.smooth_l1_loss(torch.zeros_like(o), o).item()
        print(f"    obj   zero-pred Huber = {huber:.6f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="tmp_data",
                        help="Root dir containing .pt files")
    parser.add_argument("--max-files",   type=int, default=5,
                        help="Max .pt files to inspect")
    parser.add_argument("--max-configs", type=int, default=20,
                        help="Max configs per file to include in stats")
    args = parser.parse_args()

    inspect_all(args.data_dir, args.max_files, args.max_configs)


if __name__ == "__main__":
    main()

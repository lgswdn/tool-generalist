"""validate_architecture.py — Sanity checks for centered input + 6D pose architecture.

Checks:
  1. Centering:     tool_canonical.mean ≈ 0, obj_pc.mean ≈ 0
  2. World frame:   obj_centroid.z > 0 (object grounded above floor)
  3. Contact geom:  tool at contact pose is near object surface (chamfer ≈ 5mm)
  4. SDF sign:      tool_sdf mostly positive at contact (outside object)
  5. Noising:       noised tool centroid == noised_t (exact by construction)
  6. Target:        applying target brings tool closer to contact pose

Run with:
    python validate_architecture.py --data-dir ../tmp_data --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import NewPretrainDataset, collect_pt_files
from noise_utils import sample_noised_poses_batch


def chamfer_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric chamfer distance between (P,3) and (Q,3) point clouds. Returns metres."""
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=2)[0]  # (P, Q)
    return ((d.min(dim=1).values.mean() + d.min(dim=0).values.mean()) / 2).item()


def check(name: str, ok: bool, detail: str = ""):
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def validate_item(item: dict, ds: NewPretrainDataset, device: str, verbose: bool = True):
    tc = item["tool_canonical"].to(device)        # (P, 3)
    obj = item["obj_pc"].to(device)               # (Q, 3)
    obj_cen = item["obj_centroid"].to(device)     # (3,)
    R = item["contact_R"].to(device)              # (3, 3)
    t = item["contact_t"].to(device)              # (3,)
    tool_sdf = item["tool_sdf"].to(device)        # (P,)
    obj_sdf = item["obj_sdf"].to(device)          # (Q,)

    results = []

    # ── 1. Centering ─────────────────────────────────────────────────────────
    tool_mean = tc.mean(0).abs().max().item()
    obj_mean  = obj.mean(0).abs().max().item()
    results.append(check("tool_canonical centered", tool_mean < 1e-3,
                         f"max_abs_mean={tool_mean:.6f}"))
    results.append(check("obj_pc centered", obj_mean < 1e-3,
                         f"max_abs_mean={obj_mean:.6f}"))

    # ── 2. obj_centroid is above ground ──────────────────────────────────────
    results.append(check("obj_centroid.z > 0 (world frame, grounded)",
                         obj_cen[2].item() > 0,
                         f"z={obj_cen[2].item():.4f}m"))

    # ── 3. Contact geometry: tool world vs object world ───────────────────────
    # tool in world at contact: R @ canonical + t
    tool_world = (tc @ R.T) + t.unsqueeze(0)       # (P, 3)
    obj_world  = obj + obj_cen.unsqueeze(0)         # (Q, 3)
    cd = chamfer_dist(tool_world, obj_world)
    results.append(check("contact chamfer ≤ 15mm (tool near object)",
                         cd < 0.015,
                         f"chamfer={cd*1000:.2f}mm"))

    # ── 4. SDF sign at contact ────────────────────────────────────────────────
    pct_pos = (tool_sdf > 0).float().mean().item() * 100
    results.append(check("tool_sdf mostly positive at contact (outside obj)",
                         pct_pos > 70.0,
                         f"{pct_pos:.1f}% positive"))
    pct_pos_obj = (obj_sdf > 0).float().mean().item() * 100
    results.append(check("obj_sdf mostly positive at contact (outside tool)",
                         pct_pos_obj > 70.0,
                         f"{pct_pos_obj:.1f}% positive"))

    # ── 5. Noising: noised tool centroid equals noised_t ─────────────────────
    tc_b = tc.unsqueeze(0)
    obj_b = obj.unsqueeze(0)
    obj_cen_b = obj_cen.unsqueeze(0)
    R_b = R.unsqueeze(0)
    t_b = t.unsqueeze(0)

    noise_out = sample_noised_poses_batch(
        contact_R=R_b, contact_t=t_b,
        num_steps=10, max_trans=0.1, max_rot_deg=40.0,
        interp=True, precise_prob=False,
        tool_canonical=tc_b, obj_pc=obj_b, obj_centroid=obj_cen_b,
    )
    nR = noise_out["noised_R"][0]   # (3,3)
    nt = noise_out["noised_t"][0]   # (3,)

    # Tool centroid in world at noised pose = R @ 0 + t = t  (since canonical is centered)
    noised_world = (tc @ nR.T) + nt.unsqueeze(0)
    centroid_check = (noised_world.mean(0) - nt).abs().max().item()
    results.append(check("noised tool centroid == noised_t",
                         centroid_check < 1e-4,
                         f"max_err={centroid_check:.2e}m"))

    # ── 6. Target brings tool closer to contact ──────────────────────────────
    tgt_R = noise_out["target_rot_mat"][0]  # (3,3)
    tgt_t = noise_out["target_trans"][0]    # (3,)

    # Noised chamfer vs. contact
    noised_cd = chamfer_dist(noised_world, obj_world)

    # After applying target: prev_R = tgt_R @ nR, prev_t = nt + tgt_t
    prev_R = tgt_R @ nR
    prev_t = nt + tgt_t
    prev_world = (tc @ prev_R.T) + prev_t.unsqueeze(0)
    prev_cd = chamfer_dist(prev_world, obj_world)

    t_idx = noise_out["t_idx"][0].item()
    if t_idx > 0:
        results.append(check("target step reduces chamfer to contact",
                             prev_cd < noised_cd,
                             f"noised={noised_cd*1000:.2f}mm → after_target={prev_cd*1000:.2f}mm"))
    else:
        results.append(check("t_idx=0 (contact pose, no step needed)", True,
                             "skipping target test"))

    # ── 7. Mesh cache loaded ──────────────────────────────────────────────────
    pt_path = item["pt_path"]
    mc = ds._mesh_cache.get(pt_path, {})
    results.append(check("tool mesh loaded in cache",
                         mc.get("tool_verts") is not None,
                         pt_path))
    results.append(check("object mesh loaded in cache",
                         mc.get("obj_verts") is not None,
                         pt_path))

    return all(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-items", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    print(f"\n{'='*60}")
    print(f"  Architecture Validation")
    print(f"  data_dir : {args.data_dir}")
    print(f"  device   : {device}")
    print(f"{'='*60}\n")

    files = collect_pt_files(args.data_dir)
    if not files:
        print(f"ERROR: No .pt files found in {args.data_dir}")
        sys.exit(1)

    ds = NewPretrainDataset(files[:max(5, args.num_items)], augment=False)
    print(f"Dataset: {len(ds)} items from {len(files)} files\n")

    n_pass = 0
    indices = torch.randperm(len(ds))[:args.num_items].tolist()
    for i, idx in enumerate(indices):
        item = ds[idx]
        pt = Path(item["pt_path"]).name
        print(f"── Item {i+1}/{args.num_items}  [{pt}  cfg={ds._index[idx][1]}]")
        ok = validate_item(item, ds, device)
        n_pass += int(ok)
        print()

    print(f"{'='*60}")
    print(f"  Result: {n_pass}/{args.num_items} items passed all checks")
    print(f"{'='*60}\n")
    sys.exit(0 if n_pass == args.num_items else 1)


if __name__ == "__main__":
    main()

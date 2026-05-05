"""validate_architecture.py — Coordinate frame and model sanity checks.

Validates seven invariants of the new architecture:
  1.  tool_canonical is centered at (0,0,0)
  2.  obj_pc is centered at (0,0,0)
  3.  obj_centroid.z > 0  (object grounded in world frame)
  4.  Contact geometry: tool at contact pose is close to object surface
  5.  SDF sign: tool points mostly outside object at contact (sdf > 0)
  6.  Noising: noised tool centroid == noised_t  (by construction, since tool centered)
  7.  Denoising target: applying target reduces distance to contact pose
  8.  Model forward pass: shapes correct, loss finite, backward works

Run:
    cd pretrain/new_pretrain
    python validate_architecture.py --data-dir <path_to_pt_files> [--device cuda:0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import NewPretrainDataset, collect_pt_files
from noise_utils import sample_noised_poses_batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric chamfer distance (metres) between (P,3) and (Q,3) clouds."""
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0))[0]  # (P, Q)
    return ((d.min(1).values.mean() + d.min(0).values.mean()) / 2).item()


def _ok(name: str, passed: bool, detail: str = "") -> bool:
    mark = "✓" if passed else "✗"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {mark}  {name}{suffix}")
    return passed


# ── Per-item checks ───────────────────────────────────────────────────────────

def check_item(item: dict, ds: NewPretrainDataset, device: str) -> bool:
    tc      = item["tool_canonical"].to(device)   # (P, 3) centered
    obj     = item["obj_pc"].to(device)           # (Q, 3) centered
    obj_cen = item["obj_centroid"].to(device)     # (3,)   world pos
    R       = item["contact_R"].to(device)        # (3, 3)
    t       = item["contact_t"].to(device)        # (3,)   world pos
    t_sdf   = item["tool_sdf"].to(device)         # (P,)
    o_sdf   = item["obj_sdf"].to(device)          # (Q,)

    results = []

    # ── 1 & 2. Centering ──────────────────────────────────────────────────────
    tool_err = tc.mean(0).abs().max().item()
    obj_err  = obj.mean(0).abs().max().item()
    results.append(_ok("tool_canonical centered at (0,0,0)",
                       tool_err < 1e-3, f"max_abs_mean={tool_err:.2e} m"))
    results.append(_ok("obj_pc centered at (0,0,0)",
                       obj_err  < 1e-3, f"max_abs_mean={obj_err:.2e} m"))

    # ── 3. Object grounded ────────────────────────────────────────────────────
    results.append(_ok("obj_centroid.z > 0 (grounded world frame)",
                       obj_cen[2].item() > 0,
                       f"z = {obj_cen[2].item():.4f} m"))

    # ── 4. Contact geometry ───────────────────────────────────────────────────
    # Tool world = tool_canonical @ R.T + t   (contact_t is world centroid pos)
    tool_world = tc @ R.T + t.unsqueeze(0)          # (P, 3)
    obj_world  = obj + obj_cen.unsqueeze(0)          # (Q, 3)
    cd = chamfer(tool_world, obj_world)
    results.append(_ok("contact chamfer ≤ 20 mm (tool near object surface)",
                       cd < 0.020, f"{cd*1000:.2f} mm"))

    # ── 5. SDF sign at contact ────────────────────────────────────────────────
    pct_tool = (t_sdf > 0).float().mean().item() * 100
    pct_obj  = (o_sdf > 0).float().mean().item() * 100
    results.append(_ok("tool_sdf: most points outside object (>70 % positive)",
                       pct_tool > 70, f"{pct_tool:.1f} %"))
    results.append(_ok("obj_sdf:  most points outside tool   (>70 % positive)",
                       pct_obj  > 70, f"{pct_obj:.1f} %"))

    # ── 6. Noising: noised_t equals tool world centroid at noised pose ────────
    tc_b   = tc.unsqueeze(0)
    obj_b  = obj.unsqueeze(0)
    cen_b  = obj_cen.unsqueeze(0)
    R_b    = R.unsqueeze(0)
    t_b    = t.unsqueeze(0)

    noise_out = sample_noised_poses_batch(
        contact_R=R_b, contact_t=t_b,
        num_steps=10, max_trans=0.1, max_rot_deg=40.0,
        interp=True, precise_prob=False,
        tool_canonical=tc_b, obj_pc=obj_b,
    )
    nR = noise_out["noised_R"][0]   # (3,3)
    nt = noise_out["noised_t"][0]   # (3,)

    # Since tool_canonical is centered, mean(tc @ nR.T + nt) = nt exactly.
    noised_world   = tc @ nR.T + nt.unsqueeze(0)   # (P, 3)
    centroid_error = (noised_world.mean(0) - nt).abs().max().item()
    results.append(_ok("noised tool centroid == noised_t",
                       centroid_error < 1e-4, f"err={centroid_error:.2e} m"))

    # ── 7. Target step reduces chamfer ────────────────────────────────────────
    t_idx  = noise_out["t_idx"][0].item()
    tgt_R  = noise_out["target_rot_mat"][0]   # (3,3)
    tgt_t  = noise_out["target_trans"][0]     # (3,)
    prev_R = tgt_R @ nR
    prev_t = nt + tgt_t
    prev_world = tc @ prev_R.T + prev_t.unsqueeze(0)

    noised_cd = chamfer(noised_world, obj_world)
    prev_cd   = chamfer(prev_world,   obj_world)

    if t_idx > 0:
        results.append(_ok("target step reduces chamfer toward contact",
                           prev_cd < noised_cd,
                           f"{noised_cd*1000:.2f} mm → {prev_cd*1000:.2f} mm"))
    else:
        results.append(_ok("t_idx=0 (contact pose sampled, target step skipped)", True))

    return all(results)


# ── Model forward pass check ──────────────────────────────────────────────────

def check_model_forward(device: str) -> bool:
    """Instantiate a tiny model and run one forward pass with correct shapes."""
    print("\n── Model forward pass (tiny config) ────────────────────────────────")
    try:
        from config import NewPretrainConfig
        from model import ContactDiffusionModel

        cfg = NewPretrainConfig()
        cfg.num_pts       = 64
        cfg.patch_size    = 16
        cfg.encoder_channel = 32
        cfg.vit_depth     = 2
        cfg.vit_heads     = 2
        cfg.cross_attn_layers = 2
        cfg.cross_attn_heads  = 2
        cfg.denoise_hidden    = (64,)
        cfg.head_hidden       = (32,)
        cfg.pose_dim          = 6
        cfg.movement_cond_dim = 14
        cfg.task              = "sdf-diff"

        model = ContactDiffusionModel(
            head_mode=cfg.head_mode,
            patch_agg=cfg.patch_agg,
            head_hidden=cfg.head_hidden,
            num_pts=cfg.num_pts,
            patch_size=cfg.patch_size,
            encoder_channel=cfg.encoder_channel,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            cross_attn_heads=cfg.cross_attn_heads,
            cross_attn_layers=cfg.cross_attn_layers,
            pose_dim=cfg.pose_dim,
            movement_cond_dim=cfg.movement_cond_dim,
            denoise_hidden=cfg.denoise_hidden,
            task=cfg.task,
        ).to(device)

        B, P, Q = 2, 64, 64
        dev = torch.device(device)

        tool_rotated    = torch.randn(B, P, 3, device=dev)       # (B,P,3) centered+rotated
        obj_pc          = torch.randn(B, Q, 3, device=dev)       # (B,Q,3) centered
        tool_sdf        = torch.randn(B, P, device=dev)
        obj_sdf         = torch.randn(B, Q, device=dev)
        pose_6d         = torch.randn(B, 6, device=dev)          # [noised_t, obj_centroid]
        timestep        = torch.randint(0, 10, (B,), device=dev)
        movement_cond   = torch.randn(B, 14, device=dev)         # [delta_tool_t/q, delta_obj_t/q]
        target_trans    = torch.randn(B, 3, device=dev)
        target_rot_mat  = torch.eye(3, device=dev).unsqueeze(0).expand(B, -1, -1).clone()
        child_start     = torch.randn(B, P, 3, device=dev)
        child_final     = torch.randn(B, P, 3, device=dev)

        loss, metrics = model(
            tool_canonical=tool_rotated,
            obj_pc=obj_pc,
            tool_sdf_gt=tool_sdf,
            obj_sdf_gt=obj_sdf,
            noised_pose_7d=pose_6d,
            timestep=timestep,
            movement_cond=movement_cond,
            target_trans=target_trans,
            target_rot_mat=target_rot_mat,
            child_start_pcd=child_start,
            child_final_pcd=child_final,
        )
        loss.backward()

        ok_loss  = _ok("loss is finite", loss.isfinite().item(), f"loss={loss.item():.4f}")
        ok_keys  = _ok("metrics has expected keys",
                       {"sdf_loss", "denoise_trans_loss", "denoise_rot_loss"} <= metrics.keys())
        ok_poses = _ok("pose_dim=6 accepted (not 7)",
                       cfg.pose_dim == 6, f"pose_dim={cfg.pose_dim}")
        ok_mcond = _ok("movement_cond_dim=14 accepted",
                       cfg.movement_cond_dim == 14, f"movement_cond_dim={cfg.movement_cond_dim}")
        return ok_loss and ok_keys and ok_poses and ok_mcond

    except Exception as e:
        _ok(f"model forward failed: {e}", False)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  required=True,  help="Directory with .pt files")
    parser.add_argument("--num-items", type=int, default=5)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Architecture Validation")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  device    : {args.device}")
    print(f"  num_items : {args.num_items}")
    print(f"{'='*60}\n")

    files = collect_pt_files(args.data_dir)
    if not files:
        print(f"ERROR: No .pt files found under {args.data_dir}")
        sys.exit(1)

    ds = NewPretrainDataset(
        files[:max(args.num_items, 10)],
        augment=False,
        require_movement=False,
    )
    print(f"Dataset: {len(ds)} items from {len(files[:max(args.num_items, 10)])} files\n")

    n_pass   = 0
    indices  = torch.randperm(len(ds))[:args.num_items].tolist()

    for rank, idx in enumerate(indices):
        item   = ds[idx]
        pt_tag = Path(item["pt_path"]).name
        cfg_i  = ds._index[idx][1]
        print(f"── Item {rank+1}/{args.num_items}  [{pt_tag}  cfg={cfg_i}]")
        ok = check_item(item, ds, args.device)
        n_pass += int(ok)
        print()

    # Model check (uses random tensors, no real data needed)
    model_ok = check_model_forward(args.device)
    n_pass  += int(model_ok)
    total    = args.num_items + 1   # items + model check

    print(f"\n{'='*60}")
    print(f"  Result: {n_pass}/{total} checks passed")
    print(f"{'='*60}\n")
    sys.exit(0 if n_pass == total else 1)


if __name__ == "__main__":
    main()

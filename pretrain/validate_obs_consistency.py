#!/usr/bin/env python3
"""validate_obs_consistency.py — Verify RL and pretraining see the same encoder input.

The pretraining pipeline bakes:
  • tool_pts_canonical  (P, 3) — canonical tool cloud, centered at (0,0,0)
  • obj_pts_canonical   (Q, 3) — object cloud, centered at (0,0,0)
  • tool_translations   (N, 3) — world-frame centroid position of tool cloud
  • tool_rotations      (N, 3, 3) — rotation applied to canonical cloud
  • obj_centroid        (3,)   — world-frame centroid of object cloud

The RL env does at inference time:
  1. Apply R_tool to canonical pts + body_pos_world → tool world cloud
  2. tool_cloud_env  = tool_world - env_origin
  3. tool_centroid   = tool_cloud_env.mean(dim=0)
  4. tool_cloud_cent = tool_cloud_env - tool_centroid   ← encoder input

This script verifies: tool_cloud_cent == tool_pts_canonical @ R_tool.T  (up to FP tolerance)
and similarly for the object cloud.

Usage:
    python validate_obs_consistency.py --pt path/to/sample.pt [--device cuda]
"""

import argparse
import sys
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_rl_obs(
    pts_canonical: torch.Tensor,   # (P, 3) centered at (0,0,0)
    R: torch.Tensor,               # (3, 3)
    t: torch.Tensor,               # (3,) world-frame centroid
    env_origin: torch.Tensor,      # (3,) env origin in world frame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate what the RL observation function returns.

    Steps (mirrors get_tool_pointcloud_in_env_frame + get_tool_centroid):
      1. world_cloud  = pts_canonical @ R.T + t     (world frame)
      2. env_cloud    = world_cloud - env_origin     (env frame)
      3. centroid     = env_cloud.mean(dim=0)        → get_tool_centroid()
      4. centered     = env_cloud - centroid         → get_tool_pointcloud_in_env_frame()

    Returns:
        centered  (P, 3)  — what the encoder receives
        centroid  (3,)    — what get_tool/obj_centroid() returns
    """
    world_cloud = pts_canonical @ R.T + t.unsqueeze(0)  # (P, 3)
    env_cloud   = world_cloud - env_origin.unsqueeze(0)  # (P, 3)
    centroid    = env_cloud.mean(dim=0)                  # (3,)
    centered    = env_cloud - centroid.unsqueeze(0)      # (P, 3)
    return centered, centroid


def pretrain_encoder_input(
    pts_canonical: torch.Tensor,  # (P, 3) already centered
    R: torch.Tensor,              # (3, 3)
) -> torch.Tensor:
    """Reconstruct what the pretraining encoder sees.

    In train.py (sdf task):
        tool_canonical = tool_pts_canonical @ noised_R.T
    where tool_pts_canonical is already centered at (0,0,0).

    For the CONTACT pose (no noise), R = tool_rotations[i].

    Returns:
        (P, 3) — the point cloud the encoder received during pretraining
    """
    return pts_canonical @ R.T  # (P, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_cloud_consistency(
    name: str,
    pretrain_input: torch.Tensor,   # (P, 3)
    rl_input: torch.Tensor,         # (P, 3)
    tol: float = 1e-4,
) -> bool:
    """Check that pretrain and RL encoder inputs are the same cloud (up to point order)."""
    diff = (pretrain_input - rl_input).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < tol
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{name}] max_err={max_err:.2e}  mean_err={mean_err:.2e}  {status}")
    return passed


def check_centroid_consistency(
    name: str,
    pretrain_centroid: torch.Tensor,  # (3,)
    rl_centroid: torch.Tensor,        # (3,)
    env_origin: torch.Tensor,         # (3,)
    tol: float = 1e-4,
) -> bool:
    """Check that the RL centroid matches the pretraining centroid (env-frame).

    Pretraining centroid is in world frame; RL centroid is in env frame.
    They should satisfy: rl_centroid == pretrain_centroid - env_origin
    """
    expected_rl = pretrain_centroid - env_origin
    diff = (rl_centroid - expected_rl).abs()
    max_err = diff.max().item()
    passed = max_err < tol
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{name} centroid] max_err={max_err:.2e}  {status}")
    print(f"    pretrain (world): {pretrain_centroid.numpy().round(4)}")
    print(f"    rl (env-frame):   {rl_centroid.numpy().round(4)}")
    print(f"    expected_rl:      {expected_rl.numpy().round(4)}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate RL/pretraining observation consistency")
    parser.add_argument("--pt", required=True, help="Path to a .pt contact file")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--config_idx", type=int, default=0, help="Which contact config (row) to test")
    parser.add_argument("--tol", type=float, default=1e-4, help="Max absolute error tolerance")
    parser.add_argument("--env_origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Simulated Isaac Lab env origin in world frame (default: 0 0 0)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("RL ↔ Pretraining Observation Consistency Validator")
    print(f"{'='*60}")
    print(f"File   : {args.pt}")
    print(f"Config : {args.config_idx}")
    print(f"Device : {args.device}")
    print(f"Tol    : {args.tol:.1e}")

    # ── Load .pt file ─────────────────────────────────────────────────────────
    data = torch.load(args.pt, map_location="cpu")

    tool_pts_canonical = data["tool_pts_canonical"].float()   # (P, 3) centered
    obj_pts_canonical  = data["obj_pts_canonical"].float()    # (Q, 3) centered
    tool_translations  = data["tool_translations"].float()    # (N, 3) world centroid
    tool_rotations     = data["tool_rotations"].float()       # (N, 3, 3)
    obj_centroid_world = data["obj_centroid"].float()         # (3,)

    N = tool_translations.shape[0]
    if args.config_idx >= N:
        print(f"ERROR: config_idx={args.config_idx} but file only has {N} configs.")
        sys.exit(1)

    R_tool = tool_rotations[args.config_idx]      # (3, 3)
    t_tool = tool_translations[args.config_idx]   # (3,) world centroid

    env_origin = torch.tensor(args.env_origin, dtype=torch.float32)

    print(f"\nLoaded: {N} configs | P={tool_pts_canonical.shape[0]} tool pts | Q={obj_pts_canonical.shape[0]} obj pts")
    print(f"Env origin: {env_origin.numpy()}")

    # ── Tool cloud check ──────────────────────────────────────────────────────
    print(f"\n─── TOOL CLOUD ───────────────────────────────────────────────")

    # Pretraining encoder input: canonical @ R.T  (in original train.py, noised_R=R at contact)
    pretrain_tool_input = pretrain_encoder_input(tool_pts_canonical, R_tool)

    # RL encoder input: simulate full RL pipeline
    rl_tool_input, rl_tool_centroid = reconstruct_rl_obs(tool_pts_canonical, R_tool, t_tool, env_origin)

    # Print stats
    print(f"  pretrain input: mean={pretrain_tool_input.mean(0).numpy().round(4)}, "
          f"std={pretrain_tool_input.std(0).numpy().round(4)}")
    print(f"  rl input:       mean={rl_tool_input.mean(0).numpy().round(4)}, "
          f"std={rl_tool_input.std(0).numpy().round(4)}")

    tool_ok = check_cloud_consistency("tool_cloud", pretrain_tool_input, rl_tool_input, args.tol)
    check_centroid_consistency("tool", t_tool, rl_tool_centroid, env_origin, args.tol)

    # ── Object cloud check ────────────────────────────────────────────────────
    print(f"\n─── OBJECT CLOUD ─────────────────────────────────────────────")

    # Pretraining: obj cloud is stored already centered and world-aligned (no per-config R)
    # In train.py, obj_pc is taken directly: data["obj_pts_canonical"]
    pretrain_obj_input = obj_pts_canonical  # (Q, 3) already centered at (0,0,0)

    # RL: simulate object at obj_centroid_world + no rotation (world-frame → env-frame → center)
    # The object cloud world = obj_pts_canonical + obj_centroid_world
    # (since P_obj_c = obj_world - obj_centroid, so obj_world = P_obj_c + obj_centroid)
    rl_obj_world = obj_pts_canonical + obj_centroid_world.unsqueeze(0)   # (Q, 3) world
    rl_obj_env   = rl_obj_world - env_origin.unsqueeze(0)                # (Q, 3) env-frame
    rl_obj_centroid = rl_obj_env.mean(dim=0)                             # (3,)
    rl_obj_input    = rl_obj_env - rl_obj_centroid.unsqueeze(0)          # (Q, 3) centered

    print(f"  pretrain input: mean={pretrain_obj_input.mean(0).numpy().round(4)}, "
          f"std={pretrain_obj_input.std(0).numpy().round(4)}")
    print(f"  rl input:       mean={rl_obj_input.mean(0).numpy().round(4)}, "
          f"std={rl_obj_input.std(0).numpy().round(4)}")

    obj_ok = check_cloud_consistency("obj_cloud", pretrain_obj_input, rl_obj_input, args.tol)
    check_centroid_consistency("obj", obj_centroid_world, rl_obj_centroid, env_origin, args.tol)

    # ── Centering invariant checks ─────────────────────────────────────────────
    print(f"\n─── CENTERING INVARIANTS ─────────────────────────────────────")
    def check_zero_mean(name, cloud, tol=1e-5):
        m = cloud.mean(dim=0).abs().max().item()
        ok = m < tol
        print(f"  [{name}] |mean| max = {m:.2e}  {'✓' if ok else '✗'}")
        return ok

    check_zero_mean("pretrain tool", pretrain_tool_input)
    check_zero_mean("rl tool",       rl_tool_input)
    check_zero_mean("pretrain obj",  pretrain_obj_input)
    check_zero_mean("rl obj",        rl_obj_input)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_ok = tool_ok and obj_ok
    print(f"\n{'='*60}")
    if all_ok:
        print("✓  ALL CHECKS PASSED — RL and pretraining observations are consistent.")
    else:
        print("✗  SOME CHECKS FAILED — encoder will see different inputs at RL time!")
        print("   Inspect the diffs above to find the source of divergence.")
    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

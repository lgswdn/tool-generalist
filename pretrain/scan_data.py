"""Scan all .pt files for NaN/Inf/extreme values that cause NaN gradients.

Usage:
    python scan_data.py /path/to/data_dir
"""
import torch
import glob
import sys
from pathlib import Path

data_dir = sys.argv[1]
files = sorted(glob.glob(str(Path(data_dir) / "**/*.pt"), recursive=True))
print(f"Scanning {len(files)} files...\n")

bad_files = []
all_delta_t_abs_max = []
all_delta_R_abs_max = []
all_delta_pose_norm_abs_max = []

for fi, f in enumerate(files):
    try:
        data = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print("one ee")
        continue
    fname = Path(f).name
    issues = []

    # Check all tensors for NaN/Inf
    for key in data:
        v = data[key]
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                issues.append(f"  NaN in {key}: count={torch.isnan(v).sum().item()}")
            if torch.isinf(v).any():
                issues.append(f"  Inf in {key}: count={torch.isinf(v).sum().item()}")

    # Check delta pose magnitudes
    if "init_translations" in data and "init_rotations" in data:
        n_cfg = data["tool_translations"].shape[0]
        for i in range(n_cfg):
            init_t = data["init_translations"][i]
            init_R = data["init_rotations"][i]
            contact_t = data["tool_translations"][i]
            contact_R = data["tool_rotations"][i]

            delta_t = contact_t - init_t
            delta_R = contact_R @ init_R.T
            delta_R_6d = delta_R[:, :2].reshape(6)

            # Normalized values (same as dataset.py)
            delta_t_norm = delta_t / 0.1287
            delta_R_6d_norm = delta_R_6d / 0.5773
            delta_pose_norm = torch.cat([delta_t_norm, delta_R_6d_norm])

            dt_max = delta_t.abs().max().item()
            dr_max = delta_R_6d.abs().max().item()
            dp_norm_max = delta_pose_norm.abs().max().item()

            all_delta_t_abs_max.append(dt_max)
            all_delta_R_abs_max.append(dr_max)
            all_delta_pose_norm_abs_max.append(dp_norm_max)

            if dp_norm_max > 10:
                issues.append(f"  cfg[{i}] EXTREME normalized delta_pose: abs_max={dp_norm_max:.2f} "
                              f"(dt={dt_max:.4f}, dr={dr_max:.4f})")

    # Check SDF values
    if "tool_pts_sdf" in data:
        sdf_max = data["tool_pts_sdf"].abs().max().item()
        if sdf_max > 10:
            issues.append(f"  tool_pts_sdf abs_max={sdf_max:.4f}")
    if "obj_pts_sdf" in data:
        sdf_max = data["obj_pts_sdf"].abs().max().item()
        if sdf_max > 10:
            issues.append(f"  obj_pts_sdf abs_max={sdf_max:.4f}")

    if issues:
        bad_files.append((fname, issues))
        print(f"⚠ {fname}:")
        for iss in issues:
            print(iss)

    if (fi + 1) % 100 == 0:
        print(f"  ... scanned {fi+1}/{len(files)}")

# Summary
print(f"\n{'='*60}")
print(f"Total files: {len(files)}")
print(f"Bad files: {len(bad_files)}")

if all_delta_pose_norm_abs_max:
    t = torch.tensor(all_delta_pose_norm_abs_max)
    print(f"\nNormalized delta_pose abs_max distribution:")
    print(f"  mean={t.mean():.2f}  std={t.std():.2f}")
    print(f"  median={t.median():.2f}")
    print(f"  p95={t.quantile(0.95):.2f}  p99={t.quantile(0.99):.2f}  max={t.max():.2f}")
    print(f"  count > 5: {(t > 5).sum().item()}")
    print(f"  count > 10: {(t > 10).sum().item()}")
    print(f"  count > 20: {(t > 20).sum().item()}")

if all_delta_t_abs_max:
    t = torch.tensor(all_delta_t_abs_max)
    print(f"\nRaw delta_t abs_max: mean={t.mean():.4f} max={t.max():.4f} p99={t.quantile(0.99):.4f}")

if all_delta_R_abs_max:
    t = torch.tensor(all_delta_R_abs_max)
    print(f"Raw delta_R_6d abs_max: mean={t.mean():.4f} max={t.max():.4f} p99={t.quantile(0.99):.4f}")

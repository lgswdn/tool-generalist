"""Compute normalization statistics (mean, std) for delta_pose across all .pt files.

Usage:
    python compute_norm_stats.py /path/to/data_dir

Outputs the values to paste into dataset.py.
"""
import torch
import glob
import sys
from pathlib import Path

data_dir = sys.argv[1]
files = sorted(glob.glob(str(Path(data_dir) / "*.pt")))
print(f"Scanning {len(files)} files...")

all_delta_t = []
all_delta_R_6d = []

for f in files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    if "init_translations" not in data:
        continue

    n_cfg = data["tool_translations"].shape[0]
    for i in range(n_cfg):
        init_t = data["init_translations"][i]
        init_R = data["init_rotations"][i]
        contact_t = data["tool_translations"][i]
        contact_R = data["tool_rotations"][i]

        delta_t = contact_t - init_t
        delta_R = contact_R @ init_R.T
        delta_R_6d = delta_R[:, :2].reshape(6)

        all_delta_t.append(delta_t)
        all_delta_R_6d.append(delta_R_6d)

all_delta_t = torch.stack(all_delta_t)       # (N, 3)
all_delta_R_6d = torch.stack(all_delta_R_6d)  # (N, 6)

print(f"\nTotal configs: {len(all_delta_t)}")

print(f"\n--- delta_t (translation) ---")
print(f"  mean:  {all_delta_t.mean(0).tolist()}")
print(f"  std:   {all_delta_t.std(0).tolist()}")
print(f"  global_std: {all_delta_t.std().item():.6f}")
print(f"  min:   {all_delta_t.min(0).values.tolist()}")
print(f"  max:   {all_delta_t.max(0).values.tolist()}")

print(f"\n--- delta_R_6d (rotation) ---")
print(f"  mean:  {all_delta_R_6d.mean(0).tolist()}")
print(f"  std:   {all_delta_R_6d.std(0).tolist()}")
print(f"  global_std: {all_delta_R_6d.std().item():.6f}")
print(f"  min:   {all_delta_R_6d.min(0).values.tolist()}")
print(f"  max:   {all_delta_R_6d.max(0).values.tolist()}")

t_std = all_delta_t.std().item()
r_std = all_delta_R_6d.std().item()
print(f"\n=== Recommended normalization constants ===")
print(f"  DELTA_T_STD = {t_std:.4f}")
print(f"  DELTA_R_STD = {r_std:.4f}")
print(f"\nPaste into dataset.py:")
print(f"  delta_t_norm = delta_t / {t_std:.4f}")
print(f"  delta_R_6d_norm = delta_R_6d / {r_std:.4f}")

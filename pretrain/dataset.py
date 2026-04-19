"""dataset.py — ContactDataset for geometry encoder pretraining.

Each .pt file stores:
  Shared (once per file):
    tool_pts_canonical  (P, 3)  surface points in canonical tool frame
    obj_pts_canonical   (Q, 3)  surface points in canonical object frame (before R_obj)
    object_rotation     (3, 3)  R_obj
    obj_z_shift         scalar  grounding offset

  Per contact config:
    tool_translations  (N, 3)
    tool_rotations     (N, 3, 3)
    tool_pts_sdf       (N, P)  signed SDF: tool canonical pts → object  (+out/-in)
    obj_pts_sdf        (N, Q)  signed SDF: object canonical pts → tool  (+out/-in)
    contact_pts_obj_frame   (N, 5, 3)
    contact_pts_tool_frame  (N, 5, 3)
    contact_normals         (N, 5, 3)

Each dataset item corresponds to one config index i from one .pt file.
The canonical clouds serve as the geometry input; the SDF arrays provide
the dense supervision signal for mutual tool↔object encoder training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUM_TOOL_PTS = 512  # matches ICPNet expected input size
NUM_OBJ_PTS  = 512


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ContactDataset(Dataset):
    """Each item is one (tool_pc, object_pc, contact_pts, sdf, …) bundle.

    Geometry:
        tool_pc  — canonical tool cloud applied with tool rotation + translation
        obj_pc   — canonical object cloud applied with R_obj + z_shift

    SDF supervision (per-config, same point ordering as the clouds above):
        tool_pts_sdf  — signed SDF of tool canonical pts to object
        obj_pts_sdf   — signed SDF of object canonical pts to tool

    Note: Data is generated at RL scale (tool_scale=0.1, object_scale in 0.1-0.2).
          No additional scale augmentation needed.
    """

    def __init__(
        self,
        pt_files: List[str],
        augment: bool = True,
    ):
        self.augment = augment

        # Expand: (pt_file, config_index) pairs
        self._index: List[Tuple[str, int]] = []
        self._pt_cache: dict = {}

        for path in pt_files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            n = data["tool_translations"].shape[0]
            for i in range(n):
                self._index.append((path, i))
            self._pt_cache[path] = data

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        pt_path, cfg_i = self._index[idx]
        data = self._pt_cache[pt_path]

        # ---- Canonical clouds (stored once per file, already at RL scale) ----
        P_tool = data["tool_pts_canonical"]  # (P, 3) - already scaled by tool_scale
        P_obj  = data["obj_pts_canonical"]   # (Q, 3) - already scaled by object_scale

        # ---- Reconstruct world-frame clouds ----------------------------------
        # Tool: apply rotation + translation for this config
        R_tool = data["tool_rotations"][cfg_i]      # (3, 3)
        t_tool = data["tool_translations"][cfg_i]   # (3,)
        tool_pc = P_tool @ R_tool.T + t_tool         # (P, 3)

        # Object: apply R_obj + z_shift (same for all configs in this file)
        R_obj   = data["object_rotation"]            # (3, 3)
        z_shift = data["obj_z_shift"]               # scalar
        obj_pc  = P_obj @ R_obj.T                    # (Q, 3)
        obj_pc  = obj_pc.clone()
        obj_pc[:, 2] -= z_shift

        # ---- SDF arrays (per-config) -----------------------------------------
        tool_pts_sdf = data["tool_pts_sdf"][cfg_i]  # (P,)  signed
        obj_pts_sdf  = data["obj_pts_sdf"][cfg_i]   # (Q,)  signed

        # ---- Contact geometry (sparse) ---------------------------------------
        # Use world-frame contacts (new key name)
        contact_pts     = data["contact_pts_world"][cfg_i]      # (5, 3)
        contact_normals = data["contact_normals"][cfg_i]        # (5, 3)

        # ---- Diffusion inputs: delta pose from initial to contact ----------
        delta_pose = None
        if "init_translations" in data and "init_rotations" in data:
            # Initial pose
            init_t = data["init_translations"][cfg_i]    # (3,)
            init_R = data["init_rotations"][cfg_i]       # (3, 3)

            # Contact pose
            contact_t = data["tool_translations"][cfg_i]  # (3,)
            contact_R = data["tool_rotations"][cfg_i]     # (3, 3)

            # Delta translation
            delta_t = contact_t - init_t  # (3,)

            # Delta rotation: R_delta = R_contact @ R_init^{-1}
            # Actually, for diffusion, we want: initial -> contact
            # So delta_R = R_contact @ R_init.T (apply delta to initial gives contact)
            delta_R = contact_R @ init_R.T  # (3, 3)

            # Convert to 6D representation (first two columns)
            delta_R_6d = delta_R[:, :2].reshape(6)  # (6,) = first two columns flattened

            # Full delta pose: translation (3) + 6D rotation (6) = 9D
            delta_pose = torch.cat([delta_t, delta_R_6d], dim=0)  # (9,)

        # ---- Augmentation: small Gaussian jitter only ------------------------
        if self.augment:
            tool_pc = tool_pc + torch.randn_like(tool_pc) * 1e-3
            obj_pc  = obj_pc  + torch.randn_like(obj_pc)  * 1e-3

        return {
            # World-frame clouds (for encoder input)
            "tool_pc":             tool_pc.float(),          # (P, 3)
            "obj_pc":              obj_pc.float(),           # (Q, 3)
            # Canonical clouds (pose-invariant geometry)
            "tool_pts_canonical":  P_tool.float(),           # (P, 3)
            "obj_pts_canonical":   P_obj.float(),            # (Q, 3)
            # Dense SDF supervision
            "tool_pts_sdf":        tool_pts_sdf.float(),     # (P,)
            "obj_pts_sdf":         obj_pts_sdf.float(),      # (Q,)
            # Sparse contact geometry
            "contact_pts":         contact_pts.float(),      # (5, 3)
            "contact_normals":     contact_normals.float(),  # (5, 3)
            # Diffusion supervision (optional - None if init poses not generated)
            "delta_pose":          delta_pose.float() if delta_pose is not None else None,
        }


# --------------------------------------------------------------------------- #
# Helpers for building the dataset
# --------------------------------------------------------------------------- #

def collect_pt_files(data_dir: str) -> List[str]:
    """Recursively find all .pt files under data_dir."""
    return sorted(str(p) for p in Path(data_dir).rglob("*.pt"))


def make_split(
    data_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[ContactDataset, ContactDataset]:
    """Return (train_dataset, val_dataset) split by file."""
    files = collect_pt_files(data_dir)
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files   = files[:n_val]
    train_files = files[n_val:]
    return (
        ContactDataset(train_files, augment=augment),
        ContactDataset(val_files,   augment=False),
    )

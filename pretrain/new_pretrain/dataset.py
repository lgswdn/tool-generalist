"""dataset.py — Dataset for RPDiff-style pretraining.

Reuses the existing ContactDataset and adds the fields needed for diffusion.
The key difference: we return canonical tool + object (world), plus the contact
pose separately (not applied to the tool cloud).

The noising is done on-the-fly in the training loop, NOT in the dataset.
"""

from __future__ import annotations

import sys
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import trimesh

_NEW_PRETRAIN_DIR = Path(__file__).resolve().parent
_PRETRAIN_DIR     = _NEW_PRETRAIN_DIR.parent

NUM_TOOL_PTS = 512
NUM_OBJ_PTS  = 512


class NewPretrainDataset(Dataset):
    """Dataset for RPDiff-style joint SDF + denoising pretraining.

    Each item returns:
      - tool_canonical:  (P, 3) — canonical tool points (origin, R=I, already scaled)
      - obj_pc:          (Q, 3) — object points in world frame (R_obj applied, z-grounded)
      - contact_R:       (3, 3) — contact rotation matrix
      - contact_t:       (3,)   — contact translation
      - tool_sdf:        (P,)   — signed SDF at contact pose (from dataset)
      - obj_sdf:         (Q,)   — signed SDF at contact pose

    Noised poses are generated on-the-fly in the training loop.
    """

    def __init__(
        self,
        pt_files: List[str],
        augment: bool = True,
        require_movement: bool = False,
    ):
        self.augment = augment
        self.require_movement = require_movement

        self._index: List[Tuple[str, int]] = []
        self._pt_cache: dict = {}
        self._mesh_cache: dict = {}
        self._skipped_files: List[str] = []

        for path in pt_files:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                # Skip files that are missing required movement keys
                if require_movement:
                    movement_keys = {
                        "delta_tool_translations", "delta_tool_rotations",
                        "delta_obj_translations", "delta_obj_rotations",
                    }
                    missing = movement_keys - set(data.keys())
                    if missing:
                        warnings.warn(f"Skipping {path}: missing movement keys {sorted(missing)}")
                        continue
                n = data["tool_translations"].shape[0]
                for i in range(n):
                    self._index.append((path, i))
                self._pt_cache[path] = data
                if require_movement:
                    self._mesh_cache[path] = self._load_mesh_data(data)
            except (RuntimeError, IOError, OSError) as e:
                warnings.warn(f"Skipping corrupted file {path}: {e}")
                self._skipped_files.append(path)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        pt_path, cfg_i = self._index[idx]
        data = self._pt_cache[pt_path]

        # ── Canonical tool cloud → center at (0,0,0) ─────────────────
        # Raw canonical is in mesh frame (uncentered). We subtract the
        # centroid so the encoder sees a standardised input and the pose
        # becomes a pure placement transform (consistent with RL).
        P_tool_raw = data["tool_pts_canonical"]          # (P, 3)
        tool_centroid = P_tool_raw.mean(dim=0)           # (3,)
        P_tool = P_tool_raw - tool_centroid              # (P, 3) centered

        # ── Object cloud: world frame → centered ─────────────────────
        P_obj   = data["obj_pts_canonical"]                 # (Q, 3)
        obj_pc_world = P_obj.clone()

        # Ground object
        z_min_obj = obj_pc_world[:, 2].min()
        if z_min_obj < 0:
            obj_pc_world[:, 2] -= z_min_obj

        if "object_rotation" in data:
            R_obj = data["object_rotation"].float()
            obj_pc_world = obj_pc_world @ R_obj.T
        if "obj_z_shift" in data:
            obj_pc_world = obj_pc_world.clone()
            obj_pc_world[:, 2] -= float(data["obj_z_shift"])

        obj_centroid   = obj_pc_world.mean(dim=0)            # (3,) world position
        P_obj_centered = obj_pc_world - obj_centroid         # (Q, 3) centered

        # ── Contact pose (adjusted for centroid shift) ────────────────
        contact_R     = data["tool_rotations"][cfg_i]        # (3, 3)
        contact_t_raw = data["tool_translations"][cfg_i]     # (3,)
        contact_t = contact_R @ tool_centroid + contact_t_raw  # (3,)

        # ── SDF at contact pose (signed, positive = outside) ──────────
        tool_sdf = data["tool_pts_sdf"][cfg_i]              # (P,)
        obj_sdf  = data["obj_pts_sdf"][cfg_i]               # (Q,)

        # ── Augmentation: small Gaussian jitter ───────────────────────
        if self.augment:
            P_tool         = P_tool         + torch.randn_like(P_tool)         * 1e-3
            P_obj_centered = P_obj_centered + torch.randn_like(P_obj_centered) * 1e-3

        # ── Build item dict ───────────────────────────────────────────
        item = {
            "tool_canonical":    P_tool.float(),          # (P, 3) centered
            "tool_centroid_raw": tool_centroid.float(),   # (3,)   mesh-frame centroid
            "obj_pc":            P_obj_centered.float(),  # (Q, 3) centered
            "obj_centroid":      obj_centroid.float(),    # (3,)   world-frame centroid
            "contact_R":         contact_R.float(),       # (3, 3)
            "contact_t":         contact_t.float(),       # (3,)   world frame
            "tool_sdf":          tool_sdf.float(),        # (P,)   signed
            "obj_sdf":           obj_sdf.float(),         # (Q,)   signed
            "pt_path":           pt_path,                 # str    for mesh cache
        }

        # ── Movement delta conditioning (optional) ────────────────────
        if self.require_movement:
            item["delta_tool_t"] = data["delta_tool_translations"][cfg_i].float()  # (3,)
            item["delta_tool_R"] = data["delta_tool_rotations"][cfg_i].float()     # (3,3)
            item["delta_obj_t"]  = data["delta_obj_translations"][cfg_i].float()   # (3,)
            item["delta_obj_R"]  = data["delta_obj_rotations"][cfg_i].float()      # (3,3)

        return item


# ── Build helpers ─────────────────────────────────────────────────────────────

def collect_pt_files(data_dir: str) -> List[str]:
    """Recursively find all .pt files under data_dir."""
    return sorted(str(p) for p in Path(data_dir).rglob("*.pt"))


def make_split(
    data_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    max_files: int = 0,
    require_movement: bool = False,
) -> Tuple[NewPretrainDataset, NewPretrainDataset]:
    """Return (train_dataset, val_dataset) split by file."""
    files = collect_pt_files(data_dir)
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files > 0:
        files = files[:max_files]
    n_val = max(1, int(len(files) * val_ratio))
    val_files   = files[:n_val]
    train_files = files[n_val:]
    if not train_files:
        train_files = val_files
    return (
        NewPretrainDataset(train_files, augment=augment, require_movement=require_movement),
        NewPretrainDataset(val_files,   augment=False, require_movement=require_movement),
    )

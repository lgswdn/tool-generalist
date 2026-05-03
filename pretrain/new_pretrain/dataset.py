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
    ):
        self.augment = augment

        self._index: List[Tuple[str, int]] = []
        self._pt_cache: dict = {}
        self._skipped_files: List[str] = []

        for path in pt_files:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                n = data["tool_translations"].shape[0]
                for i in range(n):
                    self._index.append((path, i))
                self._pt_cache[path] = data
            except (RuntimeError, IOError, OSError) as e:
                warnings.warn(f"Skipping corrupted file {path}: {e}")
                self._skipped_files.append(path)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        pt_path, cfg_i = self._index[idx]
        data = self._pt_cache[pt_path]

        # ── Canonical tool cloud (origin, R=I, already scaled) ────────
        P_tool = data["tool_pts_canonical"]              # (P, 3)

        # ── Object cloud (world frame) ────────────────────────────────
        P_obj   = data["obj_pts_canonical"]              # (Q, 3)
        R_obj   = data["object_rotation"]                # (3, 3)
        z_shift = data["obj_z_shift"]                    # scalar
        obj_pc  = P_obj @ R_obj.T                        # (Q, 3)
        obj_pc  = obj_pc.clone()
        obj_pc[:, 2] -= z_shift

        # ── Contact pose ──────────────────────────────────────────────
        contact_R = data["tool_rotations"][cfg_i]        # (3, 3)
        contact_t = data["tool_translations"][cfg_i]     # (3,)

        # ── SDF at contact pose ───────────────────────────────────────
        tool_sdf = data["tool_pts_sdf"][cfg_i]           # (P,)
        obj_sdf  = data["obj_pts_sdf"][cfg_i]            # (Q,)

        # ── Augmentation: small Gaussian jitter ───────────────────────
        if self.augment:
            P_tool = P_tool + torch.randn_like(P_tool) * 1e-3
            obj_pc = obj_pc + torch.randn_like(obj_pc) * 1e-3

        return {
            "tool_canonical": P_tool.float(),
            "obj_pc":         obj_pc.float(),
            "contact_R":      contact_R.float(),
            "contact_t":      contact_t.float(),
            "tool_sdf":       tool_sdf.float(),
            "obj_sdf":        obj_sdf.float(),
        }


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
        NewPretrainDataset(train_files, augment=augment),
        NewPretrainDataset(val_files,   augment=False),
    )

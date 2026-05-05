"""dataset.py — Dataset for RPDiff-style pretraining.

Architecture change: both tool and object point clouds are centered at (0,0,0).
Centroids are returned separately as conditioning signals for cross-attention.
Mesh data (tool_verts_centered, obj_verts_world) is cached per pt-file for
on-the-fly signed SDF computation via kaolin.

Contact frame is WORLD frame (object grounded at z_min=0, at origin).
obj_centroid is typically at (x~0, y~0, z>0) — NOT at origin.
The 6D cross-attention conditioning [noised_t, obj_centroid] carries absolute
world positions so the model can infer the relative tool-object offset.
"""

from __future__ import annotations

import sys
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import trimesh
from torch.utils.data import Dataset

_NEW_PRETRAIN_DIR = Path(__file__).resolve().parent
_PRETRAIN_DIR     = _NEW_PRETRAIN_DIR.parent

NUM_TOOL_PTS = 512
NUM_OBJ_PTS  = 512


class NewPretrainDataset(Dataset):
    """Dataset for RPDiff-style joint SDF + denoising pretraining.

    Each item returns:
      - tool_canonical:    (P, 3) — canonical tool points CENTERED at (0,0,0),
                           in mesh frame (R=I). Encoder rotates this with noised_R.
      - tool_centroid_raw: (3,)   — tool centroid in original mesh frame (used to
                           recover world-frame mesh vertices for kaolin SDF).
      - obj_pc:            (Q, 3) — object points CENTERED at (0,0,0).
      - obj_centroid:      (3,)   — object centroid in world frame (z-grounded).
                           Used as part of 6D cross-attention conditioning.
      - contact_R:         (3, 3) — contact rotation matrix
      - contact_t:         (3,)   — contact translation (world frame, for centered tool)
      - tool_sdf:          (P,)   — signed SDF at contact pose (positive=outside)
      - obj_sdf:           (Q,)   — signed SDF at contact pose (positive=outside)
      - pt_path:           str    — path to the .pt file (for mesh cache lookup)

    Mesh data (tool verts/faces, obj verts/faces in world frame) is accessible via
    `dataset._mesh_cache[pt_path]` for use in on-the-fly signed SDF computation.
    """

    def __init__(
        self,
        pt_files: List[str],
        augment: bool = True,
    ):
        self.augment = augment

        self._index: List[Tuple[str, int]] = []
        self._pt_cache: dict = {}
        self._mesh_cache: dict = {}   # path -> {tool_verts, tool_faces, obj_verts, obj_faces}
        self._skipped_files: List[str] = []

        for path in pt_files:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                n = data["tool_translations"].shape[0]
                for i in range(n):
                    self._index.append((path, i))
                self._pt_cache[path] = data
                self._load_mesh_cache(path, data)
            except (RuntimeError, IOError, OSError) as e:
                warnings.warn(f"Skipping corrupted file {path}: {e}")
                self._skipped_files.append(path)

    # ── Mesh loading ─────────────────────────────────────────────────────────

    def _load_mesh_cache(self, path: str, data: dict) -> None:
        """Load and preprocess tool + object mesh into _mesh_cache."""
        try:
            tool_scale = float(data.get("tool_scale", 0.1))
            obj_scale  = float(data.get("object_scale", 0.1))

            # ── Tool mesh ────────────────────────────────────────────────────
            # Scale, then subtract tool centroid so mesh is in the same frame
            # as the centered tool_canonical point cloud.
            tool_mesh_path = data.get("tool_mesh_path", "")
            tool_verts_c, tool_faces = None, None
            if tool_mesh_path and Path(tool_mesh_path).exists():
                tm = trimesh.load(tool_mesh_path, force="mesh", process=False)
                tv = torch.tensor(tm.vertices, dtype=torch.float32) * tool_scale
                # Center with the same centroid as the point cloud
                tool_centroid = data["tool_pts_canonical"].mean(dim=0)  # (3,)
                tool_verts_c = tv - tool_centroid        # centered, same frame as tool_canonical
                tool_faces   = torch.tensor(tm.faces, dtype=torch.int64)

            # ── Object mesh ──────────────────────────────────────────────────
            # Apply the same transformations as the object point cloud:
            #   scale → R_obj rotation → z_shift → this gives world-frame verts.
            obj_mesh_path = data.get("object_mesh_path", "")
            obj_verts_w, obj_faces = None, None
            if obj_mesh_path and Path(obj_mesh_path).exists():
                om = trimesh.load(obj_mesh_path, force="mesh", process=False)
                ov = torch.tensor(om.vertices, dtype=torch.float32) * obj_scale

                # Ground object (z_min = 0) — matches contact_gen_gradient behaviour
                z_min = ov[:, 2].min()
                if z_min < 0:
                    ov[:, 2] -= z_min

                # Apply object rotation if present (identity for gradient-gen files)
                if "object_rotation" in data:
                    R_obj = data["object_rotation"].float()        # (3, 3)
                    ov = ov @ R_obj.T

                # Apply z_shift if present
                if "obj_z_shift" in data:
                    ov[:, 2] -= float(data["obj_z_shift"])

                obj_verts_w = ov                         # world frame
                obj_faces   = torch.tensor(om.faces, dtype=torch.int64)

            self._mesh_cache[path] = {
                "tool_verts": tool_verts_c,   # (V_t, 3) centered, or None
                "tool_faces": tool_faces,      # (F_t, 3) or None
                "obj_verts":  obj_verts_w,     # (V_o, 3) world-frame, or None
                "obj_faces":  obj_faces,       # (F_o, 3) or None
            }

        except Exception as e:
            warnings.warn(f"Mesh load failed for {path}: {e}. SDF will be approximate.")
            self._mesh_cache[path] = {
                "tool_verts": None, "tool_faces": None,
                "obj_verts":  None, "obj_faces":  None,
            }

    # ── Item access ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        pt_path, cfg_i = self._index[idx]
        data = self._pt_cache[pt_path]

        # ── Tool canonical cloud → centered at (0,0,0) ────────────────────
        P_tool_raw    = data["tool_pts_canonical"]          # (P, 3)  raw mesh frame
        tool_centroid = P_tool_raw.mean(dim=0)              # (3,)
        P_tool        = P_tool_raw - tool_centroid          # (P, 3)  centered

        # ── Object cloud → world frame then centered at (0,0,0) ──────────
        P_obj   = data["obj_pts_canonical"]                 # (Q, 3)
        obj_pc_world = P_obj.clone()

        # Ground object (ensure z_min ≥ 0) — always applied for robustness
        z_min_obj = obj_pc_world[:, 2].min()
        if z_min_obj < 0:
            obj_pc_world[:, 2] -= z_min_obj

        # Apply object rotation if stored
        if "object_rotation" in data:
            R_obj = data["object_rotation"].float()
            obj_pc_world = obj_pc_world @ R_obj.T

        # Apply z_shift if stored
        if "obj_z_shift" in data:
            obj_pc_world = obj_pc_world.clone()
            obj_pc_world[:, 2] -= float(data["obj_z_shift"])

        # Center the object cloud; save centroid for cross-attention
        obj_centroid  = obj_pc_world.mean(dim=0)            # (3,)  world position
        P_obj_centered = obj_pc_world - obj_centroid         # (Q, 3) centered

        # ── Contact pose (adjusted for tool centroid shift) ───────────────
        contact_R     = data["tool_rotations"][cfg_i]        # (3, 3)
        contact_t_raw = data["tool_translations"][cfg_i]     # (3,)
        # World position of tool centroid at contact:
        #   tool_world_i = R @ (canonical_i - centroid) + contact_t_raw + R @ centroid
        #                = R @ canonical_i + contact_t_raw
        # Centroid world pos = R @ centroid + contact_t_raw → stored as contact_t
        contact_t = contact_R @ tool_centroid + contact_t_raw  # (3,)

        # ── Signed SDF at contact pose (positive = outside) ───────────────
        # Already signed from contact_gen_gradient.py (kaolin check_sign used)
        tool_sdf = data["tool_pts_sdf"][cfg_i]              # (P,)
        obj_sdf  = data["obj_pts_sdf"][cfg_i]               # (Q,)

        # ── Augmentation: small Gaussian jitter ───────────────────────────
        if self.augment:
            P_tool         = P_tool         + torch.randn_like(P_tool)         * 1e-3
            P_obj_centered = P_obj_centered + torch.randn_like(P_obj_centered) * 1e-3

        return {
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

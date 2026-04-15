"""dataset.py — ContactDataset for geometry encoder pretraining.

Each sample = one converged contact config from a .pt file:
  - tool_pc   : (512, 3) tool point cloud in WORLD frame (posed)
  - object_pc : (512, 3) object point cloud in WORLD frame
  - contact_pts: (5, 3)  ground-truth contact points in world/object frame

The tool cloud is constructed by loading the canonical tool OBJ, sampling
NUM_TOOL_PTS surface points, then applying tool_rotations[i] + tool_translations[i].

The object cloud is constructed by loading the object OBJ, applying
object_rotation, then sampling NUM_OBJ_PTS surface points.
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
# Mesh sampling utilities
# --------------------------------------------------------------------------- #

def _triangle_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return area of each triangle face. verts: (V,3), faces: (F,3) → (F,)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    return 0.5 * cross.norm(dim=-1)


def sample_mesh_surface(verts: torch.Tensor, faces: torch.Tensor,
                         n: int, seed: int | None = None) -> torch.Tensor:
    """Uniformly sample n surface points via area-weighted barycentric sampling."""
    areas = _triangle_areas(verts, faces)
    probs = areas / areas.sum()
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None
    face_idx = torch.multinomial(probs, n, replacement=True, generator=gen)
    selected_faces = faces[face_idx]  # (n, 3)
    v0 = verts[selected_faces[:, 0]]
    v1 = verts[selected_faces[:, 1]]
    v2 = verts[selected_faces[:, 2]]
    # Random barycentric coords
    r1 = torch.rand(n, 1, generator=gen)
    r2 = torch.rand(n, 1, generator=gen)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    pts = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
    return pts  # (n, 3)


def _load_obj_verts_faces(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal .obj loader — returns (V,3) verts and (F,3) tri faces."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                ids = [int(t.split("/")[0]) - 1 for t in tokens]
                # Fan triangulation for quads/polygons
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
    return (torch.tensor(verts, dtype=torch.float32),
            torch.tensor(faces, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ContactDataset(Dataset):
    """Each item is one (tool_pc, object_pc, contact_pts) triple."""

    def __init__(
        self,
        pt_files: List[str],
        num_tool_pts: int = NUM_TOOL_PTS,
        num_obj_pts:  int = NUM_OBJ_PTS,
        augment: bool = True,
    ):
        self.num_tool_pts = num_tool_pts
        self.num_obj_pts  = num_obj_pts
        self.augment      = augment

        # Expand: (pt_file, config_index) pairs
        self._index: List[Tuple[str, int]] = []
        # Cache: mesh data per path to avoid re-loading
        self._mesh_cache: dict = {}

        for path in pt_files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            n = data["tool_translations"].shape[0]
            for i in range(n):
                self._index.append((path, i))

        # Pre-load all .pt metadata (not the meshes)
        self._pt_cache: dict = {}
        for path in pt_files:
            self._pt_cache[path] = torch.load(
                path, map_location="cpu", weights_only=False
            )

    def __len__(self) -> int:
        return len(self._index)

    def _get_mesh(self, mesh_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if mesh_path not in self._mesh_cache:
            v, f = _load_obj_verts_faces(mesh_path)
            self._mesh_cache[mesh_path] = (v, f)
        return self._mesh_cache[mesh_path]

    def __getitem__(self, idx: int):
        pt_path, cfg_i = self._index[idx]
        data = self._pt_cache[pt_path]

        # ---- Tool point cloud (world frame) --------------------------------
        tool_v, tool_f = self._get_mesh(data["tool_mesh_path"])
        tool_pts_canonical = sample_mesh_surface(tool_v, tool_f, self.num_tool_pts)
        R_tool = data["tool_rotations"][cfg_i]      # (3, 3)
        t_tool = data["tool_translations"][cfg_i]   # (3,)
        tool_pc = tool_pts_canonical @ R_tool.T + t_tool  # (N, 3)

        # ---- Object point cloud (world frame) ------------------------------
        obj_v, obj_f = self._get_mesh(data["object_mesh_path"])
        R_obj = data["object_rotation"]              # (3, 3)
        obj_pts = obj_v @ R_obj.T                   # apply rotation
        # Ground: shift so lowest z = 0
        obj_pts[:, 2] -= obj_pts[:, 2].min()
        obj_pc = sample_mesh_surface(
            obj_pts,
            obj_f,
            self.num_obj_pts,
        )

        # ---- Contact points + normals (world ≈ object frame) ----------------
        contact_pts     = data["contact_pts_obj_frame"][cfg_i]  # (5, 3)
        contact_normals = data["contact_normals"][cfg_i]        # (5, 3)

        # ---- Optional augmentation: small Gaussian jitter on both clouds ---
        if self.augment:
            tool_pc = tool_pc + torch.randn_like(tool_pc) * 1e-3
            obj_pc  = obj_pc  + torch.randn_like(obj_pc)  * 1e-3

        return {
            "tool_pc":          tool_pc.float(),       # (N_tool, 3)
            "object_pc":        obj_pc.float(),        # (N_obj, 3)
            "contact_pts":      contact_pts.float(),   # (5, 3)
            "contact_normals":  contact_normals.float(),  # (5, 3)
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
    **dataset_kwargs,
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
        ContactDataset(train_files, **dataset_kwargs),
        ContactDataset(val_files,   augment=False, **dataset_kwargs),
    )

"""contact_config.py — Centralised hyperparameters for new_pretrain contact generation.

All magic numbers live here.  Edit this file, not gen_dataset.py or contact_gen_new.py.
"""

from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Contact Generation  (contact_gen_new.py  — rejection sampling)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContactGenHyperparams:
    """Hyperparameters for the batched-SDF rejection-sampling contact generator."""

    # ── Scales (must match Isaac Lab RL) ──────────────────────────────────────
    tool_scale:          float = 0.1
    object_scale_range:  Tuple[float, float] = (0.1, 0.2)

    # ── Surface sampling ──────────────────────────────────────────────────────
    num_surface_pts: int = 512          # K: tool surface pts used in both sampling & SDF

    # ── Rejection sampler ─────────────────────────────────────────────────────
    B:            int   = 4096          # contact pairs (tool_pt × obj_pt) per call
    M:            int   = 1024          # candidate rotations tested per pair
    chunk_B:      int   = 512           # how many pairs processed per GPU kernel
                                        # Memory: chunk_B × M × K × 3 × 4 bytes
                                        # = 512 × 1024 × 512 × 12 ≈ 3 GB — safe on 48 GB

    # ── SDF grid ──────────────────────────────────────────────────────────────
    sdf_grid_res: int   = 128           # voxel grid resolution (128³)
    sdf_padding:  float = 0.05          # bbox padding around object (metres)

    # ── Geometric constraints ─────────────────────────────────────────────────
    upright_threshold: float = 0.0      # reject rotations where R[2,2] > threshold
                                        # (tool +Z must not point upward)
    epsilon:           float = 5e-3     # penetration tolerance (metres)
                                        # Must be > voxel grid spacing (~bbox/128 ≈ 1.6 mm)
                                        # to avoid rejecting valid poses where the contact
                                        # point reads as slightly inside due to trilinear
                                        # interpolation error on the 128³ SDF grid.
    floor_eps:         float = 1e-3     # min world-z allowed for any tool point

    # ── Convergence thresholds (kept for output compat with old pipeline) ──────
    pen_max_eps: float = 3e-4
    contact_eps: float = 8e-3


# ═══════════════════════════════════════════════════════════════════════════════
#  Singleton defaults (import and use directly)
# ═══════════════════════════════════════════════════════════════════════════════

CONTACT_GEN = ContactGenHyperparams()

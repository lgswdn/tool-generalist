"""contact_config.py — Centralised hyperparameters for all contact generation scripts.

All magic numbers (loss weights, thresholds, ranges, learning rates, etc.)
live here so they can be tuned in ONE place instead of scattered across
CLI flags in contact_gen.py, gen_initial.py, gen_movement_delta.py, etc.

CLI args in those scripts should only expose I/O paths, device, seed,
and batch-level controls.  Everything else reads from here.
"""

from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Contact Generation  (contact_gen.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContactGenHyperparams:
    """Hyperparameters for the optimisation-based contact generator."""

    # ---- Scale (matches RL) ----
    tool_scale: float = 0.1
    object_scale_range: Tuple[float, float] = (0.1, 0.2)

    # ---- Surface sampling ----
    num_surface_pts: int = 512
    contact_mode_prob: float = 0.7   # prob of targeting head (vs handle/body)

    # ---- Optimisation ----
    opt_steps: int = 200
    lr: float = 1e-3

    # ---- Loss weights ----
    w_pen: float = 30.0              # penetration penalty
    w_contact: float = 1.0           # attraction loss
    w_floor: float = 20.0            # below-floor penalty
    w_upright: float = 5.0           # penalise tool +Z pointing upward
    upright_threshold: float = 0.0   # relu(R[2,2] - threshold):
    k_closest: int = 4               # how many closest points for attraction

    # ---- Convergence thresholds ----
    pen_max_eps: float = 3e-4        # max allowed single-point penetration depth
    contact_eps: float = 8e-3        # max avg distance for "in contact"


# ═══════════════════════════════════════════════════════════════════════════════
#  Initial Pose Generation  (gen_initial.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InitialPoseHyperparams:
    """Hyperparameters for random initial-pose generation."""

    init_radius: float = 0.25        # metres from object centre
    collision_threshold: float = 0.001  # max allowed penetration (m)


# ═══════════════════════════════════════════════════════════════════════════════
#  Movement Delta Generation  (gen_movement_delta.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MovementDeltaHyperparams:
    """Hyperparameters for (ΔT, ΔO) movement-delta generation."""

    # ---- ΔT sampling range ----
    delta_t_min: float = 0.002       # 2 mm  minimum push translation
    delta_t_max: float = 0.01        # 10 mm maximum push translation
    delta_r_min_deg: float = 0.0     # minimum push rotation (degrees)
    delta_r_max_deg: float = 10.0    # maximum push rotation (degrees)

    # ---- ΔO optimisation ----
    opt_steps: int = 200
    lr: float = 1e-3

    # ---- Loss weights ----
    w_pen: float = 30.0              # penetration penalty  (same semantic as ContactGen)
    w_contact: float = 0.5           # surface attraction
    w_obj_floor: float = 60.0        # below-floor penalty (object vertices)
    w_reg_rot: float = 0.02          # penalise ΔO rotation magnitude
    k_closest: int = 4               # same as ContactGen

    # ---- Convergence thresholds ----
    pen_max_eps: float = 1e-3
    floor_max_eps: float = 1e-3

    # ---- Surface sampling ----
    num_surface_pts: int = 512


# ═══════════════════════════════════════════════════════════════════════════════
#  Singleton defaults (import and use directly)
# ═══════════════════════════════════════════════════════════════════════════════

CONTACT_GEN  = ContactGenHyperparams()
INITIAL_POSE = InitialPoseHyperparams()
MOVEMENT_DELTA = MovementDeltaHyperparams()

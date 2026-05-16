"""Stabilize geometry contact candidates into success-only artifacts.

This stage reads ``*.candidate.pt`` files written by ``gen_contact.py`` and
writes ``*.stabilized_success.pt`` files containing only successful stabilized
cases.  It intentionally does not import geometry, Kaolin, or trimesh; Isaac is
still loaded lazily through the physics runner.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from utils.contact.stabilize import get_physics_runner, sample_physical_properties

from .gen_contact import candidate_debug_path_for, load_candidate_artifact
from .gen_postcontact import save_stabilized_success_artifact


def run_stabilize_contact_pair(cfg: Any, physics_runner: Optional[Any] = None) -> int:
    _log(
        "[STABILIZE-START] "
        f"tool={cfg.tool_id} object={cfg.object_id} candidate={candidate_debug_path_for(cfg.output_path)}"
    )
    candidate_payload = load_candidate_artifact(cfg.output_path)
    candidates = candidate_payload["candidates"]
    n = int(candidate_payload["num_candidates"])
    _log(f"[STABILIZE-PROPS] sampling physical properties candidates={n}")
    physical_props = sample_physical_properties(
        n,
        seed=cfg.seed + 7919,
        object_mass_range=cfg.object_mass_range,
        tool_mass_range=cfg.tool_mass_range,
        object_friction_range=cfg.object_friction_range,
        tool_friction_range=cfg.tool_friction_range,
        ground_friction_range=cfg.ground_friction_range,
    )
    runner = physics_runner if physics_runner is not None else get_physics_runner(cfg.physics_runner)
    physics_cfg = replace(
        cfg.physics_config(float(candidate_payload["object_scale"])),
        run_postcontact=False,
    )
    _log(
        "[STABILIZE-PHYSICS] "
        f"runner={cfg.physics_runner} stabilize_steps={cfg.t_stabilize} candidates={n} "
        f"visualization_enabled={physics_cfg.visualization_enabled} "
        f"stabilization_picture={physics_cfg.visualization_stabilization_picture} "
        f"postcontact_video={physics_cfg.visualization_postcontact_video} "
        f"picture_dir={physics_cfg.visualization_picture_dir}"
    )
    physics = runner.run(candidates, physical_props, physics_cfg)
    success_count = int(physics.success_mask.detach().cpu().bool().sum().item())
    _log(
        "[STABILIZE-DONE] "
        f"runner={physics.runner} status={physics.status} stabilized={success_count}/{n}"
    )
    output = save_stabilized_success_artifact(
        cfg.output_path,
        candidate_payload=candidate_payload,
        physical_props=physical_props,
        physics=physics,
    )
    _log(f"[STABILIZE-SAVE] success_only={output}")
    return success_count


def _log(message: str) -> None:
    print(f"[contact_generation.stabilize] {message}", flush=True)

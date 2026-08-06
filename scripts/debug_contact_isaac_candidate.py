#!/usr/bin/env python3
"""Run one minimal contact-adapter Isaac candidate.

This script exercises the contact adapter path after ``SimulationApp`` startup:
it loads the fork SDF experiment config, resolves paths, picks the first selected
tool and first object, creates one conservative candidate, steps Isaac for one
stabilize frame and one post-contact frame, then closes.

It does not import contact_generation, pretrain, RL, or run dataset generation.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "fork_sdf.py"


def _log(message: str) -> None:
    print(f"[debug_contact_isaac_candidate] {message}", flush=True)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    return value if len(value) <= 240 else value[:240] + "..."


def _force_headless_argv() -> None:
    forced_args = (
        "--headless",
        "--no-window",
        "--/app/window/enabled=false",
        "--/app/viewport/enabled=false",
        "--/app/livestream/enabled=false",
    )
    for arg in forced_args:
        if arg not in sys.argv:
            sys.argv.append(arg)


def _iter_object_entries(data: Any) -> Iterable[Any]:
    if isinstance(data, Mapping):
        if isinstance(data.get("objects"), list):
            yield from data["objects"]
            return
        if isinstance(data.get("candidates"), list):
            yield from data["candidates"]
            return
        yield from data.values()
        return
    if isinstance(data, list):
        yield from data


def _object_id(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        value = entry.get("name", entry.get("object_id", entry.get("id")))
        if value is not None:
            return str(value)
    return str(entry)


def _object_mesh_path(entry: Any, object_dir: Path) -> Path:
    if isinstance(entry, Mapping):
        for key in ("mesh_path", "obj_path", "object_mesh_path", "path"):
            value = entry.get(key)
            if value:
                path = Path(str(value)).expanduser()
                return path if path.is_absolute() else (object_dir / path).resolve()
    object_id = _object_id(entry)
    mesh_stem = object_id.rsplit("-", 1)[0]
    return object_dir / f"{mesh_stem}.obj"


def _scaled_centered_extent(vertices, scale):
    import numpy as np

    scale_arr = np.asarray(scale, dtype=np.float64)
    if scale_arr.ndim == 0:
        scale_arr = np.full(3, float(scale_arr), dtype=np.float64)
    elif scale_arr.size == 1:
        scale_arr = np.full(3, float(scale_arr.reshape(-1)[0]), dtype=np.float64)
    elif scale_arr.shape != (3,):
        raise ValueError(f"scale must be scalar or shape (3,), got {scale_arr.shape}")
    scaled = np.asarray(vertices, dtype=np.float64) * scale_arr.reshape(1, 3)
    bbox_min = scaled.min(axis=0)
    bbox_max = scaled.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    centered = scaled - center.reshape(1, 3)
    extent = centered.max(axis=0) - centered.min(axis=0)
    return centered, extent


def _identity_pose9d(torch):
    return torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def main() -> int:
    started = time.time()
    adapter = None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _force_headless_argv()

    _log(f"python={sys.executable}")
    _log(f"cwd={os.getcwd()}")
    _log(f"config={CONFIG_PATH}")
    _log(f"argv={sys.argv}")
    for key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "DISPLAY", "ISAACSIM_PATH", "EXP_PATH"):
        _log(f"env {key}={_env(key)!r}")

    try:
        _log("load config")
        from utils.config.loader import load_exp_cfg
        from utils.config.paths import load_project_paths, require_path
        from utils.experiment.effective_paths import apply_experiment_path_overrides

        cfg = load_exp_cfg(CONFIG_PATH)
        paths_yaml = Path(cfg.paths_yaml).expanduser()
        if not paths_yaml.is_absolute():
            paths_yaml = ROOT / paths_yaml
        paths = apply_experiment_path_overrides(
            cfg,
            load_project_paths(paths_yaml),
            stage="contact_gen",
        )

        _log("resolve paths")
        objects_json = require_path(paths, "objects.candidates_json")
        objects_obj_dir = require_path(paths, "objects.obj_dir")
        tools_selected_json = require_path(paths, "tools.tools_selected_json")
        tools_adjusted_json = require_path(paths, "tools.tools_adjusted_json")
        meshdata_adjusted_root = require_path(paths, "tools.meshdata_adjusted_root")
        _log(f"objects_json={objects_json}")
        _log(f"objects_obj_dir={objects_obj_dir}")
        _log(f"tools_selected_json={tools_selected_json}")
        _log(f"tools_adjusted_json={tools_adjusted_json}")
        _log(f"meshdata_adjusted_root={meshdata_adjusted_root}")

        _log("load object/tool manifests")
        from utils.assets import load_selected_tool_ids, load_tool_asset
        from utils.io import read_json

        objects = list(_iter_object_entries(read_json(objects_json)))
        if not objects:
            raise RuntimeError(f"No objects found in {objects_json}")
        object_entry = objects[0]
        object_id = _object_id(object_entry)
        object_mesh = _object_mesh_path(object_entry, objects_obj_dir)
        if not object_mesh.exists():
            raise RuntimeError(f"Object mesh does not exist: {object_mesh}")

        tool_id = load_selected_tool_ids(tools_selected_json)[0]
        tool_paths = SimpleNamespace(
            meshdata_adjusted_root=meshdata_adjusted_root,
            tools_adjusted_json=tools_adjusted_json,
            tools_selected_json=tools_selected_json,
        )
        tool_asset = load_tool_asset(
            tool_id,
            tool_paths,
            scale_xyz=cfg.general.tool_mount.scale_xyz,
            require_mesh=True,
        )
        _log(f"selected object={object_id} mesh={object_mesh}")
        _log(f"selected tool={tool_id} mesh={tool_asset.mesh_path}")

        _log("mesh load")
        import numpy as np
        import torch

        from utils.contact.isaac import IsaacSimAdapter
        from utils.contact.stabilize import PhysicsRunConfig
        from utils.geometry.mesh_io import load_mesh_vertices_faces

        object_scale = float(cfg.contact_gen.object_scale_range[0])
        tool_scale_xyz = tuple(float(x) for x in cfg.general.tool_mount.scale_xyz)
        object_vertices, _object_faces = load_mesh_vertices_faces(object_mesh, process=False)
        tool_vertices, _tool_faces = load_mesh_vertices_faces(tool_asset.mesh_path, process=False)
        object_centered, object_extent = _scaled_centered_extent(object_vertices, object_scale)
        _tool_centered, tool_extent = _scaled_centered_extent(tool_vertices, tool_scale_xyz)

        object_center_E = np.array(
            [0.0, 0.0, -float(object_centered[:, 2].min()) + 1.0e-3],
            dtype=np.float64,
        )
        tool_translation_O = np.array(
            [0.5 * float(object_extent[0] + tool_extent[0]) + 0.02, 0.0, 0.02],
            dtype=np.float64,
        )
        _log(
            "candidate pose "
            f"object_center_E={object_center_E.tolist()} tool_translation_O={tool_translation_O.tolist()}"
        )

        candidate = {
            "object_rotation_E": torch.eye(3, dtype=torch.float32),
            "object_bbox_center_E": torch.as_tensor(object_center_E, dtype=torch.float32),
            "tool_translation_O": torch.as_tensor(tool_translation_O, dtype=torch.float32),
            "tool_rotation_O": torch.eye(3, dtype=torch.float32),
            "contact_point_O": torch.zeros(3, dtype=torch.float32),
        }
        physical_props = {
            "object_mass": torch.tensor(0.1, dtype=torch.float32),
            "tool_mass": torch.tensor(0.1, dtype=torch.float32),
            "object_friction": torch.tensor(0.8, dtype=torch.float32),
            "tool_friction": torch.tensor(0.8, dtype=torch.float32),
            "ground_friction": torch.tensor(0.8, dtype=torch.float32),
        }
        run_cfg = PhysicsRunConfig(
            t_stabilize=1,
            t_postcontact=1,
            runner="isaac",
            post_delta_seed=int(cfg.general.seed),
            post_delta_translation_min=(0.0, 0.0, 0.0),
            post_delta_translation_max=(0.0, 0.0, 0.0),
            post_delta_rotation_max_rad=0.0,
            post_tool_reach_translation_eps=float(cfg.contact_gen.physics.post_tool_reach_translation_eps),
            post_tool_reach_rotation_eps_rad=float(cfg.contact_gen.physics.post_tool_reach_rotation_eps_rad),
            post_object_table_z_min=float(cfg.contact_gen.physics.post_object_table_z_min),
            object_mesh_path=str(object_mesh),
            tool_mesh_path=str(tool_asset.mesh_path),
            object_scale=object_scale,
            tool_scale_xyz=tool_scale_xyz,
            debug_dir=str(ROOT / "artifacts" / "debug_contact_isaac_candidate"),
            headless=True,
            close_after_run=False,
        )
        candidate_batch = {key: value.unsqueeze(0) for key, value in candidate.items()}
        physical_batch = {key: value.unsqueeze(0) for key, value in physical_props.items()}

        _log("instantiate IsaacSimAdapter")
        adapter = IsaacSimAdapter(
            headless=True,
            debug_log=lambda message: _log(f"adapter: {message}"),
        )
        _log("run_batch start")
        result = adapter.run_batch(
            candidates=candidate_batch,
            physical_props=physical_batch,
            cfg=run_cfg,
            commanded_tool_delta_pose9d_O=_identity_pose9d(torch).unsqueeze(0),
        )[0]
        _log(
            "run_batch done "
            f"success={result.success} status={result.status} "
            f"in_contact={result.stabilized_in_contact} "
            f"contact_count={result.stabilized_contact_count} "
            f"contact_impulse_norm={result.stabilized_contact_impulse_norm} "
            f"stabilize_steps={result.stabilize_steps} postcontact_steps={result.postcontact_steps}"
        )
        if result.stage_usd_path:
            _log(f"debug stage={result.stage_usd_path}")
        if result.debug_json_path:
            _log(f"debug json={result.debug_json_path}")
        if result.metrics:
            _log(f"metrics keys={sorted(result.metrics.keys())}")
        _log(f"diagnostic complete after {time.time() - started:.2f}s")
        return 0
    except BaseException:
        _log("diagnostic failed")
        traceback.print_exc()
        return 1
    finally:
        if adapter is not None:
            _log("close")
            try:
                adapter.close()
            except BaseException:
                traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render one pre/post contact-case page per tool family."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Optional experiment config exposing EXP_CFG.")
    parser.add_argument("--data-dir", default="", help="Contact artifact directory. Overrides config runtime.data_dir.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-tool PNG outputs.")
    parser.add_argument("--contacts-per-tool", type=int, default=36, help="Number of contact cases to sample per tool family.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for deterministic per-tool sampling.")
    parser.add_argument("--max-tools", type=int, default=0, help="Optional limit on number of tool families to render.")
    parser.add_argument("--tool-filter", default="", help="Optional substring filter for tool family names.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--max-faces", type=int, default=10000, help="Max faces per mesh to render. Use 0 for all.")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log(
        "start "
        f"config={args.config or '<none>'} data_dir={args.data_dir or '<from config>'} "
        f"contacts_per_tool={args.contacts_per_tool} output_dir={args.output_dir}"
    )

    import torch

    from pretrain.dataset import NewPretrainDataset
    from contact_generation.visualize_post_patch import (
        _compute_post_viz,
        _draw_post_grid_frame,
        _grid_rows_from_items,
    )

    runtime = _load_runtime_from_config(args.config) if args.config else None
    data_dir = Path(args.data_dir or getattr(runtime, "data_dir", "")).expanduser()
    if not data_dir:
        raise ValueError("Provide --data-dir or --config")
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    groups = _collect_tool_pt_groups(data_dir)
    if args.tool_filter:
        groups = {name: paths for name, paths in groups.items() if args.tool_filter in name}
    tool_names = sorted(groups)
    if args.max_tools > 0:
        tool_names = tool_names[: int(args.max_tools)]
    _log(f"found tool_families={len(tool_names)}")

    dataset_kwargs = _dataset_kwargs(runtime)
    for tool_i, tool_name in enumerate(tool_names, start=1):
        pt_files = groups[tool_name]
        _log(f"tool {tool_i}/{len(tool_names)} {tool_name}: pt_files={len(pt_files)}")
        selected_cases = _sample_cases_for_tool(
            tool_name,
            pt_files,
            contacts_per_tool=max(1, int(args.contacts_per_tool)),
            seed=int(args.seed),
            torch_module=torch,
        )
        if not selected_cases:
            _log(f"tool {tool_name}: no valid contact cases, skipping")
            continue
        selected_paths = _unique_paths_in_order(path for path, _contact_i in selected_cases)
        dataset = NewPretrainDataset(selected_paths, augment=False, **dataset_kwargs)
        dataset._index = [(str(path), int(contact_i)) for path, contact_i in selected_cases]

        viz_items = []
        for sample_i in range(len(dataset)):
            item = dataset[sample_i]
            batch = _collate_one(item, torch)
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                viz_items.append(_compute_post_viz(None, tensor_batch, predict=False))
        _write_tool_images(
            tool_name=tool_name,
            viz_items=viz_items,
            output_dir=output_dir,
            draw_frame=_draw_post_grid_frame,
            grid_rows_from_items=_grid_rows_from_items,
            max_faces=int(args.max_faces),
            elev=float(args.elev),
            azim=float(args.azim),
        )
    _log(f"wrote outputs under {output_dir}")
    return 0


def _load_runtime_from_config(config_path: str):
    from pretrain.train import build_runtime_config
    from utils.artifacts.resolver import resolve_artifacts
    from utils.config.loader import load_exp_cfg
    from utils.config.paths import load_project_paths
    from utils.experiment.effective_paths import apply_experiment_path_overrides

    cfg = load_exp_cfg(config_path)
    paths = apply_experiment_path_overrides(
        cfg,
        load_project_paths(cfg.paths_yaml),
        stage="contact_gen",
    )
    artifacts = resolve_artifacts(cfg)
    pretrain_ref = _stage_ref(artifacts, "pretrain")
    return build_runtime_config(cfg, paths, pretrain_ref.directory)


def _dataset_kwargs(runtime) -> dict[str, Any]:
    if runtime is None:
        return {}
    return {
        "require_movement": runtime.task == "sdf-diff",
        "num_points": runtime.num_pts,
        "num_precontact_steps": runtime.num_precontact_steps,
        "allow_mock_physics": runtime.allow_mock_physics,
        "noise_max_trans": runtime.noise_max_trans,
        "noise_max_rot_deg": runtime.noise_max_rot_deg,
        "noise_max_retries": runtime.noise_max_retries,
        "floor_eps": runtime.floor_eps,
        "validation_seed": runtime.validation_seed,
        "denoise_target_mode": runtime.denoise_target_mode,
        "tool_mesh_contract": runtime.tool_mesh_contract,
    }


def _collect_tool_pt_groups(data_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    variant_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    _log(f"scanning variant_dirs={len(variant_dirs)}")
    for variant_dir in variant_dirs:
        tool_name = _tool_family_name(variant_dir.name)
        files = [path for path in sorted(variant_dir.glob("*.pt")) if _is_contact_pt(path)]
        if files:
            groups.setdefault(tool_name, []).extend(files)
    return groups


def _tool_family_name(variant_name: str) -> str:
    return re.sub(r"_var_\d+$", "", variant_name)


def _is_contact_pt(path: Path) -> bool:
    blocked_suffixes = (".candidate.pt", ".physics_debug.pt", ".stabilized_success.pt", ".stabilized.pt")
    text = str(path)
    return path.suffix == ".pt" and not any(text.endswith(suffix) for suffix in blocked_suffixes)


def _sample_cases_for_tool(
    tool_name: str,
    pt_files: list[Path],
    *,
    contacts_per_tool: int,
    seed: int,
    torch_module,
) -> list[tuple[Path, int]]:
    cases: list[tuple[Path, int]] = []
    for pt_path in pt_files:
        try:
            data = torch_module.load(pt_path, map_location="cpu", weights_only=False)
            n = int(data.get("num_contacts", 0)) if isinstance(data, dict) else 0
        except Exception as exc:
            _log(f"warning: failed to inspect {pt_path}: {exc}")
            continue
        cases.extend((pt_path, contact_i) for contact_i in range(max(0, n)))
    rng = random.Random(seed + _stable_int(tool_name))
    rng.shuffle(cases)
    return cases[: min(int(contacts_per_tool), len(cases))]


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _unique_paths_in_order(paths) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        text = str(path)
        if text not in seen:
            seen.add(text)
            unique.append(path)
    return unique


def _collate_one(item: dict[str, Any], torch_module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    list_keys = {"object_mesh_vertices", "object_mesh_faces", "tool_mesh_vertices", "tool_mesh_faces"}
    for key, value in item.items():
        if key in list_keys or not isinstance(value, torch_module.Tensor):
            out[key] = [value]
        else:
            out[key] = value.unsqueeze(0)
    return out


def _write_tool_images(
    *,
    tool_name: str,
    viz_items: list[dict[str, Any]],
    output_dir: Path,
    draw_frame,
    grid_rows_from_items,
    max_faces: int,
    elev: float,
    azim: float,
) -> None:
    if not viz_items:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = grid_rows_from_items(viz_items)
    n_rows = len(rows)
    contacts_per_row = max(len(row) for row in rows)
    fig_w = max(0.68 * contacts_per_row, 4.4)
    fig_h = max(0.68 * n_rows, 1.0)
    for state, show_final in (("pre", False), ("post", True)):
        output = output_dir / f"{tool_name}_{state}.png"
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
        try:
            draw_frame(
                fig,
                rows,
                n_axis_cols=contacts_per_row,
                include_pred=False,
                show_final=show_final,
                max_faces=max_faces,
                elev=elev,
                azim=azim,
                frame_i=0,
                total_frames=1,
                title=f"{tool_name} / {state}",
            )
            fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
            _log(f"wrote {output}")
        finally:
            plt.close(fig)


def _stage_ref(artifacts, stage: str):
    for ref in artifacts.stages:
        if ref.stage == stage:
            return ref
    raise RuntimeError(f"Experiment has no {stage!r} stage")


def _log(message: str) -> None:
    print(f"[visualize_contacts_by_tool] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

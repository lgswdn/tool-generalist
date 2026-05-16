"""Experiment-config-driven batch contact generation.

This module is intentionally light to import.  The torch/trimesh/Kaolin-backed
generator is imported only when a pair is actually executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from configs.config_contact_gen import ContactGenCfg
from configs.config_exp import ExpCfg
from utils.assets import load_selected_tool_ids, load_tool_asset
from utils.contact.paths import ContactPaths
from utils.config.paths import ProjectPaths
from utils.io import hash_json, read_json, to_plain_data


GeneratorApi = tuple[type, Callable[..., Any]]


def _progress(iterable: Iterable[Any], *, total: int, desc: str, position: int = 0):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        position=position,
        leave=True,
        dynamic_ncols=True,
    )


@dataclass(frozen=True)
class ContactGenerationResult:
    artifact_dir: Path
    num_pairs: int
    num_poses: int
    ok: int
    fail: int
    skipped: int

    @property
    def total(self) -> int:
        return self.num_pairs * self.num_poses


def run_contact_generation(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> ContactGenerationResult:
    try:
        return _run_contact_generation(exp_cfg, paths, artifact_dir)
    except KeyboardInterrupt:
        _log("[INTERRUPT] contact generation interrupted")
        raise


def _run_contact_generation(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> ContactGenerationResult:
    contact_cfg = exp_cfg.contact_gen
    _log(f"[START] contact generation config={contact_cfg.name}")
    _log("[LOAD] resolving contact generation paths")
    contact_paths = contact_paths_from_project_paths(paths)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[PATH] artifact_dir={out_dir}")
    _log(f"[LOAD] objects manifest={contact_paths.objects_candidates_json}")
    objects = read_json(contact_paths.objects_candidates_json)
    _log(f"[LOAD] tools selected manifest={contact_paths.tools_selected_json}")
    tools = load_selected_tool_ids(contact_paths.tools_selected_json)
    _log(f"[LOAD] loaded objects={len(objects)} tools={len(tools)}")
    config_name = f"{exp_cfg.general.name}_{contact_cfg.name}"
    config_hash = contact_config_hash(exp_cfg)
    _log("[PAIR] building object/tool pairs")
    pairs = build_pairs(
        objects,
        tools,
        contact_paths.objects_obj_dir,
        contact_paths,
        exp_cfg.general.tool_mount.scale_xyz,
    )
    _log(f"[PAIR] available pairs={len(pairs)}")
    selected_pairs = sample_pairs(pairs, contact_cfg.num_pairs, exp_cfg.general.seed)
    _log(
        "[PAIR] selected_pairs="
        f"{len(selected_pairs)} requested={contact_cfg.num_pairs} "
        f"num_poses={contact_cfg.num_object_poses}"
    )
    if not selected_pairs:
        _log("[DONE] contact generation has no selected pairs")
        return ContactGenerationResult(
            artifact_dir=out_dir,
            num_pairs=0,
            num_poses=contact_cfg.num_object_poses,
            ok=0,
            fail=0,
            skipped=0,
        )

    requested_gpus = int(getattr(exp_cfg, "num_gpus", 0))
    visible_gpus = visible_cuda_device_indices(requested_count=requested_gpus)
    configured_workers = int(contact_cfg.physics.num_workers)
    worker_request = requested_gpus if requested_gpus > 0 else configured_workers
    num_workers = max(1, min(worker_request, len(visible_gpus)))
    if contact_cfg.visualization.enabled and num_workers > 1:
        raise ValueError(
            "Contact visualization writes headless videos from one Isaac process; "
            "set EXP_CFG.num_gpus = 1 when "
            "EXP_CFG.contact_gen.visualization.enabled = True."
        )
    gpus = visible_gpus[:num_workers]
    _log(
        "[WORKER] visible GPU worker indices="
        f"{visible_gpus} requested_num_gpus={requested_gpus} "
        f"compat_configured_workers={configured_workers} active_workers={len(gpus)}"
    )
    subsets = [[] for _ in gpus]
    for index, pair in enumerate(selected_pairs):
        subsets[index % len(gpus)].append(pair)
    for index, subset in enumerate(subsets):
        _log(f"[WORKER] assignment worker={index} gpu={gpus[index]} pairs={len(subset)}")

    physics_options = physics_options_from_config(
        contact_cfg,
        env_spacing=float(exp_cfg.rl.env.env_spacing),
    )
    skip_existing = bool(contact_cfg.skip_existing and not contact_cfg.regenerate)
    _log(f"[POLICY] skip_existing={skip_existing} regenerate={contact_cfg.regenerate}")
    worker_args = [
        (
            subsets[index],
            out_dir,
            contact_paths.tools_adjusted_json,
            exp_cfg.general.tool_mount.scale_xyz,
            gpu,
            contact_cfg,
            skip_existing,
            contact_cfg.num_object_poses,
            physics_options,
            exp_cfg.general.seed + index,
            config_name,
            config_hash,
        )
        for index, gpu in enumerate(gpus)
    ]
    _log("[PHASE] geometry candidate generation")
    geometry_ok, geometry_fail, _geometry_skipped = _run_worker_pool(
        [("geometry", *args) for args in worker_args],
        worker_fn=phase_worker,
    )
    _log(
        "[PHASE-DONE] geometry "
        f"ready={geometry_ok} fail={geometry_fail}"
    )

    _log("[PHASE] contact stabilization")
    stabilize_ok, stabilize_fail, _stabilize_skipped = _run_worker_pool(
        [("stabilize", *args) for args in worker_args],
        worker_fn=phase_worker,
    )
    _log(
        "[PHASE-DONE] stabilize "
        f"ready={stabilize_ok} fail={stabilize_fail}"
    )

    _log("[PHASE] postcontact rollout")
    ok, postcontact_fail, skipped = _run_worker_pool(
        [("postcontact", *args) for args in worker_args],
        worker_fn=phase_worker,
    )
    fail = geometry_fail + stabilize_fail + postcontact_fail

    result = ContactGenerationResult(
        artifact_dir=out_dir,
        num_pairs=len(selected_pairs),
        num_poses=contact_cfg.num_object_poses,
        ok=ok,
        fail=fail,
        skipped=skipped,
    )
    _log(
        "[DONE] contact generation "
        f"num_pairs={result.num_pairs} num_poses={result.num_poses} "
        f"ok={result.ok} fail={result.fail} skipped={result.skipped} "
        f"artifact_dir={result.artifact_dir}"
    )
    return result


def _run_worker_pool(worker_args, *, worker_fn=None) -> tuple[int, int, int]:
    if worker_fn is None:
        worker_fn = worker
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(len(worker_args))
    async_results = []
    try:
        for args in worker_args:
            async_results.append(pool.apply_async(worker_fn, args))
        while True:
            if all(result.ready() for result in async_results):
                break
            _log(
                "[HEARTBEAT] contact generation workers "
                f"done={sum(result.ready() for result in async_results)}/{len(async_results)}"
            )
            time.sleep(5.0)
        results = [result.get() for result in async_results]
        # Isaac/Kit teardown can hang after the worker has already returned its
        # result.  Once all results are collected, stop worker processes instead
        # of waiting for graceful Omniverse plugin unload.
        pool.terminate()
        pool.join()
    except KeyboardInterrupt:
        _log("[INTERRUPT] terminating contact generation worker pool")
        pool.terminate()
        pool.join()
        raise
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    ok = sum(result[0] for result in results)
    fail = sum(result[1] for result in results)
    skipped = sum(result[2] for result in results)
    return ok, fail, skipped


def contact_paths_from_project_paths(paths: ProjectPaths) -> ContactPaths:
    return ContactPaths(
        objects_candidates_json=_required_path(paths, "objects.candidates_json"),
        objects_usd_dir=_required_path(paths, "objects.usd_dir"),
        objects_obj_dir=_required_path(paths, "objects.obj_dir"),
        meshdata_adjusted_root=_required_path(paths, "tools.meshdata_adjusted_root"),
        objects_usd_root=paths.get("tools.objects_usd_root"),
        robots_usd_root=paths.get("tools.robots_usd_root"),
        tools_adjusted_json=_required_path(paths, "tools.tools_adjusted_json"),
        tools_selected_json=_required_path(paths, "tools.tools_selected_json"),
        franka_src_root=paths.get("tools.franka_src_root"),
        source_yaml=paths.source_yaml,
    )


def contact_config_hash(exp_cfg: ExpCfg) -> str:
    contact_payload = to_plain_data(exp_cfg.contact_gen)
    physics = dict(contact_payload.get("physics") or {})
    physics.pop("num_workers", None)
    contact_payload["physics"] = physics
    return hash_json(
        {
            "general": asdict(exp_cfg.general),
            "contact_gen": contact_payload,
        }
    )


def physics_options_from_config(contact_cfg: ContactGenCfg, *, env_spacing: float) -> dict[str, Any]:
    physics = contact_cfg.physics
    visualization = contact_cfg.visualization
    return {
        "physics_runner": physics.runner,
        "t_stabilize": physics.t_stabilize,
        "t_postcontact": physics.t_postcontact,
        "object_mass_range": tuple(physics.object_mass_range),
        "tool_mass_range": tuple(physics.tool_mass_range),
        "object_friction_range": tuple(physics.object_friction_range),
        "tool_friction_range": tuple(physics.tool_friction_range),
        "ground_friction_range": tuple(physics.ground_friction_range),
        "post_delta_seed": physics.post_delta_seed,
        "post_delta_translation_min": tuple(physics.post_delta_translation_min),
        "post_delta_translation_max": tuple(physics.post_delta_translation_max),
        "post_delta_rotation_max_rad": physics.post_delta_rotation_max_rad,
        "post_tool_reach_translation_eps": physics.post_tool_reach_translation_eps,
        "post_tool_reach_rotation_eps_rad": physics.post_tool_reach_rotation_eps_rad,
        "post_object_table_z_min": physics.post_object_table_z_min,
        "unsigned_distance_accept_eps": float(physics.unsigned_distance_accept_eps),
        "env_spacing": float(env_spacing),
        "visualization_enabled": bool(visualization.enabled),
        "visualization_stabilization_picture": bool(visualization.enabled and visualization.stabilization_picture),
        "visualization_stabilization_picture_num": int(visualization.stabilization_picture_num),
        "visualization_postcontact_video": bool(visualization.enabled and visualization.postcontact_video),
        "visualization_postcontact_video_num": int(visualization.postcontact_video_num),
        "visualization_video_dir": visualization.video_dir or "",
        "visualization_picture_dir": visualization.picture_dir or "",
        "visualization_video_width": int(visualization.video_width),
        "visualization_video_height": int(visualization.video_height),
        "visualization_video_fps": int(visualization.video_fps),
        "visualization_camera_pos": tuple(visualization.camera_pos),
        "visualization_camera_target": tuple(visualization.camera_target),
        "visualization_max_candidates": int(visualization.max_visualized_candidates),
        "debug_dir": "",
        "headless": True,
        "close_after_run": False,
    }


def build_pairs(objects, tools, obj_mesh_dir, contact_paths, scale_xyz):
    pairs = []
    missing_obj = missing_tool = 0
    _log(
        "[PAIR] scanning "
        f"tools={len(tools)} objects={len(objects)} obj_mesh_dir={obj_mesh_dir}"
    )
    for tool in tools:
        try:
            tool_asset = load_tool_asset(tool, contact_paths, scale_xyz=scale_xyz, require_mesh=True)
        except Exception as exc:
            missing_tool += 1
            _log(f"[WARN] tool asset '{tool}' skipped: {exc}")
            continue
        tool_path = tool_asset.mesh_path
        for obj_entry in objects:
            obj_name = _object_id(obj_entry)
            mesh_stem = obj_name.rsplit("-", 1)[0]
            obj_path = Path(obj_mesh_dir) / f"{mesh_stem}.obj"
            if not obj_path.exists():
                missing_obj += 1
                continue
            pairs.append((str(tool_path), str(obj_path), tool, obj_name, tool_asset))
    if missing_tool:
        _log(f"[WARN] {missing_tool} tool mesh(es) not found.")
    if missing_obj:
        _log(f"[WARN] {missing_obj} object mesh(es) not found.")
    _log(
        "[PAIR] build complete "
        f"pairs={len(pairs)} missing_tools={missing_tool} missing_objects={missing_obj}"
    )
    return pairs


def sample_pairs(pairs, num_pairs, seed):
    rng = random.Random(seed)
    if num_pairs <= 0 or num_pairs >= len(pairs):
        return pairs
    return rng.sample(pairs, num_pairs)


def output_path(out_dir, tool_name, obj_name, pose_idx, num_poses):
    if num_poses == 1:
        return Path(out_dir) / tool_name / f"{obj_name}.pt"
    return Path(out_dir) / tool_name / f"{obj_name}_pose{pose_idx}.pt"


def candidate_artifact_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_suffix(path.suffix + ".candidate.pt")


def stabilized_artifact_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_suffix(path.suffix + ".stabilized_success.pt")


def run_pair(
    tool_path,
    obj_path,
    tool_name,
    obj_name,
    out_dir,
    tools_meta,
    scale_xyz,
    gpu,
    contact_cfg: ContactGenCfg,
    pose_idx=0,
    num_poses=1,
    seed=42,
    physics_options=None,
    config_name="contact_generation",
    config_hash="",
    generator_api: Optional[GeneratorApi] = None,
    physics_runner: Any = None,
    phase: str = "postcontact",
) -> bool:
    pt_file = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
    pt_file.parent.mkdir(parents=True, exist_ok=True)
    physics_options = {} if physics_options is None else dict(physics_options)
    debug_root = physics_options.pop("debug_dir", "")
    debug_dir = ""
    if debug_root:
        debug_dir = str(Path(debug_root) / tool_name / f"{obj_name}_pose{pose_idx}")
    video_root = physics_options.pop("visualization_video_dir", "")
    picture_root = physics_options.pop("visualization_picture_dir", "")
    visualization_video_dir = ""
    visualization_picture_dir = ""
    if physics_options.get("visualization_postcontact_video"):
        if video_root:
            visualization_video_dir = str(Path(video_root).expanduser() / tool_name / f"{obj_name}_pose{pose_idx}")
        else:
            visualization_video_dir = str(pt_file.parent / "videos" / f"{pt_file.stem}")
    if physics_options.get("visualization_stabilization_picture"):
        if picture_root:
            visualization_picture_dir = str(Path(picture_root).expanduser() / tool_name / f"{obj_name}_pose{pose_idx}")
        else:
            visualization_picture_dir = str(pt_file.parent / "pictures" / f"{pt_file.stem}")

    try:
        _log(
            "[PAIR-START] "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses} "
            f"gpu={gpu} output={pt_file}"
        )
        _log(
            "[IMPORT] loading contact generator phase "
            f"phase={phase} "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses}"
        )
        optimize_config, optimize_main = generator_api or _load_generator_api(phase)
        _log(
            "[CONFIG] building generator config "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses}"
        )
        cfg = optimize_config(
            object_mesh_path=obj_path,
            tool_mesh_path=tool_path,
            output_path=str(pt_file),
            tools_json_path=str(tools_meta) if tools_meta and Path(tools_meta).exists() else "",
            object_id=obj_name,
            tool_id=tool_name,
            config_name=config_name,
            config_hash=config_hash,
            tool_scale=float(scale_xyz[0]),
            tool_scale_xyz=tuple(scale_xyz),
            object_scale_range=tuple(contact_cfg.object_scale_range),
            num_tool_surface_pts=contact_cfg.num_surface_pts,
            device=f"cuda:{gpu}",
            seed=seed,
            B=contact_cfg.B,
            M=contact_cfg.M,
            K=contact_cfg.num_surface_pts,
            sdf_grid_res=contact_cfg.sdf_grid_res,
            sdf_padding=contact_cfg.sdf_padding,
            chunk_B=contact_cfg.chunk_B,
            upright_threshold=contact_cfg.upright_threshold,
            epsilon=contact_cfg.epsilon,
            floor_eps=contact_cfg.floor_eps,
            penetration_eps=contact_cfg.penetration_eps,
            contact_mode_prob=dict(contact_cfg.contact_mode_prob),
            debug_dir=debug_dir,
            visualization_video_dir=visualization_video_dir,
            visualization_picture_dir=visualization_picture_dir,
            **physics_options,
        )
        _log(
            "[CALL] running contact generator phase "
            f"phase={phase} "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses}"
        )
        result = _call_optimize_main(optimize_main, cfg, physics_runner=physics_runner)
        if isinstance(result, int) and result <= 0:
            _log(
                "[PAIR-FAIL] "
                f"phase={phase} tool={tool_name} object={obj_name} "
                f"pose={pose_idx + 1}/{num_poses} output={pt_file} no usable outputs"
            )
            return False
    except KeyboardInterrupt:
        _log(
            "[INTERRUPT] pair interrupted "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses} output={pt_file}"
        )
        raise
    except Exception as exc:
        _log(
            "[PAIR-FAIL] "
            f"phase={phase} "
            f"tool={tool_name} object={obj_name} pose={pose_idx + 1}/{num_poses} "
            f"output={pt_file} error={exc}"
        )
        return False
    _log(
        "[PAIR-DONE] "
        f"phase={phase} tool={tool_name} object={obj_name} "
        f"pose={pose_idx + 1}/{num_poses} output={pt_file}"
    )
    return True


def worker(
    pairs_subset,
    out_dir,
    tools_meta,
    scale_xyz,
    gpu,
    contact_cfg,
    skip_existing,
    num_poses,
    physics_options=None,
    seed=42,
    config_name="contact_generation",
    config_hash="",
):
    ok = fail = skipped = 0
    rng = random.Random(seed)
    total_jobs = len(pairs_subset) * int(num_poses)
    _log(f"[WORKER-START] gpu={gpu} pairs={len(pairs_subset)} poses={num_poses} jobs={total_jobs}")
    jobs = []
    for pair_index, (tool_path, obj_path, tool_name, obj_name, _tool_asset) in enumerate(
        pairs_subset, start=1
    ):
        _log(
            "[WORKER] "
            f"gpu={gpu} pair={pair_index}/{len(pairs_subset)} "
            f"tool={tool_name} object={obj_name}"
        )
        for pose_idx in range(num_poses):
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            if skip_existing and pt.exists():
                skipped += 1
                _log(
                    "[PAIR-SKIP] "
                    f"gpu={gpu} tool={tool_name} object={obj_name} "
                    f"pose={pose_idx + 1}/{num_poses} output={pt}"
                )
                continue
            jobs.append(
                (
                    tool_path,
                    obj_path,
                    tool_name,
                    obj_name,
                    pose_idx,
                    rng.randint(0, 2**31 - 1),
                )
            )

    geometry_ready = []
    try:
        _log(f"[WORKER-GEOMETRY-START] gpu={gpu} jobs={len(jobs)}")
        for job in _progress(jobs, total=len(jobs), desc=f"gpu{gpu} geometry", position=int(gpu)):
            tool_path, obj_path, tool_name, obj_name, pose_idx, pose_seed = job
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            if skip_existing and candidate_artifact_path(pt).exists():
                _log(f"[GEOMETRY-SKIP] gpu={gpu} candidate={candidate_artifact_path(pt)}")
                geometry_ready.append(job)
                continue
            if run_pair(
                tool_path,
                obj_path,
                tool_name,
                obj_name,
                out_dir,
                tools_meta,
                scale_xyz,
                gpu,
                contact_cfg,
                pose_idx,
                num_poses,
                pose_seed,
                physics_options,
                config_name,
                config_hash,
                phase="geometry",
            ):
                geometry_ready.append(job)
            else:
                fail += 1

        stabilize_ready = []
        _log(f"[WORKER-STABILIZE-START] gpu={gpu} jobs={len(geometry_ready)}")
        physics_runner = _load_physics_runner(contact_cfg.physics.runner)
        _log(f"[WORKER] gpu={gpu} stabilize physics_runner={contact_cfg.physics.runner} initialized")
        try:
            for job in _progress(
                geometry_ready,
                total=len(geometry_ready),
                desc=f"gpu{gpu} stabilize",
                position=int(gpu),
            ):
                tool_path, obj_path, tool_name, obj_name, pose_idx, pose_seed = job
                pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
                if skip_existing and stabilized_artifact_path(pt).exists():
                    _log(f"[STABILIZE-SKIP] gpu={gpu} stabilized={stabilized_artifact_path(pt)}")
                    stabilize_ready.append(job)
                    continue
                if run_pair(
                    tool_path,
                    obj_path,
                    tool_name,
                    obj_name,
                    out_dir,
                    tools_meta,
                    scale_xyz,
                    gpu,
                    contact_cfg,
                    pose_idx,
                    num_poses,
                    pose_seed,
                    physics_options,
                    config_name,
                    config_hash,
                    physics_runner=physics_runner,
                    phase="stabilize",
                ):
                    stabilize_ready.append(job)
                else:
                    fail += 1
        finally:
            _close_physics_runner(physics_runner, gpu)

        _log(f"[WORKER-POSTCONTACT-START] gpu={gpu} jobs={len(stabilize_ready)}")
        physics_runner = _load_physics_runner(contact_cfg.physics.runner)
        _log(f"[WORKER] gpu={gpu} postcontact physics_runner={contact_cfg.physics.runner} initialized")
        try:
            for job in _progress(
                stabilize_ready,
                total=len(stabilize_ready),
                desc=f"gpu{gpu} postcontact",
                position=int(gpu),
            ):
                tool_path, obj_path, tool_name, obj_name, pose_idx, pose_seed = job
                success = run_pair(
                    tool_path,
                    obj_path,
                    tool_name,
                    obj_name,
                    out_dir,
                    tools_meta,
                    scale_xyz,
                    gpu,
                    contact_cfg,
                    pose_idx,
                    num_poses,
                    pose_seed,
                    physics_options,
                    config_name,
                    config_hash,
                    physics_runner=physics_runner,
                    phase="postcontact",
                )
                if success:
                    ok += 1
                else:
                    fail += 1
        finally:
            _close_physics_runner(physics_runner, gpu)

        _log(f"[WORKER-DONE] gpu={gpu} ok={ok} fail={fail} skipped={skipped}")
        return ok, fail, skipped
    except KeyboardInterrupt:
        _log(f"[INTERRUPT] worker interrupted gpu={gpu} ok={ok} fail={fail} skipped={skipped}")
        raise


def phase_worker(
    phase,
    pairs_subset,
    out_dir,
    tools_meta,
    scale_xyz,
    gpu,
    contact_cfg,
    skip_existing,
    num_poses,
    physics_options=None,
    seed=42,
    config_name="contact_generation",
    config_hash="",
):
    ok = fail = skipped = 0
    jobs = _build_jobs(pairs_subset, out_dir, num_poses, seed, skip_existing=False)
    _log(f"[PHASE-WORKER-START] phase={phase} gpu={gpu} jobs={len(jobs)}")
    physics_runner = None
    try:
        if phase in {"stabilize", "postcontact"}:
            physics_runner = _load_physics_runner(contact_cfg.physics.runner)
            _log(
                "[PHASE-WORKER] "
                f"phase={phase} gpu={gpu} physics_runner={contact_cfg.physics.runner} initialized"
            )
        for job in _progress(jobs, total=len(jobs), desc=f"gpu{gpu} {phase}", position=int(gpu)):
            tool_path, obj_path, tool_name, obj_name, pose_idx, pose_seed = job
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            if phase == "geometry":
                if skip_existing and pt.exists():
                    _log(f"[GEOMETRY-SKIP] gpu={gpu} final_output={pt}")
                    skipped += 1
                    continue
                if skip_existing and candidate_artifact_path(pt).exists():
                    _log(f"[GEOMETRY-SKIP] gpu={gpu} candidate={candidate_artifact_path(pt)}")
                    ok += 1
                    continue
            elif phase == "stabilize":
                if skip_existing and pt.exists():
                    _log(f"[STABILIZE-SKIP] gpu={gpu} final_output={pt}")
                    skipped += 1
                    continue
                if not candidate_artifact_path(pt).exists():
                    _log(f"[STABILIZE-SKIP] gpu={gpu} missing_candidate={candidate_artifact_path(pt)}")
                    skipped += 1
                    continue
                if skip_existing and stabilized_artifact_path(pt).exists():
                    _log(f"[STABILIZE-SKIP] gpu={gpu} stabilized={stabilized_artifact_path(pt)}")
                    ok += 1
                    continue
            elif phase == "postcontact":
                if skip_existing and pt.exists():
                    _log(f"[POSTCONTACT-SKIP] gpu={gpu} output={pt}")
                    skipped += 1
                    continue
                if not stabilized_artifact_path(pt).exists():
                    _log(f"[POSTCONTACT-SKIP] gpu={gpu} missing_stabilized={stabilized_artifact_path(pt)}")
                    skipped += 1
                    continue
            else:
                raise ValueError(f"Unknown contact generation phase '{phase}'")

            if run_pair(
                tool_path,
                obj_path,
                tool_name,
                obj_name,
                out_dir,
                tools_meta,
                scale_xyz,
                gpu,
                contact_cfg,
                pose_idx,
                num_poses,
                pose_seed,
                physics_options,
                config_name,
                config_hash,
                physics_runner=physics_runner,
                phase=phase,
            ):
                ok += 1
            else:
                fail += 1
        _log(f"[PHASE-WORKER-DONE] phase={phase} gpu={gpu} ok={ok} fail={fail} skipped={skipped}")
        return ok, fail, skipped
    except KeyboardInterrupt:
        _log(
            "[INTERRUPT] phase worker interrupted "
            f"phase={phase} gpu={gpu} ok={ok} fail={fail} skipped={skipped}"
        )
        raise
    finally:
        _close_physics_runner(physics_runner, gpu)


def _build_jobs(pairs_subset, out_dir, num_poses, seed, *, skip_existing):
    rng = random.Random(seed)
    jobs = []
    for _pair_index, (tool_path, obj_path, tool_name, obj_name, _tool_asset) in enumerate(
        pairs_subset, start=1
    ):
        for pose_idx in range(num_poses):
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            pose_seed = rng.randint(0, 2**31 - 1)
            if skip_existing and pt.exists():
                continue
            jobs.append((tool_path, obj_path, tool_name, obj_name, pose_idx, pose_seed))
    return jobs


def visible_cuda_device_indices(*, requested_count: int = 0) -> list[int]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        actual_count = _torch_cuda_device_count()
        if actual_count is None:
            count = int(requested_count)
            return list(range(count)) if count > 0 else [0]
        if actual_count <= 0:
            return [0]
        return list(range(actual_count))
    stripped = value.strip()
    if stripped in {"", "-1"}:
        return [0]
    return list(range(len([item for item in stripped.split(",") if item.strip()]))) or [0]


def _torch_cuda_device_count() -> int | None:
    try:
        import torch
    except Exception:
        return None
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return None


def head_contact_probability(contact_cfg: ContactGenCfg) -> float:
    weights = contact_cfg.contact_mode_prob
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        return 0.0
    return float(weights.get("head", 0.0)) / total


def _load_generator_api(phase: str = "postcontact") -> GeneratorApi:
    _log(f"[IMPORT] importing contact_generation phase={phase}")
    from .gen_postcontact import ContactPairConfig

    if phase == "geometry":
        return ContactPairConfig, _run_geometry_contact_pair_from_pair_cfg
    if phase == "stabilize":
        from .stabilize_contact import run_stabilize_contact_pair

        return ContactPairConfig, run_stabilize_contact_pair
    if phase == "postcontact":
        from .gen_postcontact import run_contact_pair as postcontact_main

        return ContactPairConfig, postcontact_main
    raise ValueError(f"Unknown contact generation phase '{phase}'")


def _run_geometry_contact_pair_from_pair_cfg(cfg: Any, physics_runner: Any = None) -> Any:
    from .gen_contact import run_geometry_contact_pair

    return run_geometry_contact_pair(cfg.geometry_config())


def _load_physics_runner(name: str):
    from utils.contact.stabilize import get_physics_runner

    return get_physics_runner(name)


def _call_optimize_main(optimize_main: Callable[..., Any], cfg: Any, *, physics_runner: Any) -> Any:
    if physics_runner is None:
        return optimize_main(cfg)
    try:
        parameters = inspect.signature(optimize_main).parameters
    except (TypeError, ValueError):
        return optimize_main(cfg, physics_runner=physics_runner)
    accepts_runner = "physics_runner" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_runner:
        return optimize_main(cfg, physics_runner=physics_runner)
    return optimize_main(cfg)


def _close_physics_runner(physics_runner: Any, gpu: int) -> None:
    if physics_runner is None:
        return
    close = getattr(physics_runner, "close", None)
    if callable(close):
        _log(f"[WORKER] gpu={gpu} closing physics_runner")
        try:
            close()
        except Exception as exc:
            _log(f"[WARN] gpu={gpu} physics_runner close failed: {exc}")


def _object_id(obj_entry):
    if isinstance(obj_entry, str):
        return obj_entry
    if isinstance(obj_entry, dict):
        return str(obj_entry.get("name", obj_entry.get("object_id", obj_entry.get("id"))))
    return str(obj_entry)


def _required_path(paths: ProjectPaths, key: str) -> Path:
    path = paths.get(key)
    if path is None:
        raise ValueError(f"Missing required paths.yaml key '{key}'")
    return path


def _log(message: str) -> None:
    print(f"[contact_generation] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "contact_generation.batch_generate has no experiment-parameter CLI; "
        "call run_contact_generation(exp_cfg, paths, artifact_dir)."
    )

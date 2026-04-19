"""gen_dataset.py — Batch contact-configuration dataset generator.

Randomly samples tool × object pairs and runs contact generation,
writing output .pt files to pretrain/tmp_data/<tool>/<object>_pose<N>.pt.

Imports torch/trimesh once per worker, avoiding subprocess overhead.

Usage (single GPU, 200 random pairs, gradient method):
    python gen_dataset.py --num-pairs 200 --method gradient

Usage (multi-GPU, optimize method):
    python gen_dataset.py --num-pairs 500 --gpus 2 3 6 --num-poses 5 --method optimize

Methods:
    gradient   - Gradient-based approach (like corn.py): random pose + single gradient step
                 Better coverage, simpler, no optimization loop
    optimize   - Original approach: anchor on surface + Adam optimization
                 More refined contacts, but orientation-biased

Optional flags:
    --objects-json   Path to yes.json                (default: see DEFAULTS)
    --tools-json     Path to tools_selected.json     (default: see DEFAULTS)
    --tools-meta     Path to tools_adjusted.json     (default: see DEFAULTS)
    --out-dir        Root output directory            (default: tmp_data)
    --gpus           Space-separated GPU indices      (default: 0)
    --num-pairs      How many pairs to sample; 0 = all  (default: 200)
    --num-poses      Poses per tool×object pair       (default: 1)
    --seed           Random seed for pair sampling    (default: 42)
    --skip-existing  Skip pairs whose .pt already exists (default: on)
    --no-skip        Re-run even if .pt already exists
    --viz            Also run visualize_contacts.py after each pair
    --viz-dir        Where to write visualizations   (default: tmp_data/viz)
    --batch-size     Batch size for generation        (default: 512)
    --opt-steps      Optimization steps (optimize method only, default: 200)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

# Import both generators (heavy imports happen once here)
from contact_gen_gradient import Config as GradientConfig, main as gradient_gen_main
from contact_gen import Config as OptimizeConfig, main as optimize_gen_main

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
PRETRAIN_DIR = Path(__file__).resolve().parent
PATHS_YAML   = PRETRAIN_DIR.parent / "paths.yaml"


def load_paths(yaml_path: Path = PATHS_YAML) -> dict:
    """Load paths.yaml and return a flat dict of the keys gen_dataset needs."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "objects_json": cfg["dgn"]["candidates_json"],
        "obj_mesh_dir": cfg["dgn"]["obj_dir"],
        "tools_json":   cfg["tools"]["tools_selected_json"],
        "tool_mesh_dir": cfg["tools"]["obj_dir"],
        "tools_meta":   cfg["tools"]["tools_json"],
    }


VISUALIZE   = str(PRETRAIN_DIR / "visualize_contacts.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pairs(objects: list[str], tools: list[str],
                obj_mesh_dir: str, tool_mesh_dir: str) -> list[tuple]:
    """Return all valid (tool_path, obj_path, tool_name, obj_name) pairs.

    Object names in the candidates json have a trailing scale suffix, e.g.:
        core-bottle-44dcea00fb1923051a4cb1c0c7bb0654-0.100
    The actual mesh file drops that suffix:
        core-bottle-44dcea00fb1923051a4cb1c0c7bb0654.obj
    """
    pairs = []
    missing_obj, missing_tool = 0, 0
    for tool in tools:
        tool_path = Path(tool_mesh_dir) / f"{tool}.obj"
        if not tool_path.exists():
            print(f"  [WARN] tool mesh not found, skipping: {tool_path}")
            missing_tool += 1
            continue
        for obj_name in objects:
            # Strip trailing scale token (e.g. "-0.100")
            mesh_stem = obj_name.rsplit("-", 1)[0]
            obj_path = Path(obj_mesh_dir) / f"{mesh_stem}.obj"
            if not obj_path.exists():
                missing_obj += 1
                continue
            pairs.append((str(tool_path), str(obj_path), tool, obj_name))
    if missing_tool:
        print(f"  [WARN] {missing_tool} tool mesh(es) not found.")
    if missing_obj:
        print(f"  [WARN] {missing_obj} object mesh(es) not found.")
    return pairs


def sample_pairs(pairs: list[tuple], num_pairs: int, seed: int) -> list[tuple]:
    """Uniformly sample num_pairs from all valid pairs (without replacement)."""
    import random
    rng = random.Random(seed)
    if num_pairs <= 0 or num_pairs >= len(pairs):
        return pairs
    return rng.sample(pairs, num_pairs)


def output_path(out_dir: str, tool_name: str, obj_name: str, pose_idx: int = 0, num_poses: int = 1) -> Path:
    """Return output path for a specific pose index."""
    if num_poses == 1:
        return Path(out_dir) / tool_name / f"{obj_name}.pt"
    return Path(out_dir) / tool_name / f"{obj_name}_pose{pose_idx}.pt"


def run_pair(
    tool_path: str,
    obj_path: str,
    tool_name: str,
    obj_name: str,
    out_dir: str,
    viz_dir: str,
    tools_meta: str,
    gpu: int,
    batch_size: int,
    opt_steps: int,
    do_viz: bool,
    method: str,
    pose_idx: int = 0,
    num_poses: int = 1,
    seed: int = 42,
) -> bool:
    """Run contact generation for one (tool, obj) pair. Returns True on success."""
    import subprocess  # only for viz

    pt_file = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
    pt_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if method == "gradient":
            cfg = GradientConfig(
                object_mesh_path=obj_path,
                tool_mesh_path=tool_path,
                output_path=str(pt_file),
                batch_size=batch_size,
                device=f"cuda:{gpu}",
                seed=seed,
            )
            gradient_gen_main(cfg)
        else:  # optimize
            cfg = OptimizeConfig(
                object_mesh_path=obj_path,
                tool_mesh_path=tool_path,
                output_path=str(pt_file),
                tools_json_path=tools_meta if tools_meta and Path(tools_meta).exists() else "",
                batch_size=batch_size,
                opt_steps=opt_steps,
                device=f"cuda:{gpu}",
                seed=seed,
            )
            optimize_gen_main(cfg)
    except Exception as e:
        print(f"  [FAIL] {tool_name} × {obj_name} (pose {pose_idx}): {e}")
        return False

    if do_viz and pt_file.exists():
        viz_out = Path(viz_dir) / tool_name / f"{obj_name}_pose{pose_idx}.png"
        viz_out.parent.mkdir(parents=True, exist_ok=True)
        viz_cmd = [
            sys.executable, VISUALIZE,
            "--input", str(pt_file),
            "--num-tools", "4",
            "--save", str(viz_out),
        ]
        subprocess.run(viz_cmd, capture_output=True)

    return True


# ---------------------------------------------------------------------------
# Multi-GPU dispatch
# ---------------------------------------------------------------------------

def worker(pairs_subset, out_dir, viz_dir, tools_meta, gpu,
           batch_size, opt_steps, do_viz, skip_existing, num_poses, method):
    ok = fail = skip = 0
    for tool_path, obj_path, tool_name, obj_name in pairs_subset:
        for pose_idx in range(num_poses):
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            if skip_existing and pt.exists():
                skip += 1
                continue
            # Random seed for each pose
            pose_seed = random.randint(0, 2**31 - 1)
            tag = f"[GPU:{gpu}] {tool_name} × {obj_name} pose{pose_idx}"
            print(f"  → {tag} (seed={pose_seed})")
            success = run_pair(
                tool_path, obj_path, tool_name, obj_name,
                out_dir, viz_dir, tools_meta, gpu,
                batch_size, opt_steps, do_viz, method,
                pose_idx, num_poses, pose_seed,
            )
            if success:
                ok += 1
                print(f"  ✓ {tag}")
            else:
                fail += 1
    return ok, fail, skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = load_paths()
    default_out_dir = str(PRETRAIN_DIR / "tmp_data")

    parser = argparse.ArgumentParser(description="Batch contact dataset generator")
    parser.add_argument("--config",      default=str(PATHS_YAML),
                        help="Path to paths.yaml (default: ../paths.yaml)")
    parser.add_argument("--out-dir",     default=default_out_dir)
    parser.add_argument("--gpus",        nargs="+", type=int, default=[0])
    parser.add_argument("--num-pairs",   type=int, default=200,
                        help="Number of pairs to randomly sample (0 = all)")
    parser.add_argument("--num-poses",   type=int, default=1,
                        help="Number of poses to generate per tool×object pair")
    parser.add_argument("--method",      choices=["gradient", "optimize"], default="gradient",
                        help="Generation method: gradient (corn-like) or optimize (original)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for pair sampling")
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-run even if .pt already exists")
    parser.add_argument("--viz",         action="store_true")
    parser.add_argument("--viz-dir",     default=str(PRETRAIN_DIR / "tmp_data/viz"))
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--opt-steps",   type=int, default=100,
                        help="Optimization steps (optimize method only)")
    args = parser.parse_args()

    # Reload paths if a custom config was given
    p = load_paths(Path(args.config))
    skip_existing = not args.no_skip

    # Load lists from paths resolved via yaml
    with open(p["objects_json"]) as f:
        objects = json.load(f)
    with open(p["tools_json"]) as f:
        tools = json.load(f)

    print(f"Config      : {args.config}")
    print(f"Objects     : {len(objects)}  ({p['objects_json']})")
    print(f"Tools       : {len(tools)}  ({p['tools_json']})")

    all_pairs = build_pairs(objects, tools, p["obj_mesh_dir"], p["tool_mesh_dir"])
    pairs = sample_pairs(all_pairs, args.num_pairs, args.seed)

    print(f"Valid pairs : {len(all_pairs)}  →  sampling {len(pairs)}  (seed={args.seed}, poses={args.num_poses})")
    print(f"Method      : {args.method}")
    print(f"GPUs        : {args.gpus}")
    print(f"Output dir  : {args.out_dir}")
    print()

    if not pairs:
        print("No valid pairs found. Check mesh directories.")
        return

    # Distribute pairs across GPUs (round-robin)
    n_gpus = len(args.gpus)
    subsets = [[] for _ in range(n_gpus)]
    for i, pair in enumerate(pairs):
        subsets[i % n_gpus].append(pair)

    if n_gpus == 1:
        # Single GPU: run inline
        ok, fail, skip = worker(
            subsets[0], args.out_dir, args.viz_dir, p["tools_meta"],
            args.gpus[0], args.batch_size, args.opt_steps,
            args.viz, skip_existing, args.num_poses, args.method,
        )
    else:
        # Multi-GPU: one subprocess per GPU
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        with mp.Pool(n_gpus) as pool:
            results = pool.starmap(worker, [
                (subsets[i], args.out_dir, args.viz_dir, p["tools_meta"],
                 args.gpus[i], args.batch_size, args.opt_steps,
                 args.viz, skip_existing, args.num_poses, args.method)
                for i in range(n_gpus)
            ])
        ok   = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)
        skip = sum(r[2] for r in results)

    total = len(pairs) * args.num_poses
    print()
    print(f"Done.  ✓ {ok}  ✗ {fail}  ⟳ {skip} skipped  (total {total} = {len(pairs)} pairs × {args.num_poses} poses)")


if __name__ == "__main__":
    main()

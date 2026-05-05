"""gen_dataset.py — Batch contact-configuration dataset generator (new_pretrain).

Randomly samples tool × object pairs and runs contact_gen_new.py (rejection sampling),
writing output .pt files to new_pretrain/tmp_data/<tool>/<object>_pose<N>.pt.

Usage (single GPU, 200 random pairs, 10 poses each):
    python new_pretrain/gen_dataset.py --num-pairs 200 --num-poses 10 --gpus 0

Usage (multi-GPU):
    python new_pretrain/gen_dataset.py --num-pairs 500 --gpus 0 1 2 3 --num-poses 10

Optional flags:
    --objects-json   Path to yes.json                (default: see paths.yaml)
    --tools-json     Path to tools_selected.json     (default: see paths.yaml)
    --tools-meta     Path to tools_adjusted.json     (default: see paths.yaml)
    --out-dir        Root output directory            (default: new_pretrain/tmp_data)
    --gpus           Space-separated GPU indices      (default: 0)
    --num-pairs      How many pairs to sample; 0 = all  (default: 200)
    --num-poses      Poses per tool×object pair       (default: 10)
    --seed           Random seed for pair sampling    (default: 42)
    --no-skip        Re-run even if .pt already exists
    --B              Contact pairs per sampler call   (default: from contact_config)
    --M              Candidate rotations per pair     (default: from contact_config)
    --chunk-B        GPU memory chunk size            (default: from contact_config)
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

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PRETRAIN_DIR = _THIS_DIR.parent

# Make sure we can import from new_pretrain directly
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from contact_gen_new import Config as OptimizeConfig, main as optimize_gen_main
from contact_config import CONTACT_GEN

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
PATHS_YAML = _PRETRAIN_DIR.parent / "paths.yaml"


def load_paths(yaml_path: Path = PATHS_YAML) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "objects_json":  cfg["dgn"]["candidates_json"],
        "obj_mesh_dir":  cfg["dgn"]["obj_dir"],
        "tools_json":    cfg["tools"]["tools_selected_json"],
        "tool_mesh_dir": cfg["tools"]["obj_dir"],
        "tools_meta":    cfg["tools"]["tools_json"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pairs(objects, tools, obj_mesh_dir, tool_mesh_dir):
    pairs = []
    missing_obj = missing_tool = 0
    for tool in tools:
        tool_path = Path(tool_mesh_dir) / f"{tool}.obj"
        if not tool_path.exists():
            missing_tool += 1
            continue
        for obj_name in objects:
            mesh_stem = obj_name.rsplit("-", 1)[0]
            obj_path  = Path(obj_mesh_dir) / f"{mesh_stem}.obj"
            if not obj_path.exists():
                missing_obj += 1
                continue
            pairs.append((str(tool_path), str(obj_path), tool, obj_name))
    if missing_tool:
        print(f"  [WARN] {missing_tool} tool mesh(es) not found.")
    if missing_obj:
        print(f"  [WARN] {missing_obj} object mesh(es) not found.")
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


def run_pair(
    tool_path, obj_path, tool_name, obj_name,
    out_dir, tools_meta, gpu,
    B, M, chunk_B,
    pose_idx=0, num_poses=1, seed=42,
) -> bool:
    pt_file = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
    pt_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        cfg = OptimizeConfig(
            object_mesh_path=obj_path,
            tool_mesh_path=tool_path,
            output_path=str(pt_file),
            tools_json_path=tools_meta if tools_meta and Path(tools_meta).exists() else "",
            device=f"cuda:{gpu}",
            seed=seed,
            B=B,
            M=M,
            chunk_B=chunk_B,
        )
        optimize_gen_main(cfg)
    except Exception as e:
        print(f"  [FAIL] {tool_name} × {obj_name} (pose {pose_idx}): {e}")
        import traceback; traceback.print_exc()
        return False
    return True


# ---------------------------------------------------------------------------
# Multi-GPU dispatch
# ---------------------------------------------------------------------------

def worker(pairs_subset, out_dir, tools_meta, gpu,
           B, M, chunk_B, skip_existing, num_poses):
    ok = fail = skip = 0
    for tool_path, obj_path, tool_name, obj_name in pairs_subset:
        for pose_idx in range(num_poses):
            pt = output_path(out_dir, tool_name, obj_name, pose_idx, num_poses)
            if skip_existing and pt.exists():
                skip += 1
                continue
            pose_seed = random.randint(0, 2**31 - 1)
            tag = f"[GPU:{gpu}] {tool_name} × {obj_name} pose{pose_idx}"
            print(f"  → {tag} (seed={pose_seed})")
            success = run_pair(
                tool_path, obj_path, tool_name, obj_name,
                out_dir, tools_meta, gpu,
                B, M, chunk_B, pose_idx, num_poses, pose_seed,
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
    default_out_dir = str(_THIS_DIR / "tmp_data")

    parser = argparse.ArgumentParser(description="Batch contact dataset generator (rejection sampler)")
    parser.add_argument("--config",    default=str(PATHS_YAML))
    parser.add_argument("--out-dir",   default=default_out_dir)
    parser.add_argument("--gpus",      nargs="+", type=int, default=[0])
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--num-poses", type=int, default=10,
                        help="Poses per tool×object pair (default: 10)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-skip",   action="store_true")
    # Sampler params — default from contact_config.py
    parser.add_argument("--B",         type=int, default=CONTACT_GEN.B,
                        help=f"Contact pairs per call (default: {CONTACT_GEN.B})")
    parser.add_argument("--M",         type=int, default=CONTACT_GEN.M,
                        help=f"Candidate rotations per pair (default: {CONTACT_GEN.M})")
    parser.add_argument("--chunk-B",   type=int, default=CONTACT_GEN.chunk_B,
                        help=f"GPU memory chunk (default: {CONTACT_GEN.chunk_B}; "
                             f"memory ≈ chunk_B×M×K×3×4 bytes)")
    args = parser.parse_args()

    p = load_paths(Path(args.config))
    skip_existing = not args.no_skip

    with open(p["objects_json"]) as f:
        objects = json.load(f)
    with open(p["tools_json"]) as f:
        tools = json.load(f)

    print(f"Config      : {args.config}")
    print(f"Objects     : {len(objects)}  ({p['objects_json']})")
    print(f"Tools       : {len(tools)}  ({p['tools_json']})")
    print(f"Sampler     : B={args.B}, M={args.M}, chunk_B={args.chunk_B}")

    all_pairs = build_pairs(objects, tools, p["obj_mesh_dir"], p["tool_mesh_dir"])
    pairs     = sample_pairs(all_pairs, args.num_pairs, args.seed)

    print(f"Valid pairs : {len(all_pairs)}  →  sampling {len(pairs)}  "
          f"(seed={args.seed}, poses={args.num_poses})")
    print(f"GPUs        : {args.gpus}")
    print(f"Output dir  : {args.out_dir}")
    print()

    if not pairs:
        print("No valid pairs found. Check mesh directories.")
        return

    n_gpus  = len(args.gpus)
    subsets = [[] for _ in range(n_gpus)]
    for i, pair in enumerate(pairs):
        subsets[i % n_gpus].append(pair)

    if n_gpus == 1:
        ok, fail, skip = worker(
            subsets[0], args.out_dir, p["tools_meta"],
            args.gpus[0], args.B, args.M, args.chunk_B,
            skip_existing, args.num_poses,
        )
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        with mp.Pool(n_gpus) as pool:
            results = pool.starmap(worker, [
                (subsets[i], args.out_dir, p["tools_meta"],
                 args.gpus[i], args.B, args.M, args.chunk_B,
                 skip_existing, args.num_poses)
                for i in range(n_gpus)
            ])
        ok   = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)
        skip = sum(r[2] for r in results)

    total = len(pairs) * args.num_poses
    print()
    print(f"Done.  ✓ {ok}  ✗ {fail}  ⟳ {skip} skipped  "
          f"(total {total} = {len(pairs)} pairs × {args.num_poses} poses)")


if __name__ == "__main__":
    main()

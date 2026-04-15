"""gen_dataset.py — Batch contact-configuration dataset generator.

Randomly samples tool × object pairs and runs contact_gen.py for each,
writing output .pt files to pretrain/tmp_data/<tool>/<object>.pt.

Usage (single GPU, 200 random pairs):
    python gen_dataset.py --num-pairs 200

Usage (multi-GPU):
    python gen_dataset.py --num-pairs 500 --gpus 2 3 6

Optional flags:
    --objects-json   Path to yes.json                (default: see DEFAULTS)
    --tools-json     Path to tools_selected.json     (default: see DEFAULTS)
    --tools-meta     Path to tools_adjusted.json     (default: see DEFAULTS)
    --out-dir        Root output directory            (default: tmp_data)
    --gpus           Space-separated GPU indices      (default: 0)
    --num-pairs      How many pairs to sample; 0 = all  (default: 200)
    --seed           Random seed for pair sampling    (default: 42)
    --skip-existing  Skip pairs whose .pt already exists (default: on)
    --no-skip        Re-run even if .pt already exists
    --viz            Also run visualize_contacts.py after each pair
    --viz-dir        Where to write visualizations   (default: tmp_data/viz)
    --batch-size     Passed to contact_gen.py        (default: 512)
    --opt-steps      Passed to contact_gen.py        (default: 300)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

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


CONTACT_GEN = str(PRETRAIN_DIR / "contact_gen.py")
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


def output_path(out_dir: str, tool_name: str, obj_name: str) -> Path:
    return Path(out_dir) / tool_name / f"{obj_name}.pt"


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
) -> bool:
    """Run contact_gen.py for one (tool, obj) pair. Returns True on success."""
    pt_file = output_path(out_dir, tool_name, obj_name)
    pt_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, CONTACT_GEN,
        "--object", obj_path,
        "--tool",   tool_path,
        "--output", str(pt_file),
        "--device", f"cuda:{gpu}",
        "--batch-size", str(batch_size),
        "--opt-steps",  str(opt_steps),
    ]
    if tools_meta and Path(tools_meta).exists():
        cmd += ["--tools-json", tools_meta]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {tool_name} × {obj_name}")
        print(result.stderr[-800:])
        return False

    if do_viz and pt_file.exists():
        viz_out = Path(viz_dir) / tool_name / f"{obj_name}.png"
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
           batch_size, opt_steps, do_viz, skip_existing):
    ok = fail = skip = 0
    for tool_path, obj_path, tool_name, obj_name in pairs_subset:
        pt = output_path(out_dir, tool_name, obj_name)
        if skip_existing and pt.exists():
            skip += 1
            continue
        tag = f"[GPU:{gpu}] {tool_name} × {obj_name}"
        print(f"  → {tag}")
        success = run_pair(
            tool_path, obj_path, tool_name, obj_name,
            out_dir, viz_dir, tools_meta, gpu,
            batch_size, opt_steps, do_viz,
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
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for pair sampling")
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-run even if .pt already exists")
    parser.add_argument("--viz",         action="store_true")
    parser.add_argument("--viz-dir",     default=str(PRETRAIN_DIR / "tmp_data/viz"))
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--opt-steps",   type=int, default=300)
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

    print(f"Valid pairs : {len(all_pairs)}  →  sampling {len(pairs)}  (seed={args.seed})")
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
            args.viz, skip_existing,
        )
    else:
        # Multi-GPU: one subprocess per GPU
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        with mp.Pool(n_gpus) as pool:
            results = pool.starmap(worker, [
                (subsets[i], args.out_dir, args.viz_dir, p["tools_meta"],
                 args.gpus[i], args.batch_size, args.opt_steps,
                 args.viz, skip_existing)
                for i in range(n_gpus)
            ])
        ok   = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)
        skip = sum(r[2] for r in results)

    total = len(pairs)
    print()
    print(f"Done.  ✓ {ok}  ✗ {fail}  ⟳ {skip} skipped  (total {total})")


if __name__ == "__main__":
    main()

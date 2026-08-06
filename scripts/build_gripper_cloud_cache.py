#!/usr/bin/env python3
"""Build the sole strict 128-bin corresponding-point cache for grippers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.assets.one_dof_gripper_assets import load_one_dof_gripper_manifest
from utils.assets.generated_gripper_assets import load_generated_gripper_manifest
from utils.geometry.gripper_cloud_cache import (
    build_cache_payload,
    cache_path_for_asset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--generated-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.generated_manifest is not None:
        source_manifest_path = args.generated_manifest
        assets = load_generated_gripper_manifest(source_manifest_path)
    else:
        source_manifest_path = args.manifest
        assets = load_one_dof_gripper_manifest(
            source_manifest_path, require_usd=False
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source_manifest_path.expanduser().resolve().parent
        / "kinematic_cloud_cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    entries = {
        str(entry["id"]): entry
        for entry in previous.get("grippers", [])
    }
    source_manifest = str(source_manifest_path.expanduser().resolve())
    for asset in assets:
        path = cache_path_for_asset(asset, output_dir)
        cache_payload = build_cache_payload(asset)
        cache_payload["source_manifest"] = source_manifest
        torch.save(cache_payload, path)
        entries[asset.gripper_id] = {
            "id": asset.gripper_id,
            "path": str(path),
            "source_manifest": source_manifest,
        }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "gripper_cloud_cache_manifest_v1",
                "num_bins": 128,
                "num_points": 512,
                "grippers": [entries[key] for key in sorted(entries)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(assets)} caches to {output_dir}")


if __name__ == "__main__":
    main()

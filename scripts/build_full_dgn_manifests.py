#!/usr/bin/env python
"""Build full DGN train/test manifests from the normalized OBJ directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_DGN_ROOT = Path("/mnt/project/world_model/tool_generalist/assets/DGN")
DEFAULT_SCALES = ("0.060", "0.080", "0.100", "0.120", "0.150")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgn-root", type=Path, default=DEFAULT_DGN_ROOT)
    parser.add_argument("--mesh-dir", default="coacd_normalized")
    parser.add_argument("--test-source", default="test_set.json")
    parser.add_argument("--train-output", default="full_yes.json")
    parser.add_argument("--test-output", default="full_test.json")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--test-mode",
        choices=("copy", "all-scales"),
        default="copy",
        help="copy preserves the existing held-out test_set entries.",
    )
    args = parser.parse_args()

    dgn_root = args.dgn_root.expanduser()
    mesh_dir = dgn_root / args.mesh_dir
    test_source = dgn_root / args.test_source
    train_output = dgn_root / args.train_output
    test_output = dgn_root / args.test_output

    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Missing DGN mesh dir: {mesh_dir}")
    test_entries = _read_json_list(test_source)
    test_bases = sorted({_mesh_stem(entry) for entry in test_entries})
    mesh_bases = sorted(path.stem for path in mesh_dir.glob("*.obj"))
    mesh_base_set = set(mesh_bases)
    missing_test = sorted(base for base in test_bases if base not in mesh_base_set)
    if missing_test:
        raise FileNotFoundError(
            f"{len(missing_test)} test object(s) are missing OBJ meshes, first={missing_test[0]}"
        )

    scales = tuple(str(scale) for scale in args.scales)
    if not scales:
        raise ValueError("--scales must contain at least one scale string")

    full_yes = [
        f"{base}-{_stable_scale(base, scales)}"
        for base in mesh_bases
        if base not in test_bases
    ]
    if args.test_mode == "copy":
        full_test = test_entries
    else:
        full_test = [f"{base}-{scale}" for base in test_bases for scale in scales]

    _write_json(train_output, full_yes)
    _write_json(test_output, full_test)
    print(
        "wrote "
        f"train={train_output} entries={len(full_yes)} "
        f"test={test_output} entries={len(full_test)} "
        f"mesh_bases={len(mesh_bases)} heldout_bases={len(test_bases)}"
    )
    return 0


def _read_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON: {path}")
    entries: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            entries.append(str(item.get("name", item.get("object_id", item.get("id")))))
        else:
            entries.append(str(item))
    return entries


def _write_json(path: Path, payload: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _mesh_stem(entry: str) -> str:
    return str(entry).rsplit("-", 1)[0]


def _stable_scale(base: str, scales: tuple[str, ...]) -> str:
    digest = hashlib.sha256(base.encode("utf-8")).digest()
    return scales[int.from_bytes(digest[:4], "big") % len(scales)]


if __name__ == "__main__":
    raise SystemExit(main())

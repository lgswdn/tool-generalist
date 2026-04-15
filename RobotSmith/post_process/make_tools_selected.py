"""
make_tools_selected.py
----------------------
Populate tools_selected.json with *all* tool names found in tools_adjusted.json.

Usage:
    python post_process/make_tools_selected.py
    python post_process/make_tools_selected.py \
        --adjusted eef/tools_adjusted.json \
        --selected eef/tools_selected.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    # Default paths: sibling eef/ directory relative to this script's parent (RobotSmith/)
    repo_root = Path(__file__).parent.parent
    eef_dir = repo_root / "eef"

    parser = argparse.ArgumentParser(description="Sync tools_selected.json from tools_adjusted.json")
    parser.add_argument(
        "--adjusted",
        default=str(eef_dir / "tools_adjusted.json"),
        help="Source file (default: eef/tools_adjusted.json)",
    )
    parser.add_argument(
        "--selected",
        default=str(eef_dir / "tools_selected.json"),
        help="Destination file (default: eef/tools_selected.json)",
    )
    args = parser.parse_args()

    adjusted_path = Path(args.adjusted)
    selected_path = Path(args.selected)

    with open(adjusted_path, "r") as f:
        adjusted: list[dict] = json.load(f)

    names = [entry["name"] for entry in adjusted]

    with open(selected_path, "w") as f:
        json.dump(names, f, indent=2)

    print(f"✓ Wrote {len(names)} tool names → {selected_path}")


if __name__ == "__main__":
    main()

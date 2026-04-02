import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Collect tool metadata into tools.json")
    parser.add_argument("--eef-dir", type=str, default=None,
                        help="Path to the eef directory (default: ../eef relative to this script)")
    args = parser.parse_args()

    if args.eef_dir is None:
        eef_dir = Path(__file__).resolve().parent.parent / "eef"
    else:
        eef_dir = Path(args.eef_dir).resolve()

    metadata_dir = eef_dir / "objects_metadata"
    output_path = eef_dir / "tools.json"

    tools = []
    for f in metadata_dir.glob("*_metadata.json"):
        data = json.loads(f.read_text())
        tools.append({
            "name": f.stem.replace("_metadata", ""),
            "head_area": data.get("head_area"),
            "base_center": data.get("base_center"),
        })

    output_path.write_text(json.dumps(tools, indent=2))
    print(f"[INFO] Wrote {len(tools)} tools to {output_path}")

if __name__ == "__main__":
    main()

import json
from pathlib import Path

tools = []
for f in Path("../eef/objects_metadata").glob("*_metadata.json"):
    data = json.loads(f.read_text())
    tools.append({"name": f.stem.replace("_metadata", ""), "head_area": data.get("head_area"), "base_center": data.get("base_center")})

Path("../eef/tools.json").write_text(json.dumps(tools, indent=2))

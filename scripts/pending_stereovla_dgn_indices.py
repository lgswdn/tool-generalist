#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


TASK_INDEX_RE = re.compile(r"^dgn_(\d{6})_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--retry-count", type=int, default=0)
    args = parser.parse_args()

    attempts = {}
    episodes_path = Path(args.episodes)
    if episodes_path.exists():
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                base_task = record.get("failure_retry_base_task_id") or record.get("task_id", "")
                match = TASK_INDEX_RE.match(str(base_task))
                if not match:
                    continue
                index = int(match.group(1))
                retry_index = record.get("failure_retry_index")
                retry_index = 0 if retry_index is None else int(retry_index)
                state = attempts.setdefault(index, {"success": False, "max_retry": -1})
                state["success"] = state["success"] or bool(record.get("success"))
                state["max_retry"] = max(state["max_retry"], retry_index)

    end = args.start + args.count
    for index in range(args.start, end):
        state = attempts.get(index)
        complete = state is not None and (
            state["success"] or state["max_retry"] >= args.retry_count
        )
        if not complete:
            print(index)


if __name__ == "__main__":
    main()

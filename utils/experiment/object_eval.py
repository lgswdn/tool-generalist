"""Pure helpers shared by object-level policy evaluation code."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def load_candidate_entries(path: str, label: str) -> list[str | dict[str, Any]]:
    """Load a homogeneous string or mapping object-candidate manifest."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{label} is empty: {path}")

    string_schema = all(isinstance(item, str) for item in data)
    mapping_schema = all(isinstance(item, Mapping) for item in data)
    if not string_schema and not mapping_schema:
        raise ValueError(
            f"Expected {label} to contain either '<object>-<scale>' strings or "
            f"object/scale mappings without mixing schemas: {path}"
        )
    return data


def scale_stats(scales: list[float]) -> dict:
    values = [round(float(scale), 8) for scale in scales]
    unique_values = sorted(set(values))
    if not values:
        return {
            "scale_mean": None,
            "scale_min": None,
            "scale_max": None,
            "scale_values": [],
            "episode_scales": [],
        }
    return {
        "scale_mean": sum(values) / len(values),
        "scale_min": min(values),
        "scale_max": max(values),
        "scale_values": unique_values,
        "episode_scales": values,
    }


def merge_rows_by_object(rows: list[dict]) -> list[dict]:
    """Merge rank-local statistics into one row per object."""

    grouped: dict[str, dict] = {}
    for row in rows:
        name = str(row["name"])
        merged = grouped.setdefault(
            name,
            {
                "name": name,
                "episodes": 0,
                "successes": 0,
                "episode_scales": [],
                "ranks": set(),
            },
        )
        merged["episodes"] += int(row["episodes"])
        merged["successes"] += int(row["successes"])
        merged["episode_scales"].extend(float(value) for value in row.get("episode_scales", []))
        row_ranks = row.get("ranks")
        if row_ranks is None:
            row_ranks = [row.get("rank")]
        merged["ranks"].update(int(value) for value in row_ranks if value is not None)

    result = []
    for name in sorted(grouped):
        merged = grouped[name]
        episodes = int(merged["episodes"])
        successes = int(merged["successes"])
        ranks = sorted(merged["ranks"])
        result.append(
            {
                "name": name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": float(successes) / float(episodes) if episodes > 0 else 0.0,
                **scale_stats(merged["episode_scales"]),
                "rank": ranks[0] if len(ranks) == 1 else None,
                "ranks": ranks,
            }
        )
    return result

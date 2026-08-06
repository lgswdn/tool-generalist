from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.experiment.object_eval import load_candidate_entries, merge_rows_by_object


ROOT = Path(__file__).resolve().parents[1]


def test_load_candidate_entries_preserves_listed_scale_mappings(tmp_path):
    path = tmp_path / "objects.json"
    payload = [
        {"object": "object_a", "scale": 0.125},
        {"object": "object_b", "scale": 0.25},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_candidate_entries(str(path), "candidates_json") == payload


def test_load_candidate_entries_rejects_mixed_schemas(tmp_path):
    path = tmp_path / "objects.json"
    path.write_text(
        json.dumps(["object_a-0.125", {"object": "object_b", "scale": 0.25}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="without mixing schemas"):
        load_candidate_entries(str(path), "candidates_json")


def test_merge_rows_by_object_aggregates_all_ranks():
    rows = [
        {
            "name": "object_a",
            "episodes": 2,
            "successes": 1,
            "episode_scales": [0.125, 0.125],
            "rank": 0,
            "ranks": [0],
        },
        {
            "name": "object_a",
            "episodes": 2,
            "successes": 2,
            "episode_scales": [0.125, 0.125],
            "rank": 1,
            "ranks": [1],
        },
        {
            "name": "object_b",
            "episodes": 2,
            "successes": 0,
            "episode_scales": [0.25, 0.25],
            "rank": 0,
            "ranks": [0],
        },
    ]

    merged = merge_rows_by_object(rows)

    assert [row["name"] for row in merged] == ["object_a", "object_b"]
    assert merged[0]["episodes"] == 4
    assert merged[0]["successes"] == 3
    assert merged[0]["success_rate"] == pytest.approx(0.75)
    assert merged[0]["scale_values"] == [0.125]
    assert merged[0]["ranks"] == [0, 1]
    assert merged[0]["rank"] is None


def test_eval_script_exposes_full_set_random_gripper_mode():
    source = (ROOT / "scripts/eval_objects.py").read_text(encoding="utf-8")

    assert '"--replicate_objects_across_ranks"' in source
    assert '"--require_one_env_per_object"' in source
    assert '"--randomize_grippers"' in source
    assert 'asset_assignment["randomize_tool_assignment"] = True' in source
    assert "list(all_candidates)" in source
    assert "merge_rows_by_object(all_rows)" in source

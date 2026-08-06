from __future__ import annotations

import json

from scripts.refine_high_conf_colliding_objects import refine_manifest


def _candidate(confidence: float, *, free: bool, colliding: bool) -> dict:
    return {
        "confidence": confidence,
        "valid_se3": True,
        "hand_and_fingers_collision_free": free,
        "panda_hand_object_collision": colliding,
        "panda_hand_ground_collision": False,
        "panda_fingers_object_collision": False,
        "panda_fingers_ground_collision": False,
    }


def test_refinement_is_strict_subset_and_preserves_scales(tmp_path):
    input_path = tmp_path / "input.json"
    report_path = tmp_path / "report.jsonl"
    output_path = tmp_path / "output.json"
    input_rows = [
        {"object": "selected", "scale": 0.11},
        {"object": "exact_threshold", "scale": 0.22},
        {"object": "has_free", "scale": 0.33},
    ]
    input_path.write_text(json.dumps(input_rows), encoding="utf-8")
    report_rows = [
        {
            "object": "selected",
            "episode_success": False,
            "candidates": [_candidate(0.91, free=False, colliding=True)],
        },
        {
            "object": "exact_threshold",
            "episode_success": False,
            "candidates": [_candidate(0.9, free=False, colliding=True)],
        },
        {
            "object": "has_free",
            "episode_success": False,
            "candidates": [
                _candidate(0.96, free=False, colliding=True),
                _candidate(0.95, free=True, colliding=False),
            ],
        },
        {
            "object": "outside_source_manifest",
            "episode_success": False,
            "candidates": [_candidate(0.99, free=False, colliding=True)],
        },
    ]
    report_path.write_text(
        "".join(json.dumps(row) + "\n" for row in report_rows),
        encoding="utf-8",
    )

    summary = refine_manifest(report_path, input_path, output_path, 0.9)

    assert json.loads(output_path.read_text(encoding="utf-8")) == [input_rows[0]]
    assert summary["confidence_operator"] == ">"
    assert summary["source_object_count"] == 3
    assert summary["selected_object_count"] == 1
    assert summary["removed_object_count"] == 2

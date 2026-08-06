from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import contact_generation.gen_contact as gen_contact
from configs.config_contact_gen import ContactGenCfg
from contact_generation.gen_postcontact import ContactPairConfig


def _cfg(**overrides):
    values = {
        "device": "cpu",
        "B": 4,
        "M": 1,
        "chunk_B": 4,
        "upright_threshold": 0.0,
        "floor_eps": 0.0,
        "penetration_eps": 5e-4,
        "penetration_check_mode": (
            gen_contact.PENETRATION_CHECK_TOOL_INTO_OBJECT
        ),
        "rotation_selection": gen_contact.ROTATION_SELECTION_RANDOM_LEGAL,
        "visualization_enabled": False,
        "rejection_refill": True,
        "rejection_max_rounds": 4,
        "rejection_apply_tangent_gaussian": False,
        "tangent_translation_noise_std": 0.002,
        "tangent_rotation_noise_std_rad": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rejection_refill_returns_exactly_b_candidates(monkeypatch):
    sdf_calls = 0

    def fake_sdf(points, *_args):
        nonlocal sdf_calls
        sdf_calls += 1
        # Reject half of the initial four anchors, then accept both refill
        # anchors. Negative SDF means penetration.
        if sdf_calls == 1:
            return torch.tensor([0.01, -0.01, 0.01, -0.01])
        return torch.full((points.shape[0],), 0.01)

    monkeypatch.setattr(gen_contact, "query_sdf_grid", fake_sdf)
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )

    tool = torch.tensor([[0.0, 0.0, 0.0]])
    object_surface = torch.tensor([[0.0, 0.0, 1.0]])
    result = gen_contact.rejection_sample_candidates(
        tool,
        object_surface,
        _cfg(),
        sdf_grid=None,
        bbox_min=None,
        bbox_max=None,
        P_anchor=tool,
    )

    assert result["tool_rotation_E"].shape == (4, 3, 3)
    assert result["tool_translation_E"].shape == (4, 3)
    assert result["source_candidate_index"].tolist() == [0, 2, 4, 5]
    assert result["geometry_diagnostics"]["refill_rounds"] == 2


def test_rejection_refill_fails_instead_of_returning_short_dataset(monkeypatch):
    monkeypatch.setattr(
        gen_contact,
        "query_sdf_grid",
        lambda points, *_args: torch.full((points.shape[0],), -0.01),
    )
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )

    with pytest.raises(RuntimeError, match="accepted=0 required=4"):
        gen_contact.rejection_sample_candidates(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            _cfg(rejection_max_rounds=2),
            sdf_grid=None,
            bbox_min=None,
            bbox_max=None,
            P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
        )


def test_rejection_refill_oversamples_the_last_missing_candidate(monkeypatch):
    sdf_calls = 0

    def fake_sdf(points, *_args):
        nonlocal sdf_calls
        sdf_calls += 1
        if sdf_calls == 1:
            # Initial batch accepts 3/4.
            return torch.tensor([0.01, 0.01, 0.01, -0.01])
        # A refill batch must contain multiple proposals; only its final
        # proposal is legal. The old remaining=1 behavior could never find it.
        values = torch.full((points.shape[0],), -0.01)
        values[-1] = 0.01
        return values

    monkeypatch.setattr(gen_contact, "query_sdf_grid", fake_sdf)
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )

    result = gen_contact.rejection_sample_candidates(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        _cfg(rejection_max_rounds=2),
        sdf_grid=None,
        bbox_min=None,
        bbox_max=None,
        P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
    )

    assert result["tool_rotation_E"].shape[0] == 4
    assert result["geometry_diagnostics"]["attempted_anchor_pairs"] == 8


def test_bidirectional_rejection_checks_object_points_inside_tool(monkeypatch):
    sdf_calls = 0

    def fake_sdf(points, *_args):
        nonlocal sdf_calls
        sdf_calls += 1
        if sdf_calls % 2 == 1:
            # Forward tool-into-object query is legal.
            return torch.full((points.shape[0],), 0.01)
        # Reverse object-into-tool query rejects the first round and accepts
        # the refill round.
        value = -0.01 if sdf_calls == 2 else 0.01
        return torch.full((points.shape[0],), value)

    monkeypatch.setattr(gen_contact, "query_sdf_grid", fake_sdf)
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )

    result = gen_contact.rejection_sample_candidates(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        _cfg(
            B=2,
            chunk_B=2,
            rejection_max_rounds=2,
            penetration_check_mode=(
                gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
            ),
        ),
        sdf_grid="object_sdf",
        bbox_min=torch.zeros(3),
        bbox_max=torch.ones(3),
        object_points_E=torch.tensor([[0.0, 0.0, 1.0]]),
        tool_sdf_grid="tool_sdf",
        tool_bbox_min=torch.zeros(3),
        tool_bbox_max=torch.ones(3),
        P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
    )

    assert result["tool_rotation_E"].shape[0] == 2
    assert result["geometry_diagnostics"]["refill_rounds"] == 2


def test_bidirectional_mode_propagates_to_geometry_config():
    contact_cfg = ContactGenCfg()
    contact_cfg.penetration_check_mode = (
        gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
    )
    pair_cfg = ContactPairConfig.from_contact_cfg(
        contact_cfg=contact_cfg,
        object_mesh_path="object.obj",
        tool_mesh_path="tool.obj",
        output_path="output.pt",
        tools_json_path="tools.json",
        object_id="object",
        tool_id="tool",
        config_name="test",
        config_hash="hash",
        device="cpu",
        seed=0,
        tool_scale_xyz=(1.0, 1.0, 1.0),
    )

    assert (
        pair_cfg.penetration_check_mode
        == gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
    )
    assert (
        pair_cfg.geometry_config().penetration_check_mode
        == gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
    )


def test_rejection_chooses_most_cavity_centered_legal_rotation(monkeypatch):
    identity = torch.eye(3)
    flip_x_y = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda *_args: torch.stack((identity, flip_x_y)),
    )
    monkeypatch.setattr(
        gen_contact,
        "query_sdf_grid",
        lambda points, *_args: torch.full((points.shape[0],), 0.01),
    )

    result = gen_contact.rejection_sample_candidates(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        _cfg(
            B=1,
            M=2,
            chunk_B=1,
            penetration_check_mode=(
                gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
            ),
            rotation_selection=(
                gen_contact.ROTATION_SELECTION_MOST_CAVITY_CENTERED
            ),
        ),
        sdf_grid="object_sdf",
        bbox_min=torch.zeros(3),
        bbox_max=torch.ones(3),
        object_points_E=torch.tensor([[1.0, 0.0, 0.0]]),
        tool_sdf_grid="tool_sdf",
        tool_bbox_min=torch.zeros(3),
        tool_bbox_max=torch.ones(3),
        P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
        object_center_E=torch.tensor([1.0, 0.0, 0.0]),
        finger_cavity_halfspaces_T=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        ),
    )

    assert torch.equal(result["tool_rotation_E"][0], flip_x_y)
    assert result["cavity_capture_fraction"].tolist() == [1.0]
    assert result["geometry_diagnostics"]["attempted_anchor_pairs"] == 4


def test_cavity_ranking_samples_more_until_b_pair_winners_are_legal(monkeypatch):
    sdf_calls = 0

    def fake_sdf(points, *_args):
        nonlocal sdf_calls
        sdf_calls += 1
        if sdf_calls == 1:
            # The initial 4B=8 anchor pairs contain only one legal winner.
            values = torch.full((points.shape[0],), -0.01)
            values[0] = 0.01
            return values
        return torch.full((points.shape[0],), 0.01)

    monkeypatch.setattr(gen_contact, "query_sdf_grid", fake_sdf)
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )

    result = gen_contact.rejection_sample_candidates(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        _cfg(
            B=2,
            M=1,
            chunk_B=8,
            rejection_max_rounds=2,
            penetration_check_mode=(
                gen_contact.PENETRATION_CHECK_BIDIRECTIONAL
            ),
            rotation_selection=(
                gen_contact.ROTATION_SELECTION_MOST_CAVITY_CENTERED
            ),
        ),
        sdf_grid="object_sdf",
        bbox_min=torch.zeros(3),
        bbox_max=torch.ones(3),
        object_points_E=torch.tensor([[0.0, 0.0, 0.0]]),
        tool_sdf_grid="tool_sdf",
        tool_bbox_min=torch.zeros(3),
        tool_bbox_max=torch.ones(3),
        P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
        object_center_E=torch.zeros(3),
        finger_cavity_halfspaces_T=torch.tensor(
            [[1.0, 0.0, 0.0, -1.0]]
        ),
    )

    assert result["tool_rotation_E"].shape[0] == 2
    assert result["geometry_diagnostics"]["refill_rounds"] == 2
    assert result["geometry_diagnostics"]["attempted_anchor_pairs"] == 12
    assert result["geometry_diagnostics"]["legal_anchor_pair_winners"] == 5


def test_rejection_perturbation_propagates_to_geometry_config():
    contact_cfg = ContactGenCfg()
    contact_cfg.rejection_apply_tangent_gaussian = True
    pair_cfg = ContactPairConfig.from_contact_cfg(
        contact_cfg=contact_cfg,
        object_mesh_path="object.obj",
        tool_mesh_path="tool.obj",
        output_path="output.pt",
        tools_json_path="tools.json",
        object_id="object",
        tool_id="tool",
        config_name="test",
        config_hash="hash",
        device="cpu",
        seed=0,
        tool_scale_xyz=(1.0, 1.0, 1.0),
    )

    assert pair_cfg.rejection_apply_tangent_gaussian is True
    assert pair_cfg.geometry_config().rejection_apply_tangent_gaussian is True


def test_rejection_checks_legality_after_perturbation(monkeypatch):
    monkeypatch.setattr(
        gen_contact,
        "sample_upright_rotations",
        lambda count, *_args: torch.eye(3).repeat(count, 1, 1),
    )
    original_randn = torch.randn

    def fake_randn(*shape, **kwargs):
        if len(shape) == 3:
            return torch.full(shape, 1000.0, **kwargs)
        return original_randn(*shape, **kwargs) * 0.0

    monkeypatch.setattr(torch, "randn", fake_randn)

    def reject_perturbed(points, *_args):
        assert float(points.abs().max()) > 1.0
        return torch.full((points.shape[0],), -0.01)

    monkeypatch.setattr(gen_contact, "query_sdf_grid", reject_perturbed)

    with pytest.raises(RuntimeError, match="accepted=0 required=1"):
        gen_contact.rejection_sample_candidates(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _cfg(
                B=1,
                M=1,
                chunk_B=1,
                rejection_max_rounds=1,
                rejection_apply_tangent_gaussian=True,
            ),
            sdf_grid=None,
            bbox_min=None,
            bbox_max=None,
            P_anchor=torch.tensor([[0.0, 0.0, 0.0]]),
        )

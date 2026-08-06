from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import pytest
import torch

from configs.config_contact_gen import (
    CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
    CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
    CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
    PENETRATION_CHECK_BIDIRECTIONAL,
    ContactGenCfg,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    PARALLEL_NONPENETRATING_1M_CONTACT_DATASET,
    PARALLEL_PAPER_1M_CONTACT_DATASET,
    parallel_concavity_sdf_regression_rl_cfg,
    parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg,
    parallel_new200_proven_nonpenetrating_recipe_rl_cfg,
    parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg,
    parallel_paper_contact_quality_rl_cfg,
    parallel_proven_nonpenetrating_recipe_paper_dataset_rl_cfg,
)
from configs.panda_comparison_common import configure_full_yes_comparison
from configs.experiments.ce_prl_unicorn_d1_full_nonpenetrating_contact_1mm_no_scale_dgn_5k import (
    EXP_CFG as NONPENETRATING_1MM_NO_SCALE_CFG,
)
from configs.experiments.ce_prl_unicorn_d1_full_paper_contact_1mm_no_scale_dgn_5k import (
    EXP_CFG as PAPER_1MM_NO_SCALE_CFG,
)
from contact_generation.batch_generate import (
    ContactGenerationResult,
    _MeshRecord,
    _SelectedToolPairCatalog,
    _assert_complete_geometry_dataset,
    sample_pairs,
)
from contact_generation.gen_contact import (
    GeometryContactConfig,
    intersecting_anchor_sample_candidates,
    tangent_gaussian_sample_candidates,
)
from pretrain.model import ContactDiffusionModel, _aggregate_sdf
from utils.io import to_plain_data
from utils.config.hash_payloads import (
    contact_payload,
    model_payload,
    pretrain_payload,
    rl_payload,
)


VARIANTS = (
    "paper_contact",
    "paper_head",
    "raw_contact",
    "nonpenetrating_contact",
)


def _variant_cfg(variant: str):
    return parallel_paper_contact_quality_rl_cfg(
        f"test_{variant}",
        contact_variant=variant,
    )


def _without_keys(payload: dict, *keys: str) -> dict:
    result = deepcopy(payload)
    for key in keys:
        result.pop(key, None)
    return result


def test_balanced_pair_plan_is_exact_and_deterministic():
    objects = tuple(
        _MeshRecord(name=f"object_{index}", mesh_stem=f"o{index}", path=f"/o/{index}.obj")
        for index in range(20)
    )
    tools = tuple(
        _MeshRecord(name=f"gripper_{index}", mesh_stem=f"g{index}", path=f"/g/{index}.obj")
        for index in range(200)
    )
    catalog = _SelectedToolPairCatalog(objects=objects, tools=tools)

    first = sample_pairs(catalog, 1000, 0, balanced_tool_pairs=True)
    second = sample_pairs(catalog, 1000, 0, balanced_tool_pairs=True)

    assert first == second
    assert len(first) == 1000
    counts = Counter(pair[2] for pair in first)
    assert set(counts.values()) == {5}
    assert len(counts) == 200
    assert len({(pair[2], pair[3]) for pair in first}) == 1000


def test_raw_contact_sampler_uses_only_explicit_tip_anchors(monkeypatch):
    cfg = GeometryContactConfig(
        object_mesh_path="/unused/object.obj",
        tool_mesh_path="/unused/tool.obj",
        tools_json_path="/unused/tools.json",
        object_id="object",
        tool_id="tool",
        config_name="test",
        config_hash="test",
        output_path="/unused/output.pt",
        device="cpu",
        seed=0,
        B=16,
        M=1,
        K=1,
        sdf_grid_res=8,
        sdf_padding=0.03,
        chunk_B=4,
        tool_scale_xyz=(1.0, 1.0, 1.0),
        object_scale_range=(1.0, 1.0),
    )
    monkeypatch.setattr(
        "contact_generation.gen_contact.random_rotation_matrices",
        lambda count, device: torch.eye(3).repeat(count, 1, 1),
    )
    full_tool = torch.tensor([[100.0, 0.0, 0.0]])
    tip = torch.tensor([[1.0, 2.0, 3.0]])
    object_surface = torch.tensor([[4.0, 5.0, 6.0]])

    result = intersecting_anchor_sample_candidates(
        full_tool,
        object_surface,
        cfg,
        P_anchor=tip,
    )

    assert torch.equal(
        result["contact_pt_tool_T"],
        tip.repeat(cfg.B, 1),
    )


def test_contact_quality_variants_share_training_and_rl_contracts():
    configs = {variant: _variant_cfg(variant) for variant in VARIANTS}
    for cfg in configs.values():
        cfg.validate()
        assert cfg.paths_yaml == GENERATED_GRIPPER_NEW_PATHS_YAML
        assert cfg.num_gpus == 8
        assert cfg.contact_gen.num_pairs == 1000
        assert cfg.contact_gen.B == 500
        assert cfg.contact_gen.num_pairs * cfg.contact_gen.B == 500_000
        assert cfg.contact_gen.balanced_tool_pairs is True
        assert cfg.contact_gen.require_complete is True
        assert cfg.contact_gen.precompute_convex_union_labels is False
        assert cfg.contact_gen.precompute_mesh_sdf is True
        assert cfg.contact_gen.geometry_only is True
        assert cfg.model.tce.vit_depth == 1
        assert cfg.model.tce.vit_attention_mode == "joint_self"
        assert cfg.pretrain.epochs == 50
        assert cfg.pretrain.batch.batch_size * cfg.num_gpus == 1024
        assert cfg.pretrain.optimizer.name == "sam"
        assert cfg.pretrain.optimizer.learning_rate == 2e-4
        assert cfg.pretrain.optimizer.min_learning_rate == 1e-6
        assert cfg.pretrain.optimizer.max_gradient_norm == 1000.0
        assert cfg.pretrain.optimizer.weight_decay == 0.001
        assert cfg.pretrain.unicorn.decoder_type == "paper_cmlp_cbn"
        assert cfg.pretrain.unicorn.augment.paper_pair_augmentation is True
        assert cfg.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
        assert cfg.pretrain.unicorn.label.contact_eps == 0.0
        assert cfg.pretrain.unicorn.augment.noise_std == 0.0
        assert cfg.rl.ppo.max_iterations == 10_000
    for variant in ("paper_head", "raw_contact", "nonpenetrating_contact"):
        assert configs[variant].contact_gen.require_tool_tip_anchor is True
    assert configs["paper_contact"].contact_gen.require_tool_tip_anchor is False

    base = configs["paper_contact"]
    base_pretrain = _without_keys(
        to_plain_data(base.pretrain), "name", "wandb_run_name"
    )
    base_model = _without_keys(to_plain_data(base.model), "name")
    base_rl = _without_keys(to_plain_data(base.rl), "name")
    base_rl["launch"].pop("run_name", None)
    for variant in VARIANTS[1:]:
        cfg = configs[variant]
        variant_pretrain = _without_keys(
            to_plain_data(cfg.pretrain), "name", "wandb_run_name"
        )
        assert variant_pretrain == base_pretrain
        assert _without_keys(to_plain_data(cfg.model), "name") == base_model
        rl = _without_keys(to_plain_data(cfg.rl), "name")
        rl["launch"].pop("run_name", None)
        assert rl == base_rl


def test_only_contact_construction_controls_differ():
    configs = {variant: _variant_cfg(variant) for variant in VARIANTS}
    expected_modes = {
        "paper_contact": CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
        "paper_head": CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
        "raw_contact": CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
        "nonpenetrating_contact": CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
    }
    algorithm_controls = {
        "name",
        "contact_geometry_mode",
        "M",
        "rejection_refill",
        "rejection_max_rounds",
        "penetration_check_mode",
        "penetration_eps",
        "require_tool_tip_anchor",
        "rejection_apply_tangent_gaussian",
    }
    common = None
    for variant, cfg in configs.items():
        assert cfg.contact_gen.contact_geometry_mode == expected_modes[variant]
        if variant == "nonpenetrating_contact":
            assert (
                cfg.contact_gen.penetration_check_mode
                == PENETRATION_CHECK_BIDIRECTIONAL
            )
            assert cfg.contact_gen.penetration_eps == 0.002
            assert cfg.contact_gen.rejection_apply_tangent_gaussian is True
        payload = to_plain_data(cfg.contact_gen)
        for key in algorithm_controls:
            payload.pop(key, None)
        if common is None:
            common = payload
        else:
            assert payload == common


def test_historical_nonpenetrating_1mm_recipe_stays_explicit():
    base = _variant_cfg("nonpenetrating_contact")
    configure_full_yes_comparison(base)
    historical = NONPENETRATING_1MM_NO_SCALE_CFG
    historical.validate()

    historical_contact = contact_payload(historical.contact_gen)
    active_contact = contact_payload(base.contact_gen)
    assert active_contact.pop("rejection_apply_tangent_gaussian") is True
    assert active_contact.pop("penetration_eps") == 0.002
    assert historical_contact.pop("penetration_eps") == 5e-4
    assert historical_contact == active_contact
    assert model_payload(historical) == model_payload(base)
    assert rl_payload(historical.rl) == rl_payload(base.rl)
    assert historical.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
    assert historical.pretrain.unicorn.label.contact_eps == 0.002
    assert base.pretrain.unicorn.label.contact_eps == 0.0
    assert historical.pretrain.unicorn.augment.rotation_range == (
        -3.141592653589793,
        3.141592653589793,
    )
    assert historical.pretrain.unicorn.augment.translation_range == (-0.1, 0.1)
    assert historical.pretrain.unicorn.augment.log_scale_range == (0.0, 0.0)
    assert historical.pretrain.unicorn.augment.noise_std == 0.001
    assert base.pretrain.unicorn.augment.noise_std == 0.0

    base_pretrain = pretrain_payload(base.pretrain)
    historical_pretrain = pretrain_payload(historical.pretrain)
    historical_pretrain["unicorn"]["augment"]["noise_std"] = 0.0
    historical_pretrain["unicorn"]["label"]["contact_eps"] = 0.0
    assert historical_pretrain == base_pretrain


def test_historical_paper_1mm_recipe_stays_explicit():
    base = _variant_cfg("paper_contact")
    configure_full_yes_comparison(base)
    historical = PAPER_1MM_NO_SCALE_CFG
    historical.validate()

    assert contact_payload(historical.contact_gen) == contact_payload(
        base.contact_gen
    )
    assert model_payload(historical) == model_payload(base)
    assert rl_payload(historical.rl) == rl_payload(base.rl)
    assert historical.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
    assert historical.pretrain.unicorn.label.contact_eps == 0.002
    assert base.pretrain.unicorn.label.contact_eps == 0.0
    assert historical.pretrain.unicorn.augment.rotation_range == (
        -3.141592653589793,
        3.141592653589793,
    )
    assert historical.pretrain.unicorn.augment.translation_range == (-0.1, 0.1)
    assert historical.pretrain.unicorn.augment.log_scale_range == (0.0, 0.0)
    assert historical.pretrain.unicorn.augment.noise_std == 0.001
    assert base.pretrain.unicorn.augment.noise_std == 0.0

    base_pretrain = pretrain_payload(base.pretrain)
    historical_pretrain = pretrain_payload(historical.pretrain)
    historical_pretrain["unicorn"]["augment"]["noise_std"] = 0.0
    historical_pretrain["unicorn"]["label"]["contact_eps"] = 0.0
    assert historical_pretrain == base_pretrain


def test_one_million_case_controls_match_except_contact_geometry_and_labels():
    paper = parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg(
        "test_paper_1m",
        contact_variant="paper_contact",
    )
    nonpenetrating = parallel_paper_contact_quality_1m_1mm_no_scale_rl_cfg(
        "test_nonpenetrating_1m",
        contact_variant="nonpenetrating_contact",
    )

    for cfg in (paper, nonpenetrating):
        configure_full_yes_comparison(cfg)
        cfg.validate()
        assert cfg.paths_yaml == GENERATED_GRIPPER_NEW_PATHS_YAML
        assert cfg.num_gpus == 8
        assert cfg.contact_gen.num_pairs == 2000
        assert cfg.contact_gen.B == 500
        assert cfg.contact_gen.num_pairs * cfg.contact_gen.B == 1_000_000
        assert cfg.contact_gen.balanced_tool_pairs is True
        assert cfg.pretrain.epochs == 50
        assert cfg.pretrain.unicorn.augment.log_scale_range == (0.0, 0.0)
        assert cfg.pretrain.unicorn.augment.noise_std == 0.001
        assert cfg.model.tce.vit_depth == 1
        assert cfg.model.tce.vit_attention_mode == "joint_self"

    paper_pretrain = _without_keys(
        to_plain_data(paper.pretrain), "name", "wandb_run_name"
    )
    nonpenetrating_pretrain = _without_keys(
        to_plain_data(nonpenetrating.pretrain), "name", "wandb_run_name"
    )
    assert nonpenetrating_pretrain == paper_pretrain
    assert paper.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
    assert paper.pretrain.unicorn.label.contact_eps == 0.002
    assert nonpenetrating.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
    assert nonpenetrating.pretrain.unicorn.label.contact_eps == 0.002


def test_proven_nonpenetrating_recipe_paper_control_changes_only_dataset():
    base = parallel_proven_nonpenetrating_recipe_paper_dataset_rl_cfg(
        "test_paper_dataset"
    )
    reference = parallel_depth1_full_attention_nonpenetrating_unicorn_rl_cfg(
        "test_nonpenetrating_reference"
    )

    assert base.pretrain.dataset_manifest == PARALLEL_PAPER_1M_CONTACT_DATASET
    base_pretrain = to_plain_data(base.pretrain)
    reference_pretrain = to_plain_data(reference.pretrain)
    for payload in (base_pretrain, reference_pretrain):
        payload.pop("name", None)
        payload.pop("wandb_run_name", None)
        payload.pop("dataset_manifest", None)
    assert base_pretrain == reference_pretrain
    assert _without_keys(to_plain_data(base.model), "name") == _without_keys(
        to_plain_data(reference.model), "name"
    )
    base_rl = _without_keys(to_plain_data(base.rl), "name")
    reference_rl = _without_keys(to_plain_data(reference.rl), "name")
    base_rl["launch"].pop("run_name", None)
    reference_rl["launch"].pop("run_name", None)
    assert base_rl == reference_rl
    assert to_plain_data(base.contact_gen) == to_plain_data(
        reference.contact_gen
    )
    assert base.pretrain.unicorn.label.source == "mesh_sdf"
    assert base.pretrain.unicorn.label.contact_eps == 0.002
    assert base.pretrain.epochs == 40
    assert base.pretrain.optimizer.name == "adamw"
    assert base.model.tce.vit_depth == 1
    assert base.model.tce.vit_attention_mode == "joint_self"


def test_new200_legacy_nonpenetrating_recipe_trains_pretrain_and_rl():
    cfg = parallel_new200_proven_nonpenetrating_recipe_rl_cfg(
        "test_new200_legacy_nonpenetrating"
    )
    configure_full_yes_comparison(cfg)
    cfg.validate()

    assert cfg.paths_yaml == GENERATED_GRIPPER_NEW_PATHS_YAML
    assert (
        cfg.pretrain.dataset_manifest
        == PARALLEL_NONPENETRATING_1M_CONTACT_DATASET
    )
    assert cfg.contact_gen.enabled is False
    assert cfg.contact_gen.regenerate is False
    assert cfg.pretrain.enabled is True
    assert cfg.pretrain.retrain is True
    assert cfg.model.pretrained_encoder.checkpoint_path is None
    assert cfg.model.tce.vit_depth == 1
    assert cfg.model.tce.vit_attention_mode == "joint_self"
    assert cfg.pretrain.epochs == 40
    assert cfg.pretrain.optimizer.name == "adamw"
    assert cfg.pretrain.optimizer.learning_rate == 3e-4
    assert cfg.pretrain.condition_normalization is True
    assert cfg.pretrain.unicorn.decoder_type == "relu_mlp"
    assert cfg.pretrain.unicorn.label.source == "mesh_sdf"
    assert cfg.pretrain.unicorn.label.contact_eps == 0.002
    assert cfg.rl.enabled is True
    assert cfg.num_gpus == 8


def test_tangent_sampler_reaches_sampled_surface_without_dense_sdf():
    tool = torch.tensor(
        [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=torch.float32,
    )
    obj = torch.tensor(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    cfg = GeometryContactConfig(
        object_mesh_path="object.obj",
        tool_mesh_path="tool.obj",
        tools_json_path="tools.json",
        object_id="object",
        tool_id="tool",
        config_name="test",
        config_hash="hash",
        output_path="unused.pt",
        device="cpu",
        seed=3,
        B=8,
        M=1,
        K=4,
        sdf_grid_res=8,
        sdf_padding=0.03,
        chunk_B=4,
        tool_scale_xyz=(1.0, 1.0, 1.0),
        object_scale_range=(1.0, 1.0),
        contact_geometry_mode=CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
        tangent_translation_noise_std=0.0,
        tangent_rotation_noise_std_rad=0.0,
    )

    torch.manual_seed(cfg.seed)
    result = tangent_gaussian_sample_candidates(
        tool,
        obj,
        cfg,
        P_anchor=tool,
    )
    placed = (
        torch.einsum("bij,kj->bki", result["tool_rotation_E"], tool)
        + result["tool_translation_E"][:, None, :]
    )
    distances = torch.cdist(placed, obj.unsqueeze(0).expand(cfg.B, -1, -1))
    assert torch.all(distances.flatten(1).amin(dim=1) < 1e-5)
    assert torch.isnan(result["initial_min_sdf"]).all()


def test_paper_head_tangent_sampler_uses_only_explicit_tip_anchors(monkeypatch):
    cfg = GeometryContactConfig(
        object_mesh_path="object.obj",
        tool_mesh_path="tool.obj",
        tools_json_path="tools.json",
        object_id="object",
        tool_id="tool",
        config_name="test",
        config_hash="hash",
        output_path="unused.pt",
        device="cpu",
        seed=0,
        B=4,
        M=1,
        K=2,
        sdf_grid_res=8,
        sdf_padding=0.03,
        chunk_B=4,
        tool_scale_xyz=(1.0, 1.0, 1.0),
        object_scale_range=(1.0, 1.0),
        contact_geometry_mode=CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
        require_tool_tip_anchor=True,
        tangent_translation_noise_std=0.0,
        tangent_rotation_noise_std_rad=0.0,
    )
    monkeypatch.setattr(
        "contact_generation.gen_contact.random_rotation_matrices",
        lambda count, device: torch.eye(3).repeat(count, 1, 1),
    )
    full_tool = torch.tensor(
        [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    tip = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    obj = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)

    result = tangent_gaussian_sample_candidates(
        full_tool,
        obj,
        cfg,
        P_anchor=tip,
    )

    assert torch.equal(
        result["contact_pt_tool_T"],
        tip.repeat(cfg.B, 1),
    )
    assert (
        result["geometry_diagnostics"]["anchor_source"]
        == "contact_tip_mesh"
    )


def test_paper_contact_model_consumes_precomputed_labels_without_mesh_backend():
    model = ContactDiffusionModel(
        num_pts=32,
        patch_size=8,
        encoder_channel=32,
        vit_depth=1,
        vit_heads=4,
        vit_attention_mode="joint_self",
        encoder_input_centering="object_center",
        enabled_heads=("contact",),
        contact_label_source="precomputed_convex_union",
        contact_decoder_type="paper_cmlp_cbn",
        contact_decoder_hidden=(128, 128),
        contact_pair_augmentation=True,
        contact_aug_rotation_range=(-3.141592653589793, 3.141592653589793),
        contact_aug_translation_range=(-0.1, 0.1),
        contact_aug_log_scale_range=(-1.0, 1.0),
        contact_aug_noise_std=0.01,
    )
    model.train()
    batch, points = 2, 32
    tool = torch.randn(batch, 1, points, 3)
    obj = torch.randn(batch, 1, points, 3)
    tool_labels = torch.zeros(batch, points, dtype=torch.bool)
    obj_labels = torch.zeros(batch, points, dtype=torch.bool)
    tool_labels[:, :4] = True
    obj_labels[:, 4:8] = True

    loss, metrics = model(
        tool_points_E_k=tool,
        object_points_E_k=obj,
        rel_tool_object_t_k=torch.zeros(batch, 1, 3),
        cond_tool_post_delta9d=torch.zeros(batch, 9),
        cond_object_post_delta9d=torch.zeros(batch, 9),
        physics=torch.zeros(batch, 5),
        tool_point_inside_object=tool_labels,
        object_point_inside_tool=obj_labels,
        target_tool_denoise_pose9d_k=torch.zeros(batch, 0, 9),
        target_object_post_delta9d=torch.zeros(batch, 9),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["contact_total_patches"] == 16.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_contact_model_consumes_precomputed_mesh_sdf_without_mesh_backend():
    model = ContactDiffusionModel(
        num_pts=32,
        patch_size=8,
        encoder_channel=32,
        vit_depth=1,
        vit_heads=4,
        vit_attention_mode="joint_self",
        encoder_input_centering="object_center",
        enabled_heads=("contact",),
        contact_label_source="precomputed_mesh_sdf",
        contact_eps=0.002,
        contact_decoder_type="paper_cmlp_cbn",
        contact_decoder_hidden=(128, 128),
    )
    model.train()
    batch, points = 2, 32
    tool_sdf = torch.full((batch, points), 0.01)
    object_sdf = torch.full((batch, points), 0.01)
    tool_sdf[:, :4] = 0.001
    object_sdf[:, 4:8] = -0.001

    loss, metrics = model(
        tool_points_E_k=torch.randn(batch, 1, points, 3),
        object_points_E_k=torch.randn(batch, 1, points, 3),
        rel_tool_object_t_k=torch.zeros(batch, 1, 3),
        cond_tool_post_delta9d=torch.zeros(batch, 9),
        cond_object_post_delta9d=torch.zeros(batch, 9),
        physics=torch.zeros(batch, 5),
        tool_point_object_signed_sdf=tool_sdf,
        object_point_tool_signed_sdf=object_sdf,
        target_tool_denoise_pose9d_k=torch.zeros(batch, 0, 9),
        target_object_post_delta9d=torch.zeros(batch, 9),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["contact_total_patches"] == 16.0


def test_concavity_sdf_experiment_reuses_contacts_and_regresses_patch_minimum():
    sdf_cfg = parallel_concavity_sdf_regression_rl_cfg("test_concavity_sdf")
    contact_cfg = parallel_paper_contact_quality_rl_cfg(
        "test_concavity_contact",
        contact_variant="nonpenetrating_contact_concavity_biased",
    )

    assert contact_payload(sdf_cfg.contact_gen) == contact_payload(
        contact_cfg.contact_gen
    )
    assert sdf_cfg.pretrain.enabled_heads == ["sdf"]
    assert sdf_cfg.pretrain.tasks.sdf is True
    assert sdf_cfg.pretrain.tasks.contact is False
    assert sdf_cfg.pretrain.sdf_head_mode == "patch"
    assert sdf_cfg.pretrain.decoder_pooling == "min"
    assert sdf_cfg.pretrain.unicorn.label.source == "precomputed_mesh_sdf"

    sdf = torch.tensor([[0.003, -0.001, 0.004, 0.002]])
    patch_idx = torch.tensor([[[0, 1], [2, 3]]])
    assert torch.equal(
        _aggregate_sdf(sdf, patch_idx, "min"),
        torch.tensor([[-0.001, 0.002]]),
    )


def test_sdf_model_consumes_precomputed_distances_without_mesh_backend():
    model = ContactDiffusionModel(
        num_pts=32,
        patch_size=8,
        encoder_channel=32,
        vit_depth=1,
        vit_heads=4,
        vit_attention_mode="joint_self",
        encoder_input_centering="object_center",
        enabled_heads=("sdf",),
        head_mode="patch",
        patch_agg="min",
        contact_label_source="precomputed_mesh_sdf",
        contact_pair_augmentation=True,
        contact_aug_rotation_range=(0.0, 0.0),
        contact_aug_translation_range=(0.0, 0.0),
        contact_aug_log_scale_range=(0.0, 0.0),
        contact_aug_noise_std=0.0,
    )
    model.train()
    batch, points = 2, 32
    loss, metrics = model(
        tool_points_E_k=torch.randn(batch, 1, points, 3),
        object_points_E_k=torch.randn(batch, 1, points, 3),
        rel_tool_object_t_k=torch.zeros(batch, 1, 3),
        tool_point_object_signed_sdf=torch.randn(batch, points) * 0.002,
        object_point_tool_signed_sdf=torch.randn(batch, points) * 0.002,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "tool_sdf_loss",
        "obj_sdf_loss",
        "sdf_loss",
        "total_loss",
    }


def test_strict_geometry_audit_checks_exact_case_and_label_counts(tmp_path):
    cfg = ContactGenCfg(
        B=3,
        num_surface_pts=4,
        precompute_convex_union_labels=True,
    )
    for index in range(2):
        path = tmp_path / f"tool_{index}" / "object.pt.candidate.manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "num_candidates": 3,
                    "precomputed_convex_union_labels": True,
                    "tool_point_label_shape": [3, 4],
                    "object_point_label_shape": [3, 4],
                }
            ),
            encoding="utf-8",
        )
    result = ContactGenerationResult(
        artifact_dir=tmp_path,
        num_pairs=2,
        num_poses=1,
        ok=2,
        fail=0,
        skipped=0,
    )
    _assert_complete_geometry_dataset(result, cfg)

    broken = tmp_path / "tool_1" / "object.pt.candidate.manifest.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["tool_point_label_shape"] = [2, 4]
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="label shape mismatch"):
        _assert_complete_geometry_dataset(result, cfg)


def test_strict_geometry_audit_checks_precomputed_mesh_sdf(tmp_path):
    cfg = ContactGenCfg(
        B=3,
        num_surface_pts=4,
        precompute_mesh_sdf=True,
    )
    path = tmp_path / "tool" / "object.pt.candidate.manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "num_candidates": 3,
                "precomputed_mesh_sdf": True,
                "tool_point_sdf_shape": [3, 4],
                "object_point_sdf_shape": [3, 4],
            }
        ),
        encoding="utf-8",
    )
    result = ContactGenerationResult(
        artifact_dir=tmp_path,
        num_pairs=1,
        num_poses=1,
        ok=1,
        fail=0,
        skipped=0,
    )
    _assert_complete_geometry_dataset(result, cfg)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["object_point_sdf_shape"] = [2, 4]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SDF shape mismatch"):
        _assert_complete_geometry_dataset(result, cfg)

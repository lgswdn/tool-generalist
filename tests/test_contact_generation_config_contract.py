from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from configs.config_contact_gen import ContactGenCfg, ContactPhysicsCfg
from configs.config_exp import ExpCfg
from utils.experiment.stages import all_stages


ROOT = Path(__file__).resolve().parents[1]


def _field_names(cls):
    return {field.name for field in fields(cls)}


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_contact_generation_business_config_lives_in_exp_config():
    assert {
        "object_scale_range",
        "num_surface_pts",
        "sdf_grid_res",
        "sdf_padding",
        "B",
        "M",
        "chunk_B",
        "epsilon",
        "floor_eps",
        "upright_threshold",
        "contact_geometry_mode",
        "rotation_selection",
        "tool_source",
        "object_tool_manifest",
        "shard_count",
        "shard_index",
        "penetration_eps",
        "skip_existing",
        "physics",
    }.issubset(_field_names(ContactGenCfg))
    assert not {
        "tool_scale",
        "stabilize_steps",
        "postcontact_steps",
        "object_mass_range",
        "tool_mass_range",
        "object_friction_range",
        "tool_friction_range",
        "ground_friction_range",
    }.intersection(_field_names(ContactGenCfg))
    assert {
        "runner",
        "num_workers",
        "t_stabilize",
        "t_postcontact",
        "post_delta_seed",
        "post_delta_translation_min",
        "post_delta_translation_max",
        "post_delta_rotation_max_rad",
        "post_tool_reach_translation_eps",
        "post_tool_reach_rotation_eps_rad",
        "post_object_table_z_min",
        "post_linear_velocity_eps",
        "post_angular_velocity_eps",
    }.issubset(_field_names(ContactPhysicsCfg))
    assert not {"debug_dir", "headless", "close_after_run"}.intersection(
        _field_names(ContactPhysicsCfg)
    )

    assert "num_" + "contact_candidates" not in _field_names(ContactGenCfg)
    assert "contact_" + "eps" not in _field_names(ContactGenCfg)
    assert ContactPhysicsCfg().runner == "isaac"
    assert ContactPhysicsCfg().num_workers == 1


def test_contact_generation_directory_has_only_core_files():
    entries = {path.name for path in (ROOT / "contact_generation").iterdir()}
    assert entries == {
        "batch_generate.py",
        "gen_contact.py",
        "stabilize_contact.py",
        "gen_postcontact.py",
    }


def test_generator_and_batch_have_no_business_cli_or_old_config_dependency():
    for rel_path in (
        "contact_generation/gen_contact.py",
        "contact_generation/stabilize_contact.py",
        "contact_generation/gen_postcontact.py",
        "contact_generation/batch_generate.py",
        "utils/contact/stabilize.py",
    ):
        source = _source(rel_path)
        assert "argparse" not in source
        assert ".add_argument" not in source
        assert "CONTACT_GEN" not in source
        assert "from .config import" not in source
        assert "contact_generation.config" not in source


def test_batch_generate_lazy_imports_generator_and_exposes_exp_entrypoint():
    tree = ast.parse(_source("contact_generation/batch_generate.py"))
    top_level_generator_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module in {"generator", "gen_contact", "stabilize", "gen_postcontact"}
    ]
    assert top_level_generator_imports == []
    assert "def run_contact_generation(" in _source("contact_generation/batch_generate.py")
    assert "def _load_generator_api(" in _source("contact_generation/batch_generate.py")
    assert "phase == \"geometry\"" in _source("contact_generation/batch_generate.py")
    assert "phase == \"stabilize\"" in _source("contact_generation/batch_generate.py")
    assert "phase == \"postcontact\"" in _source("contact_generation/batch_generate.py")


def test_contact_generation_algorithm_contract_is_static_and_explicit():
    gen_contact = _source("contact_generation/gen_contact.py")
    stabilize = _source("utils/contact/stabilize.py")
    isaac = _source("utils/contact/isaac.py")
    stabilize_contact = _source("contact_generation/stabilize_contact.py")
    gen_postcontact = _source("contact_generation/gen_postcontact.py")
    batch = _source("contact_generation/batch_generate.py")

    assert "def generate_contact_candidates(" in gen_contact
    assert "def rejection_sample_candidates(" in gen_contact
    assert "def bbox_translation_nearest_sample_candidates(" in gen_contact
    assert "CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST" in gen_contact
    assert "_nearest_surface_pairs" in gen_contact
    assert "_safe_contact_alpha" in gen_contact
    assert "sample_upright_rotations(M, device" in gen_contact
    assert "rotations = random_rotation_matrices(M, device)" in gen_contact
    assert '"bmij,kj->bmki"' in gen_contact
    assert "contact_" + "ok" not in gen_contact
    assert "valid = floor_ok & penetration_ok" in gen_contact
    assert "final_close_enough = final_min_sdf <= float(cfg.epsilon)" in gen_contact
    assert "penetration_depth <= float(cfg.penetration_eps)" in gen_contact
    assert "ROTATION_SELECTION_RANDOM_LEGAL" in gen_contact
    assert "contact_mode_prob" not in gen_contact
    assert "choose_head" not in gen_contact
    assert "candidate_" + "contact_" + "distance" not in gen_contact
    assert "contact_" + "distance" not in gen_contact
    assert "centralize_points_by_bbox" in gen_contact
    assert "assert_adjusted_decomposed_mesh_path" in gen_contact
    assert "sample_object_pose_and_ground" in gen_contact
    assert "object_bbox_center_E[2] = -rotated[:, 2].min()" in gen_contact
    assert "contact_candidate_v1" in gen_contact
    assert "def run_geometry_contact_pair(" in gen_contact
    assert "def load_candidate_artifact(" in gen_contact

    assert "class IsaacPhysicsRunner" in stabilize
    assert "from .isaac import IsaacSimAdapter" in stabilize
    assert "def run_batch(" in stabilize
    assert "getattr(adapter, \"run_batch\"" in stabilize
    assert "adapter.run_candidate" not in stabilize
    assert "MockPhysicsRunner" not in stabilize
    assert "UnavailablePhysicsRunner" not in stabilize
    assert "contact_" + "distance_min[index]) >" not in stabilize
    assert "penetration_depth_max[index]) > float(cfg.penetration_eps)" in stabilize
    assert "postcontact_steps[index]) <= 0" in stabilize
    assert "cfg.run_postcontact and int(postcontact_steps[index]) <= 0" in stabilize
    assert "def run_batch(" in isaac
    assert "\"/World/Env_" in isaac
    assert "batch stabilize step active_envs" in isaac
    assert "batch postcontact step active_envs" in isaac
    assert "UsdPhysics.MeshCollisionAPI.Apply" in isaac
    assert "\"convexHull\"" in isaac

    assert "contact_stabilized_success_v1" in stabilize_contact
    assert "def run_stabilize_contact_pair(" in stabilize_contact
    assert "run_postcontact=False" in stabilize_contact

    assert "def assemble_contact_pt_env_v1(" in gen_postcontact
    assert "schema_version\": \"contact_pt_env_v1\"" in gen_postcontact
    assert "def run_contact_pair(" in gen_postcontact
    assert "load_stabilized_success_artifact" in gen_postcontact
    assert "generate_contact_candidates" not in gen_postcontact
    assert "contact_manifest_v1" in gen_postcontact
    assert "contact_" + "eps" not in gen_postcontact
    assert "tool_asset_" + "metadata" + "_hash" not in gen_postcontact

    assert "contact_config_hash(exp_cfg)" in batch
    assert "configured_workers={contact_cfg.physics.num_workers}" in batch
    assert "\"tools.tool_asset_" + "metadata_json\"" not in batch
    assert "load_selected_tool_ids(contact_paths.tools_selected_json)" in batch


def test_core_contact_modules_do_not_import_heavy_modules_at_top_level():
    for rel_path in (
        "contact_generation/gen_contact.py",
        "contact_generation/stabilize_contact.py",
        "contact_generation/gen_postcontact.py",
        "utils/contact/stabilize.py",
    ):
        tree = ast.parse(_source(rel_path))
        imports = [
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all("torch" not in item for item in imports)
        assert all("kaolin" not in item for item in imports)
        assert all("trimesh" not in item for item in imports)
        assert all("isaac" not in item.lower() for item in imports)
        assert all("omni" not in item.lower() for item in imports)
        assert all("pxr" not in item.lower() for item in imports)


def test_experiment_stages_publish_contact_stage_entrypoint():
    cfg = ExpCfg()
    stages = {stage.name: stage for stage in all_stages(cfg)}
    assert stages["contact_gen"].entrypoint == "utils.experiment.contact_stage:run_contact_stage"

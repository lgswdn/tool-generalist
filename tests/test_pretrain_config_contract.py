from __future__ import annotations

import ast
from pathlib import Path

from configs.config_pretrain import (
    POST_CFG,
    PretrainCfg,
    SDF_CFG,
    SDF_DIFF_CFG,
    SDF_DIFF_POST_CFG,
    SDF_POST_CFG,
    clone_cfg,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _tree(rel_path: str) -> ast.Module:
    return ast.parse(_source(rel_path))


def _top_level_names(rel_path: str, node_type: type[ast.AST]) -> set[str]:
    return {node.name for node in _tree(rel_path).body if isinstance(node, node_type)}


def test_pretrain_model_contract_is_tce_only_and_uses_signed_sdf_targets():
    source = _source("pretrain/model.py")
    sdf_source = _source("utils/geometry/sdf.py")
    classes = _top_level_names("pretrain/model.py", ast.ClassDef)

    assert {
        "ContactDiffusionModel",
        "ConditionQueryGenerator",
        "Pose9DHead",
        "TCEPointCloudEncoder",
        "TCEPointCloudEncoderCfg",
    } <= classes
    assert "DenoisingHead" not in classes
    assert "SDFPointCloudEncoder" not in source
    assert "SDFEncoderCfg" not in source
    assert "sys.path" not in source
    assert "importlib" not in source
    assert "rpdiff" not in source.lower()
    assert "_approximate_mutual_sdf" not in source
    assert "_temporary_signed_mesh_sdf_placeholder" not in source
    assert "_fail_fast_signed_mesh_sdf_required" not in source
    assert "from utils.geometry.sdf import mutual_signed_sdf_labels_env_frame" in source
    assert "mutual_signed_sdf_labels_env_frame(" in source
    assert "object_mesh_vertices" in source
    assert "tool_mesh_vertices" in source
    assert "unsigned distance fallback is forbidden" in sdf_source
    assert "torch.cdist" not in sdf_source
    assert "_pose9d_loss" in source
    assert "_pose_denoising_loss" not in source


def test_pretrain_tce_and_decoder_static_algorithm_contract():
    source = _source("pretrain/model.py")

    for token in [
        "_fps_indices",
        "_knn_patch_indices",
        "relative_patch_coords",
        "patch_pointnet",
        "patch_center_pos",
        "type_embedding",
        "joint_transformer",
        "query_A",
        "query_B",
        "query_C",
        "query_D",
        "query_cross_attn",
        "token_cross_attn",
        "fused_tokens=fused",
    ]:
        assert token in source


def test_diffusion_branch_does_not_condition_on_post_or_physics():
    model_source = _source("pretrain/model.py")
    train_source = _source("pretrain/train.py")
    diff_block = model_source.split('if "diff" in self.enabled_heads and K > 0:', 1)[1].split(
        'if "postcontact" in self.enabled_heads:', 1
    )[0]

    assert "_compose_condition" not in diff_block
    assert "torch.zeros(B * K, self.movement_cond_dim" in diff_block
    assert "diff_rel = rel_tool_object_t_k[:, 1:, :]" in diff_block
    assert 'require_movement = "postcontact" in cfg.enabled_heads' in train_source


def test_contact_pt_env_v1_schema_and_dataset_static_contract():
    schema_source = _source("utils/contact/schema.py")
    dataset_source = _source("pretrain/dataset.py")

    assert "CONTACT_SCHEMA_VERSION = \"contact_pt_env_v1\"" in schema_source
    assert "def load_and_validate_contact_pt" in schema_source
    assert "allow_mock: bool = False" in schema_source
    assert "require_complete: bool = True" in schema_source
    assert "mock_complete contact artifact requires allow_mock=True" not in schema_source
    assert "assert_adjusted_decomposed_mesh_path" in schema_source
    assert "scaled_mesh_bbox" in schema_source

    assert "load_and_validate_contact_pt" in dataset_source
    assert "allow_mock=self.allow_mock_physics" in dataset_source
    assert "load_scaled_sampled_surface_points" in dataset_source
    assert "load_mesh_vertices_faces" in dataset_source
    assert "_reconstruct_meshes" in dataset_source
    assert "object_mesh_vertices" in dataset_source
    assert "tool_mesh_faces" in dataset_source
    assert "sdf_gt" not in dataset_source
    assert "sdf_label" not in dataset_source
    assert "bbox_center" in dataset_source
    assert "- bbox_center" in dataset_source
    assert ".candidate.pt" in dataset_source
    assert ".physics_debug.pt" in dataset_source


def test_pretrain_train_contract_has_no_business_cli_or_runtime_defaults():
    source = _source("pretrain/train.py")
    tree = _tree("pretrain/train.py")
    functions = _top_level_names("pretrain/train.py", ast.FunctionDef)
    runtime_cfg = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PretrainRuntimeConfig"
    )
    runtime_fields = [node for node in runtime_cfg.body if isinstance(node, ast.AnnAssign)]

    assert "run_pretrain" in functions
    assert "build_runtime_config" in functions
    assert runtime_fields
    assert all(field.value is None for field in runtime_fields)
    assert "argparse" not in source
    assert ".add_argument" not in source
    assert "from config import" not in source
    assert "from dataset import" not in source
    assert "from model import" not in source
    assert "field(default" not in source
    assert "exp_cfg.pretrain" in source
    assert "exp_cfg.model" in source
    assert "exp_cfg.general" in source
    assert "artifact_dir" in source
    assert "\"sdf\" in enabled_heads" in source
    assert "Remove 'sdf' from PretrainCfg.enabled_heads" not in source
    assert "PretrainCfg.sdf_target.mode must be 'signed'" in source
    assert "object_mesh_vertices" in source
    assert "\"object_mesh_vertices\"" in source
    assert "condition_mlp_hidden_dims" in source
    assert "num_query_A" in source
    assert "full_config_hash" in source
    assert "tool_asset_" + "metadata" + "_hash" not in source
    assert "git_metadata" in source
    assert "runtime_metadata" in source
    assert "contact_generation" not in source
    assert "rsl_rl" not in source
    assert "IsaacLab" not in source


def test_pretrain_config_defaults_do_not_enable_unimplemented_sdf():
    source = _source("configs/config_pretrain.py")

    assert 'enabled_heads: list[str] = field(default_factory=lambda: ["diff", "postcontact"])' in source
    assert "sdf: bool = False" in source
    assert "pose_loss" not in source
    assert "pose9d_geodesic" not in source


def test_pretrain_sdf_preset_and_clone_contract():
    assert isinstance(SDF_CFG, PretrainCfg)
    assert SDF_CFG.enabled is True
    assert SDF_CFG.enabled_heads == ["sdf"]
    assert SDF_CFG.tasks.sdf is True
    assert SDF_CFG.tasks.diffusion is False
    assert SDF_CFG.tasks.postcontact is False

    cloned = clone_cfg(SDF_CFG)
    cloned.name = "mutated"
    cloned.enabled_heads.append("diff")
    cloned.tasks.diffusion = True

    assert SDF_CFG.name == "sdf_only"
    assert SDF_CFG.enabled_heads == ["sdf"]
    assert SDF_CFG.tasks.diffusion is False


def test_pretrain_sdf_combo_presets_are_ready():
    expected = [
        (SDF_DIFF_CFG, "sdf_diff", ["sdf", "diff"], True, True, False),
        (SDF_POST_CFG, "sdf_post", ["sdf", "postcontact"], True, False, True),
        (SDF_DIFF_POST_CFG, "sdf_diff_post", ["sdf", "diff", "postcontact"], True, True, True),
    ]
    for cfg, name, heads, sdf, diffusion, postcontact in expected:
        assert isinstance(cfg, PretrainCfg)
        assert cfg.name == name
        assert cfg.enabled is True
        assert cfg.enabled_heads == heads
        assert cfg.tasks.sdf is sdf
        assert cfg.tasks.diffusion is diffusion
        assert cfg.tasks.postcontact is postcontact

    cloned = clone_cfg(SDF_DIFF_POST_CFG)
    cloned.enabled_heads.remove("diff")
    cloned.tasks.diffusion = False

    assert SDF_DIFF_POST_CFG.enabled_heads == ["sdf", "diff", "postcontact"]
    assert SDF_DIFF_POST_CFG.tasks.diffusion is True


def test_non_diff_pretrain_config_hashes_are_stable():
    from configs.experiments.multitools_new import EXP_CFG as MULTITOOL_CFG
    from utils.artifacts.naming import encoder_artifact_name
    from utils.config.hash import config_hash

    assert config_hash(SDF_CFG) == "7304e382af818d73f72602fe3adfe65dbfc3cb51b7cfb54ea0ad550ea0cbea6b"
    assert config_hash(SDF_POST_CFG) == "c78810093bf21bfbedc10ac1f5c4ea04cd9b35fb2bfa319c64228150d76d85fc"
    assert config_hash(POST_CFG) == "47cf8aef21f15a5e7c73615d83b7d696a11487414ece3f2b3fc923dbfd9856bd"
    assert (
        encoder_artifact_name(MULTITOOL_CFG)
        == "encoder/multitool_sdf/contact_gen_multitool_new/sdf_only_multitool_sdf/"
        "dc7bb3ef18544ebfaf882026f282d0a72141c09f8021e4ad56a38d9f9a722307"
    )


def test_fork_sdf_uses_sdf_preset():
    source = _source("configs/experiments/fork_sdf.py")

    assert "from configs.config_pretrain import SDF_CFG, clone_cfg" in source
    assert "EXP_CFG.pretrain = clone_cfg(SDF_CFG)" in source
    assert "EXP_CFG.pretrain.enabled_heads" not in source
    assert "EXP_CFG.pretrain.tasks.sdf" not in source

    from configs.experiments.fork_sdf import EXP_CFG

    EXP_CFG.validate()
    assert EXP_CFG.pretrain.enabled is True
    assert EXP_CFG.pretrain.enabled_heads == ["sdf"]


def test_pretrain_static_boundaries_remove_old_entries_and_keep_stage_lazy():
    removed_entries = [
        "pretrain/config.py",
        "pretrain/config_all.py",
        "pretrain/contact_config.py",
        "pretrain/contact_gen.py",
        "pretrain/corn.py",
        "pretrain/gen_dataset.py",
        "pretrain/gen_initial.py",
        "pretrain/gen_movement_delta.py",
        "pretrain/orig_config.py",
        "pretrain/orig_model.py",
        "pretrain/orig_train.py",
        "pretrain/validate_contact_physics.py",
        "pretrain/new_pretrain/__init__.py",
        "pretrain/new_pretrain/config.py",
        "pretrain/new_pretrain/contact_config.py",
        "pretrain/new_pretrain/contact_gen_new.py",
        "pretrain/new_pretrain/corn_dataset.py",
        "pretrain/new_pretrain/dataset.py",
        "pretrain/new_pretrain/gen_dataset.py",
        "pretrain/new_pretrain/model.py",
        "pretrain/new_pretrain/noise_utils.py",
        "pretrain/new_pretrain/train.py",
        "pretrain/new_pretrain/validate_architecture.py",
        "pretrain/new_pretrain/visualize_diffusion_checkpoint.py",
        "pretrain/new_pretrain/new_pretrain_diffusion.csv",
        "pretrain/new_pretrain/new_pretrain_diffusion.obj",
    ]
    for rel_path in removed_entries:
        assert not (ROOT / rel_path).exists()

    tree = _tree("utils/experiment/pretrain_stage.py")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("pretrain" not in ast.unparse(node) for node in imports)
    stage_source = _source("utils/experiment/pretrain_stage.py")
    assert "run_pretrain" in stage_source
    assert "pretrain.train" in stage_source
    assert '"utils.experiment.pretrain_stage:run_pretrain_stage"' in _source("utils/experiment/stages.py")

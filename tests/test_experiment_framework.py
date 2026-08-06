import ast
from dataclasses import fields
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from configs.config_exp import ExpCfg
from configs.config_contact_gen import ContactGenCfg
from configs.config_general import GeneralCfg
from configs.config_model import (
    ConcertoCfg,
    ModelCfg,
    P2VCfg,
    PretrainedEncoderCfg,
    TCECfg,
)
from configs.config_pretrain import CheckpointPolicyCfg, PretrainCfg
from configs.config_rl import RLCfg
from utils.artifacts.manifest import read_manifest, write_manifest
from utils.artifacts.resolver import resolve_artifacts
from utils.config.hash import config_hash
from utils.config.loader import load_exp_cfg
from utils.config.paths import load_project_paths
from utils.experiment.effective_paths import (
    apply_experiment_path_overrides,
    materialize_runtime_paths_yaml,
)
import utils.experiment.runner as experiment_runner
from utils.experiment.runner import plan_experiment, run_experiment
from utils.experiment.validation import (
    ExperimentValidationError,
    validate_contact_schema_version,
    validate_cuda_visible_devices_gpu_count,
    validate_encoder_checkpoint_path_and_declared_dims,
    validate_for_plan,
    validate_for_run,
    validate_full_config,
    validate_generated_gripper_manifest_root,
    validate_isaac_task_and_rsl_rl_entrypoint_strings,
    validate_model_general_num_points_match,
    validate_object_tool_manifests_non_empty,
)


ROOT = Path(__file__).resolve().parents[1]


def _empty_paths_yaml(tmp_path):
    path = tmp_path / "paths.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _contact_paths_yaml(tmp_path):
    objects = tmp_path / "objects.json"
    tools_selected = tmp_path / "tools_selected.json"
    tools_adjusted = tmp_path / "tools_adjusted.json"
    object_dir = tmp_path / "objects"
    usd_dir = tmp_path / "objects_usd"
    mesh_root = tmp_path / "meshdata_adjusted"
    for directory in (object_dir, usd_dir, mesh_root):
        directory.mkdir()
    objects.write_text('["object_a"]\n', encoding="utf-8")
    tools_selected.write_text('["tool_a"]\n', encoding="utf-8")
    tools_adjusted.write_text(
        '[{"name": "tool_a", "head_area": [[0, 0, 0], [1, 1, 1]]}]\n',
        encoding="utf-8",
    )
    path = tmp_path / "paths_contact.yaml"
    path.write_text(
        "\n".join(
            [
                "objects:",
                f"  candidates_json: {objects}",
                f"  usd_dir: {usd_dir}",
                f"  obj_dir: {object_dir}",
                "tools:",
                f"  meshdata_adjusted_root: {mesh_root}",
                f"  tools_adjusted_json: {tools_adjusted}",
                f"  tools_selected_json: {tools_selected}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _field_names(cls):
    return {field.name for field in fields(cls)}


def test_config_skeleton_exposes_planned_fields():
    assert {
        "name",
        "seed",
        "num_points",
        "tools_selected_json",
        "tools_manifest",
        "objects_manifest",
        "randomize_tool_assignment",
        "randomize_object_assignment",
        "deterministic",
        "dtype",
        "artifact_root",
        "wandb",
        "tool_mount",
    }.issubset(_field_names(GeneralCfg))
    assert not {
        "paths_yaml",
        "output_root",
        "contact_schema_version",
        "wandb_enabled",
        "wandb_entity",
        "wandb_mode",
        "wandb_tags",
        "wandb_notes",
        "wandb_metadata_level",
        "wandb_log_code",
    }.intersection(_field_names(GeneralCfg))
    assert GeneralCfg().tool_mount.scale_xyz == [0.1, 0.1, 0.1]

    assert {
        "name",
        "num_pairs",
        "num_object_poses",
        "B",
        "M",
        "chunk_B",
        "epsilon",
        "floor_eps",
        "upright_threshold",
        "schema_version",
        "regenerate",
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
        "name",
        "policy_fusion",
        "pretrained_encoder",
        "tce",
        "p2v",
    }.issubset(_field_names(ModelCfg))
    assert not {
        "num_points",
        "patch_size",
        "encoder_channel",
        "vit_depth",
        "vit_heads",
        "encoder",
        "concerto",
        "action_dim",
        "observation_dim",
        "physics_dim",
        "actor_critic_class",
        "hidden_dims",
    }.intersection(_field_names(ModelCfg))
    assert TCECfg().encoder_type == "TCE"
    assert PretrainedEncoderCfg().adapter == "tce_strict"
    assert P2VCfg().encoder_type == "Point2Vec"
    assert ConcertoCfg().encoder_type == "Concerto"

    assert {
        "enabled_heads",
        "retrain",
        "num_precontact_steps",
        "translation_noise_range",
        "rotation_noise_range_deg",
        "noise_schedule_mode",
        "legal_pose_max_tries",
        "num_query_A",
        "num_query_B",
        "num_query_C",
        "num_query_D",
        "condition_mlp_hidden_dims",
        "cross_attn_layers",
        "cross_attn_heads",
        "decoder_pooling",
        "sdf_head_hidden_dims",
        "denoise_head_hidden_dims",
        "postcontact_head_hidden_dims",
        "validation_noising_seed",
        "fixed_validation_sampling",
        "optimizer",
        "batch",
        "epochs",
        "loss",
        "checkpoint_policy",
    }.issubset(_field_names(PretrainCfg))
    assert not {
        "K",
        "checkpoint_path",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "num_workers",
        "denoise_rot_weight",
        "chamfer_weight",
        "quat_norm_beta",
    }.intersection(_field_names(PretrainCfg))
    assert {
        "save_best",
        "monitor",
        "mode",
        "best_filename",
        "save_optimizer",
        "write_manifest",
        "schema_version",
        "dataset_hash_algo",
        "resume_checkpoint",
    }.issubset(_field_names(CheckpointPolicyCfg))

    assert {
        "name",
        "isaac_task_id",
        "task_id",
        "rsl_rl_cfg_entry_point",
        "ppo",
        "env",
        "table",
        "domain_randomization",
        "reward",
        "encoder_checkpoint",
        "init_checkpoint",
        "freeze_encoder",
    }.issubset(_field_names(RLCfg))
    assert not {
        "num_envs",
        "max_iterations",
        "action_dim",
        "physics_dim",
    }.intersection(_field_names(RLCfg))


def test_legacy_config_aliases_are_properties_not_hash_fields():
    cfg = ExpCfg(name="alias_unit")
    before = config_hash(cfg)

    cfg.general.wandb_enabled = True
    assert cfg.general.wandb.enabled is True
    assert config_hash(cfg) != before

    cfg.contact_gen.stabilize_steps = 31
    assert cfg.contact_gen.physics.t_stabilize == 31

    cfg.pretrain.K = 7
    assert cfg.pretrain.num_precontact_steps == 7
    cfg.pretrain.batch_size = 16
    assert cfg.pretrain.batch.batch_size == 16

    cfg.model.num_points = 128
    assert cfg.model.tce.num_points == 128

    cfg.rl.num_envs = 4
    assert cfg.rl.env.num_envs == 4


def test_full_config_validation_skeleton_has_named_checks_and_catches_bad_config(tmp_path):
    validators = [
        validate_for_plan,
        validate_for_run,
        validate_object_tool_manifests_non_empty,
        validate_model_general_num_points_match,
        validate_encoder_checkpoint_path_and_declared_dims,
        validate_contact_schema_version,
        validate_isaac_task_and_rsl_rl_entrypoint_strings,
        validate_cuda_visible_devices_gpu_count,
    ]
    assert all(callable(validator) for validator in validators)

    cfg = ExpCfg(name="bad_validation_unit")
    cfg.paths_yaml = str(_empty_paths_yaml(tmp_path))
    cfg.model.tce.num_points = 128
    paths = load_project_paths(cfg.paths_yaml)

    try:
        validate_full_config(cfg, paths, strict_paths=False, cuda_visible_devices="0")
    except ExperimentValidationError as exc:
        assert "ModelCfg.encoder.num_points must match GeneralCfg.num_points" in str(exc)
    else:
        raise AssertionError("validate_full_config accepted mismatched num_points")

    cfg = ExpCfg(name="bad_gpu_unit")
    cfg.num_gpus = 2
    assert validate_cuda_visible_devices_gpu_count(cfg, "0")

    cfg = ExpCfg(name="bad_rl_unit")
    cfg.rl.enabled = True
    cfg.rl.isaac_task_id = ""
    assert validate_isaac_task_and_rsl_rl_entrypoint_strings(cfg)


def test_full_config_validation_catches_empty_contact_manifests(tmp_path):
    paths_yaml = _contact_paths_yaml(tmp_path)
    paths = load_project_paths(paths_yaml)
    paths.get("objects.candidates_json").write_text("[]\n", encoding="utf-8")
    cfg = ExpCfg(name="empty_manifest_unit")
    cfg.paths_yaml = str(paths_yaml)
    cfg.contact_gen.enabled = True

    try:
        validate_full_config(cfg, paths, strict_paths=True, cuda_visible_devices=None)
    except ExperimentValidationError as exc:
        assert "objects.candidates_json must contain non-empty JSON" in str(exc)
    else:
        raise AssertionError("validate_full_config accepted an empty object manifest")


def test_generated_gripper_manifest_cannot_escape_configured_root(tmp_path):
    configured_root = tmp_path / "gripper"
    wrong_root = tmp_path / "gripper_new"
    configured_root.mkdir()
    wrong_root.mkdir()
    manifest = configured_root / "generated_grippers.json"
    manifest.write_text(
        "\n".join(
            [
                "{",
                f'  "generated_root": {str(wrong_root)!r},'.replace("'", '"'),
                '  "grippers": [',
                f'    {{"id": "000000", "root_dir": {str(wrong_root / "000000")!r}}}'.replace("'", '"'),
                "  ]",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths_yaml = tmp_path / "paths_generated.yaml"
    paths_yaml.write_text(
        "\n".join(
            [
                "generated_grippers:",
                f"  root: {configured_root}",
                f"  manifest: {manifest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = ExpCfg(name="generated_root_mismatch_unit")
    cfg.rl.enabled = True
    cfg.rl.env.robot_mode = "generated_gripper"

    errors = validate_generated_gripper_manifest_root(
        cfg,
        load_project_paths(paths_yaml),
        strict_paths=True,
    )

    assert any("generated_root must be inside" in error for error in errors)
    assert any("entry 0 root_dir must be inside" in error for error in errors)


def test_general_tools_selected_json_overrides_paths_yaml(tmp_path):
    paths_yaml = _contact_paths_yaml(tmp_path)
    override = tmp_path / "tools_selected_override.json"
    override.write_text('["tool_b"]\n', encoding="utf-8")
    cfg = ExpCfg(name="tool_selection_override_unit")
    cfg.paths_yaml = str(paths_yaml)
    cfg.contact_gen.enabled = True
    cfg.general.tools_selected_json = str(override)

    paths = apply_experiment_path_overrides(cfg, load_project_paths(paths_yaml))

    assert paths.get("tools.tools_selected_json") == override
    validate_full_config(cfg, paths, strict_paths=True, cuda_visible_devices=None)

    runtime_paths = materialize_runtime_paths_yaml(
        cfg,
        paths,
        tmp_path / "paths.runtime.yaml",
    )
    assert runtime_paths.read_text(encoding="utf-8").count(str(override)) >= 1


def test_config_load_and_hash_are_stable(tmp_path):
    cfg = ExpCfg(name="hash_unit")
    same = ExpCfg(name="hash_unit")
    changed = ExpCfg(name="hash_unit")
    changed.general.seed = 7

    assert config_hash(cfg) == config_hash(same)
    assert config_hash(cfg) != config_hash(changed)

    cfg_path = tmp_path / "exp.py"
    cfg_path.write_text(
        "\n".join(
            [
                "from configs.config_exp import ExpCfg",
                "EXP_CFG = ExpCfg(name='loaded_unit')",
                f"EXP_CFG.paths_yaml = {str(_empty_paths_yaml(tmp_path))!r}",
                f"EXP_CFG.general.artifact_root = {str(tmp_path / 'artifacts')!r}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_exp_cfg(cfg_path)

    assert cfg.name == "loaded_unit"


def test_artifact_paths_and_manifest_are_local(tmp_path):
    cfg = ExpCfg(name="artifact unit")
    cfg.general.artifact_root = str(tmp_path / "artifacts")
    cfg.contact_gen.enabled = True
    cfg.general.name = "General Unit"
    cfg.contact_gen.name = "Contact Unit"

    resolved = resolve_artifacts(cfg, timestamp="20260101T000000Z")
    written = write_manifest(
        resolved.experiment.directory,
        {
            "schema_version": "artifact_manifest_v1",
            "status": "planned",
            "config_hash": resolved.experiment.config_hash,
        },
    )

    assert resolved.experiment.directory.is_relative_to(tmp_path / "artifacts")
    assert resolved.stages[0].stage == "contact_gen"
    assert resolved.stages[0].status == "planned"
    assert resolved.stages[0].directory.parts[-4:-1] == ("contact", "General_Unit", "Contact_Unit")
    assert resolved.stages[1].stage == "pretrain"
    assert resolved.stages[1].status == "skipped"
    assert resolved.stages[1].directory.parts[-5:-2] == ("encoder", "General_Unit", "Contact_Unit")
    assert resolved.stages[2].stage == "rl"
    assert resolved.stages[2].status == "skipped"
    assert resolved.stages[2].directory.parts[-6:-1] == (
        "RL",
        "General_Unit",
        "Contact_Unit",
        "TCE",
        "rl_default",
    )
    assert written == resolved.experiment.manifest_path
    assert read_manifest(written)["config_hash"] == resolved.experiment.config_hash


def test_entrypoint_only_config_and_runner_import_is_light(tmp_path):
    tree = ast.parse((ROOT / "run_experiment.py").read_text(encoding="utf-8"))
    flags = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    ]
    assert flags == ["--config", "--mode"]
    assert flags.count("--config") == 1
    assert flags.count("--mode") == 1

    cfg_path = tmp_path / "exp_cli.py"
    paths_yaml = _empty_paths_yaml(tmp_path)
    cfg_path.write_text(
        "\n".join(
            [
                "from configs.config_exp import ExpCfg",
                "EXP_CFG = ExpCfg(name='cli_unit')",
                f"EXP_CFG.paths_yaml = {str(paths_yaml)!r}",
                f"EXP_CFG.general.artifact_root = {str(tmp_path / 'artifacts')!r}",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "run_experiment.py", "--config", str(cfg_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode: run" in result.stdout
    assert "experiment: cli_unit" in result.stdout
    assert "contact_gen: status=skipped" in result.stdout

    code = """
import importlib
import sys
before = set(sys.modules)
for module_name in ("run_experiment", "utils.experiment.runner"):
    importlib.import_module(module_name)
created = set(sys.modules).difference(before)
forbidden = []
for name in created:
    prefixes = ("torch", "isaacsim", "omni", "pxr", "contact_generation", "pretrain", "rsl_rl")
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
        forbidden.append(name)
print("\\n".join(sorted(forbidden)))
raise SystemExit(1 if forbidden else 0)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_plan_returns_ordered_in_memory_artifacts_without_manifests(tmp_path):
    cfg = ExpCfg(name="manifest_unit")
    cfg.paths_yaml = str(_contact_paths_yaml(tmp_path))
    cfg.general.artifact_root = str(tmp_path / "artifacts")
    cfg.general.name = "general"
    cfg.contact_gen.name = "contact"
    cfg.pretrain.name = "pretrain"
    cfg.model.name = "model"
    cfg.contact_gen.enabled = True

    run = plan_experiment(cfg, config_source="unit")
    statuses = [ref.status for ref in (run.artifacts.experiment, *run.artifacts.stages)]
    actions = [ref.action for ref in (run.artifacts.experiment, *run.artifacts.stages)]
    stages = [ref.stage for ref in (run.artifacts.experiment, *run.artifacts.stages)]

    assert stages == ["experiment", "contact_gen", "pretrain", "rl"]
    assert statuses == ["planned", "planned", "skipped", "skipped"]
    assert actions == ["planned", "run-if-needed", "skipped", "skipped"]
    assert run.manifests == ()
    assert not any(ref.manifest_path.exists() for ref in run.artifacts.stages)


def _stage_test_cfg(tmp_path, *, contact_enabled=True, pretrain_enabled=True):
    cfg = ExpCfg(name="stage_dependency_unit")
    cfg.paths_yaml = str(_contact_paths_yaml(tmp_path))
    cfg.general.artifact_root = str(tmp_path / "artifacts")
    cfg.general.name = "general"
    cfg.contact_gen.name = "contact"
    cfg.pretrain.name = "pretrain"
    cfg.model.name = "model"
    cfg.contact_gen.enabled = contact_enabled
    cfg.pretrain.enabled = pretrain_enabled
    cfg.rl.enabled = False
    return cfg


def _fake_stage_loader(calls, *, fail_pretrain=False):
    def load(entrypoint):
        def stage(cfg, paths, artifact_dir, **kwargs):
            if "contact_stage" in entrypoint:
                calls.append("contact_gen")
                return {
                    "stage": "contact_gen",
                    "num_pairs": 1,
                    "num_poses": 1,
                    "ok": 1,
                    "fail": 0,
                    "skipped": 0,
                    "artifact_dir": str(artifact_dir),
                }
            if "pretrain_stage" in entrypoint:
                calls.append("pretrain")
                if fail_pretrain:
                    raise NotImplementedError("SDF supervision is not implemented")
                return {"best_checkpoint_path": str(Path(artifact_dir) / "best.pt")}
            calls.append(entrypoint)
            return {"stage": entrypoint}

        return stage

    return load


def _fake_contact_failure_loader(calls):
    def load(entrypoint):
        def stage(cfg, paths, artifact_dir, **kwargs):
            if "contact_stage" in entrypoint:
                calls.append("contact_gen")
                return {"stage": "contact_gen", "ok": 0, "skipped": 0, "fail": 3}
            calls.append(entrypoint)
            return {"stage": entrypoint}

        return stage

    return load


def _fake_stage_importer(imported, calls):
    def import_module(module_name):
        imported.append(module_name)

        def contact_stage(cfg, paths, artifact_dir, **kwargs):
            calls.append("contact_gen")
            return {
                "stage": "contact_gen",
                "num_pairs": 1,
                "num_poses": 1,
                "ok": 1,
                "fail": 0,
                "skipped": 0,
                "artifact_dir": str(artifact_dir),
            }

        def pretrain_stage(cfg, paths, artifact_dir, **kwargs):
            calls.append("pretrain")
            return {"best_checkpoint_path": str(Path(artifact_dir) / "best.pt")}

        def rl_stage(cfg, paths, artifact_dir, **kwargs):
            calls.append("rl")
            return {"stage": "rl"}

        modules = {
            "utils.experiment.contact_stage": SimpleNamespace(run_contact_stage=contact_stage),
            "utils.experiment.pretrain_stage": SimpleNamespace(run_pretrain_stage=pretrain_stage),
            "utils.experiment.rl_stage": SimpleNamespace(run_rl_stage=rl_stage),
        }
        if module_name not in modules:
            raise AssertionError(f"unexpected import_module call: {module_name}")
        return modules[module_name]

    return import_module


def _top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _write_complete_manifest(ref):
    return write_manifest(
        ref.directory,
        {
            "schema_version": "artifact_manifest_v1",
            "status": "complete",
            "config_hash": ref.config_hash,
            "metrics": {
                "stage": ref.stage,
                "result": {"best_checkpoint_path": str(ref.directory / "best.pt")},
            },
        },
    )


def _write_incomplete_manifest(ref):
    return write_manifest(
        ref.directory,
        {
            "schema_version": "artifact_manifest_v1",
            "status": "running",
            "config_hash": ref.config_hash,
            "metrics": {"stage": ref.stage},
        },
    )


def test_default_run_calls_required_stage_functions(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == ["contact_gen"]
    assert run.stage_results["contact_gen"]["status"] == "complete"
    assert run.stage_results["contact_gen"]["executed"] is True


def test_run_emits_stage_progress_and_contact_summary(tmp_path, monkeypatch, capsys):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run_experiment(cfg, config_source="unit_config.py")

    out = capsys.readouterr().out
    assert "[RUN] experiment=stage_dependency_unit config=unit_config.py mode=run" in out
    assert "[RUN] stage=contact_gen action=run artifact=" in out
    assert "[START] stage=contact_gen entrypoint=utils.experiment.contact_stage:run_contact_stage" in out
    assert "[DONE] stage=contact_gen" in out
    assert "num_pairs=1" in out
    assert "num_poses=1" in out
    assert "ok=1" in out
    assert "fail=0" in out
    assert "skipped=0" in out
    assert "artifact_dir=" in out


def test_contact_only_run_imports_only_contact_stage(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    imported = []
    calls = []
    monkeypatch.setattr(
        experiment_runner.importlib,
        "import_module",
        _fake_stage_importer(imported, calls),
    )

    run = run_experiment(cfg, config_source="unit")

    assert imported == ["utils.experiment.contact_stage"]
    assert calls == ["contact_gen"]
    assert run.stage_results["contact_gen"]["executed"] is True


def test_pretrain_with_dataset_manifest_imports_only_pretrain_stage(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    cfg.pretrain.dataset_manifest = str(tmp_path / "complete_contact_manifest.json")
    imported = []
    calls = []
    monkeypatch.setattr(
        experiment_runner.importlib,
        "import_module",
        _fake_stage_importer(imported, calls),
    )

    run = run_experiment(cfg, config_source="unit")

    assert imported == ["utils.experiment.pretrain_stage"]
    assert calls == ["pretrain"]
    assert run.stage_results["contact_gen"]["status"] == "skipped"
    assert run.stage_results["pretrain"]["executed"] is True


def test_pretrain_without_dataset_manifest_imports_contact_then_pretrain(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    imported = []
    calls = []
    monkeypatch.setattr(
        experiment_runner.importlib,
        "import_module",
        _fake_stage_importer(imported, calls),
    )

    run = run_experiment(cfg, config_source="unit")

    assert imported == [
        "utils.experiment.contact_stage",
        "utils.experiment.pretrain_stage",
    ]
    assert calls == ["contact_gen", "pretrain"]
    assert run.stage_results["contact_gen"]["dependency_reason"] == (
        "pretrain_without_dataset_manifest"
    )


def test_plan_mode_does_not_call_stage_entrypoints(tmp_path, monkeypatch, capsys):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError(f"unexpected stage import/load: {args}")

    monkeypatch.setattr(experiment_runner, "_load_entrypoint", fail_if_called)
    monkeypatch.setattr(experiment_runner.importlib, "import_module", fail_if_called)
    run = plan_experiment(cfg, config_source="unit")

    assert run.stage_results == {}
    assert run.manifests == ()
    assert [ref.status for ref in run.artifacts.stages] == ["planned", "planned", "skipped"]
    assert not any(ref.manifest_path.exists() for ref in run.artifacts.stages)
    assert capsys.readouterr().out == ""


def test_stage_wrappers_keep_runtime_imports_lazy():
    forbidden = {
        "utils/experiment/contact_stage.py": {
            "contact_generation",
            "pretrain",
            "scripts.train",
            "rsl_rl",
            "torch",
            "isaacsim",
            "omni",
        },
        "utils/experiment/pretrain_stage.py": {
            "contact_generation",
            "pretrain",
            "scripts.train",
            "rsl_rl",
            "torch",
            "isaacsim",
            "omni",
        },
        "utils/experiment/rl_stage.py": {
            "contact_generation",
            "pretrain",
            "scripts.train",
            "rsl_rl",
            "torch",
            "isaacsim",
            "omni",
        },
    }
    for relative_path, forbidden_prefixes in forbidden.items():
        imports = _top_level_imports(ROOT / relative_path)
        bad = [
            name
            for name in imports
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
        ]
        assert bad == []


def test_plan_survives_when_strict_readers_would_raise(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=True)
    cfg.model.pretrained_encoder.checkpoint_path = str(tmp_path / "checkpoint.pt")

    def fail_reader(*args, **kwargs):
        raise AssertionError("plan should not read JSON, checkpoints, or manifests")

    monkeypatch.setattr("utils.experiment.validation.read_json", fail_reader)
    monkeypatch.setattr("utils.experiment.validation._read_checkpoint_manifest", fail_reader)
    monkeypatch.setattr(experiment_runner, "manifest_is_complete", fail_reader)
    monkeypatch.setattr(experiment_runner, "read_manifest", fail_reader)

    run = plan_experiment(cfg, config_source="unit")

    assert run.manifests == ()
    assert [ref.action for ref in run.artifacts.stages] == [
        "run-if-needed",
        "run-if-needed",
        "skipped",
    ]


def test_pretrain_without_dataset_manifest_requires_contact_dependency(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")
    contact_ref = next(ref for ref in run.artifacts.stages if ref.stage == "contact_gen")

    assert calls == ["contact_gen", "pretrain"]
    assert contact_ref.requested is False
    assert contact_ref.required is True
    assert contact_ref.dependency_reason == "pretrain_without_dataset_manifest"
    assert run.stage_results["contact_gen"]["status"] == "complete"
    assert run.stage_results["pretrain"]["status"] == "complete"


def test_complete_contact_manifest_reuse_skips_contact_and_runs_pretrain(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    contact_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "contact_gen")
    _write_complete_manifest(contact_ref)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == ["pretrain"]
    assert run.stage_results["contact_gen"]["action"] == "reused"
    assert run.stage_results["contact_gen"]["reused"] is True
    assert run.stage_results["pretrain"]["executed"] is True


def test_contact_regenerate_ignores_complete_manifest(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    cfg.contact_gen.regenerate = True
    contact_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "contact_gen")
    _write_complete_manifest(contact_ref)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == ["contact_gen"]
    assert run.stage_results["contact_gen"]["action"] == "run"
    assert run.stage_results["contact_gen"]["executed"] is True


def test_complete_pretrain_manifest_reuse_skips_pretrain(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    cfg.pretrain.dataset_manifest = str(tmp_path / "contact_manifest.json")
    pretrain_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "pretrain")
    _write_complete_manifest(pretrain_ref)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == []
    assert run.stage_results["pretrain"]["action"] == "reused"
    assert run.stage_results["pretrain"]["reused"] is True


def test_pretrain_retrain_ignores_complete_manifest(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    cfg.pretrain.dataset_manifest = str(tmp_path / "contact_manifest.json")
    cfg.pretrain.retrain = True
    pretrain_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "pretrain")
    _write_complete_manifest(pretrain_ref)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == ["pretrain"]
    assert run.stage_results["pretrain"]["action"] == "run"
    assert run.stage_results["pretrain"]["executed"] is True


def test_incomplete_contact_manifest_runs_contact(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    contact_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "contact_gen")
    _write_incomplete_manifest(contact_ref)
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_stage_loader(calls))

    run = run_experiment(cfg, config_source="unit")

    assert calls == ["contact_gen"]
    assert run.stage_results["contact_gen"]["action"] == "run"
    assert read_manifest(contact_ref.manifest_path)["status"] == "complete"


def test_empty_failed_contact_result_marks_stage_failed(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    refs = {ref.stage: ref for ref in resolve_artifacts(cfg).stages}
    calls = []
    monkeypatch.setattr(experiment_runner, "_load_entrypoint", _fake_contact_failure_loader(calls))

    try:
        run_experiment(cfg, config_source="unit")
    except RuntimeError as exc:
        assert "Contact generation produced no usable outputs" in str(exc)
    else:
        raise AssertionError("empty failed contact result was not rejected")

    assert calls == ["contact_gen"]
    contact_manifest = read_manifest(refs["contact_gen"].manifest_path)
    assert contact_manifest["status"] == "failed"
    assert "no usable outputs" in contact_manifest["metrics"]["error"]


def test_contact_run_pair_logs_and_rethrows_keyboard_interrupt(tmp_path, capsys):
    from contact_generation.batch_generate import run_pair

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_main(cfg):
        return None

    contact_cfg = ContactGenCfg()
    success = run_pair(
        "tool.obj",
        "object.obj",
        "tool_a",
        "object_a",
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        contact_cfg,
        pose_idx=0,
        num_poses=1,
        seed=1,
        physics_options={},
        generator_api=(FakeConfig, fake_main),
    )
    out = capsys.readouterr().out
    assert success is True
    assert "[PAIR-START] tool=tool_a object=object_a" in out
    assert "[IMPORT] loading contact generator phase" in out
    assert "[CALL] running contact generator phase" in out
    assert "[PAIR-DONE] phase=postcontact tool=tool_a object=object_a" in out

    def interrupting_main(cfg):
        raise KeyboardInterrupt()

    try:
        run_pair(
            "tool.obj",
            "object.obj",
            "tool_a",
            "object_a",
            tmp_path,
            tmp_path / "tools_adjusted.json",
            (0.1, 0.1, 0.1),
            0,
            contact_cfg,
            pose_idx=0,
            num_poses=1,
            seed=1,
            physics_options={},
            generator_api=(FakeConfig, interrupting_main),
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("run_pair swallowed KeyboardInterrupt")
    out = capsys.readouterr().out
    assert "[INTERRUPT] pair interrupted tool=tool_a object=object_a" in out


def test_contact_worker_logs_skip_without_heavy_generator(tmp_path, capsys):
    from contact_generation.batch_generate import output_path, worker

    contact_cfg = ContactGenCfg()
    tool_name = "tool_a"
    object_name = "object_a"
    existing = output_path(tmp_path, tool_name, object_name, 0, 1)
    existing.parent.mkdir(parents=True)
    existing.write_text("existing\n", encoding="utf-8")

    result = worker(
        [("tool.obj", "object.obj", tool_name, object_name, None)],
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        contact_cfg,
        True,
        1,
        physics_options={},
        seed=1,
    )

    out = capsys.readouterr().out
    assert result == (0, 0, 1)
    assert "[WORKER-START] gpu=0 pairs=1 poses=1 jobs=1" in out
    assert "[PAIR-SKIP] gpu=0 tool=tool_a object=object_a" in out
    assert "[WORKER-DONE] gpu=0 ok=0 fail=0 skipped=1" in out


def test_contact_worker_reuses_one_physics_runner(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    class FakeRunner:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            calls.append(("close", id(self)))

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_main(cfg, *, physics_runner=None):
        calls.append(("run", id(physics_runner)))

    calls = []
    runner = FakeRunner()
    loaded = []
    monkeypatch.setattr(
        batch_generate,
        "_load_physics_runner",
        lambda name: loaded.append(name) or runner,
    )
    monkeypatch.setattr(batch_generate, "_load_generator_api", lambda phase="postcontact": (FakeConfig, fake_main))

    contact_cfg = ContactGenCfg()
    result = batch_generate.worker(
        [
            ("tool.obj", "object_a.obj", "tool_a", "object_a", None),
            ("tool.obj", "object_b.obj", "tool_a", "object_b", None),
        ],
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        contact_cfg,
        False,
        2,
        physics_options={},
        seed=1,
    )

    run_calls = [call for call in calls if call[0] == "run"]
    assert result == (4, 0, 0)
    assert loaded == [contact_cfg.physics.runner, contact_cfg.physics.runner]
    assert run_calls[:4] == [("run", id(None))] * 4
    assert run_calls[4:] == [("run", id(runner))] * 8
    assert calls[-1] == ("close", id(runner))
    assert runner.closed is True


def test_contact_generation_default_caps_to_one_worker(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    cfg.contact_gen.num_pairs = 3
    cfg.contact_gen.num_object_poses = 1
    objects = ["object_a", "object_b", "object_c"]
    tools = ["tool_a"]
    pairs = [
        ("tool.obj", "object_a.obj", "tool_a", "object_a", None),
        ("tool.obj", "object_b.obj", "tool_a", "object_b", None),
        ("tool.obj", "object_c.obj", "tool_a", "object_c", None),
    ]
    calls = []

    monkeypatch.setattr(batch_generate, "read_json", lambda path: objects)
    monkeypatch.setattr(batch_generate, "load_selected_tool_ids", lambda path: tools)
    monkeypatch.setattr(batch_generate, "build_pairs", lambda *args, **kwargs: pairs)
    monkeypatch.setattr(batch_generate, "visible_cuda_device_indices", lambda **kwargs: [0, 1, 2, 3])
    def fake_pool(worker_args, *, worker_fn):
        calls.append((worker_args, worker_fn))
        return (len(worker_args[0][1]), 0, 0)

    monkeypatch.setattr(batch_generate, "_run_worker_pool", fake_pool)

    result = batch_generate.run_contact_generation(
        cfg,
        load_project_paths(cfg.paths_yaml),
        tmp_path / "artifacts" / "contact",
    )

    assert result.ok == 3
    assert len(calls) == 3
    assert calls[0][0][0][0] == "geometry"
    assert calls[1][0][0][0] == "stabilize"
    assert calls[2][0][0][0] == "postcontact"
    assert calls[0][0][0][5] == 0
    assert len(calls[0][0][0][1]) == 3


def test_contact_generation_skips_worker_pools_when_final_outputs_exist(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    cfg.num_gpus = 4
    cfg.contact_gen.num_pairs = 2
    cfg.contact_gen.num_object_poses = 2
    objects = ["object_a", "object_b"]
    tools = ["tool_a"]
    pairs = [
        ("tool.obj", "object_a.obj", "tool_a", "object_a", None),
        ("tool.obj", "object_b.obj", "tool_a", "object_b", None),
    ]
    out_dir = tmp_path / "artifacts" / "contact"
    for _tool_path, _obj_path, tool_name, object_name, _asset in pairs:
        for pose_idx in range(cfg.contact_gen.num_object_poses):
            output = batch_generate.output_path(
                out_dir,
                tool_name,
                object_name,
                pose_idx,
                cfg.contact_gen.num_object_poses,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("existing\n", encoding="utf-8")

    monkeypatch.setattr(batch_generate, "read_json", lambda path: objects)
    monkeypatch.setattr(batch_generate, "load_selected_tool_ids", lambda path: tools)
    monkeypatch.setattr(batch_generate, "build_pairs", lambda *args, **kwargs: pairs)
    captured_workers = []
    original_count = batch_generate.count_existing_final_outputs
    def count_wrapper(*args, max_workers=1, **kwargs):
        captured_workers.append(max_workers)
        return original_count(*args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(batch_generate, "count_existing_final_outputs", count_wrapper)
    monkeypatch.setattr(
        batch_generate,
        "_run_worker_pool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker pool should not start when every final output exists")
        ),
    )

    result = batch_generate.run_contact_generation(
        cfg,
        load_project_paths(cfg.paths_yaml),
        out_dir,
    )

    assert result.ok == 0
    assert result.fail == 0
    assert result.skipped == 4
    assert result.num_pairs == 2
    assert result.num_poses == 2
    assert captured_workers == [4]


def test_existing_output_check_workers_env_override(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    cfg = _stage_test_cfg(tmp_path, contact_enabled=True, pretrain_enabled=False)
    cfg.num_gpus = 8
    monkeypatch.setenv("CONTACT_EXISTING_OUTPUT_CHECK_WORKERS", "3")

    assert batch_generate.existing_output_check_workers(cfg) == 3


def test_worker_pool_terminates_after_stall_with_partial_results(monkeypatch):
    from contact_generation import batch_generate

    class FakeAsyncResult:
        def __init__(self, ready, value):
            self._ready = ready
            self._value = value

        def ready(self):
            return self._ready

        def get(self):
            return self._value

    class FakePool:
        def __init__(self, _size):
            self.results = [
                FakeAsyncResult(True, (2, 1, 3)),
                FakeAsyncResult(False, (0, 0, 0)),
            ]
            self.index = 0
            self._pool = []
            pools.append(self)

        def apply_async(self, _worker_fn, _args):
            result = self.results[self.index]
            self.index += 1
            return result

    pools = []
    hard_terminations = []
    worker_args = [
        ("postcontact", [("tool.obj", "object.obj", "tool_a", "object_a", None)], None, None, None, 0, None, True, 1),
        ("postcontact", [("tool.obj", "object.obj", "tool_b", "object_b", None)] * 4, None, None, None, 1, None, True, 2),
    ]
    times = iter([0.0, 0.0, 31.0])
    monkeypatch.setenv("CONTACT_WORKER_STALL_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(batch_generate.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        batch_generate.mp,
        "get_context",
        lambda _name: SimpleNamespace(Pool=FakePool),
    )
    monkeypatch.setattr(
        batch_generate,
        "_hard_terminate_worker_pool",
        lambda _pool, *, reason: hard_terminations.append(reason),
    )

    result = batch_generate._run_worker_pool(worker_args, worker_fn=lambda *args: None)

    assert result == (2, 9, 3)
    assert pools
    assert hard_terminations == ["stalled"]


def test_phase_worker_does_not_start_physics_runner_for_all_final_skips(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    contact_cfg = ContactGenCfg()
    output = batch_generate.output_path(tmp_path, "tool_a", "object_a", 0, 1)
    output.parent.mkdir(parents=True)
    output.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        batch_generate,
        "_load_physics_runner",
        lambda name: (_ for _ in ()).throw(
            AssertionError("physics runner should not start for all-skip stabilize phase")
        ),
    )

    result = batch_generate.phase_worker(
        "stabilize",
        [("tool.obj", "object.obj", "tool_a", "object_a", None)],
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        contact_cfg,
        True,
        1,
        physics_options={},
        seed=1,
    )

    assert result == (0, 0, 1)


def test_geometry_resume_regenerates_candidate_missing_manifest(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    output = batch_generate.output_path(tmp_path, "tool_a", "object_a", 0, 1)
    candidate = batch_generate.candidate_artifact_path(output)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("partial candidate\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        batch_generate,
        "run_pair",
        lambda *args, **kwargs: calls.append(kwargs["phase"]) or True,
    )

    result = batch_generate.phase_worker(
        "geometry",
        [("tool.obj", "object.obj", "tool_a", "object_a", None)],
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        ContactGenCfg(),
        True,
        1,
        physics_options={},
        seed=1,
    )

    assert result == (1, 0, 0)
    assert calls == ["geometry"]


def test_geometry_existing_count_requires_candidate_and_manifest(tmp_path):
    from contact_generation import batch_generate

    pairs = [("tool.obj", "object.obj", "tool_a", "object_a", None)]
    output = batch_generate.output_path(tmp_path, "tool_a", "object_a", 0, 1)
    candidate = batch_generate.candidate_artifact_path(output)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("partial candidate\n", encoding="utf-8")

    assert batch_generate.count_existing_final_outputs(
        pairs, tmp_path, 1, geometry_only=True
    ) == 0

    batch_generate.candidate_manifest_path(output).write_text(
        "{}\n", encoding="utf-8"
    )
    assert batch_generate.count_existing_final_outputs(
        pairs, tmp_path, 1, geometry_only=True
    ) == 1


def test_phase_worker_does_not_block_on_physics_close_by_default(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    class FakeRunner:
        def close(self):
            raise AssertionError("phase worker should leave runner for pool termination")

    contact_cfg = ContactGenCfg()
    pt = batch_generate.output_path(tmp_path, "tool_a", "object_a", 0, 1)
    candidate = batch_generate.candidate_artifact_path(pt)
    candidate.parent.mkdir(parents=True)
    candidate.write_text("candidate\n", encoding="utf-8")

    monkeypatch.setattr(batch_generate, "_load_physics_runner", lambda name: FakeRunner())
    monkeypatch.setattr(batch_generate, "run_pair", lambda *args, **kwargs: True)

    result = batch_generate.phase_worker(
        "stabilize",
        [("tool.obj", "object.obj", "tool_a", "object_a", None)],
        tmp_path,
        tmp_path / "tools_adjusted.json",
        (0.1, 0.1, 0.1),
        0,
        contact_cfg,
        True,
        1,
        physics_options={},
        seed=1,
    )

    assert result == (1, 0, 0)


def test_contact_worker_closes_runner_on_keyboard_interrupt(tmp_path, monkeypatch):
    from contact_generation import batch_generate

    class FakeRunner:
        def close(self):
            calls.append("close")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def interrupting_main(cfg, *, physics_runner=None):
        calls.append(("run", physics_runner))
        if physics_runner is not None:
            raise KeyboardInterrupt()

    calls = []
    runner = FakeRunner()
    monkeypatch.setattr(batch_generate, "_load_physics_runner", lambda name: runner)
    monkeypatch.setattr(
        batch_generate,
        "_load_generator_api",
        lambda phase="postcontact": (FakeConfig, interrupting_main),
    )

    try:
        batch_generate.worker(
            [("tool.obj", "object.obj", "tool_a", "object_a", None)],
            tmp_path,
            tmp_path / "tools_adjusted.json",
            (0.1, 0.1, 0.1),
            0,
            ContactGenCfg(),
            False,
            1,
            physics_options={},
            seed=1,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("worker swallowed KeyboardInterrupt")

    assert calls[0] == ("run", None)
    assert calls[1] == ("run", runner)
    assert calls[-1] == "close"


def test_failed_sdf_pretrain_happens_after_contact_and_marks_failed(tmp_path, monkeypatch):
    cfg = _stage_test_cfg(tmp_path, contact_enabled=False, pretrain_enabled=True)
    cfg.pretrain.enabled_heads = ["sdf"]
    calls = []
    monkeypatch.setattr(
        experiment_runner,
        "_load_entrypoint",
        _fake_stage_loader(calls, fail_pretrain=True),
    )
    refs = {ref.stage: ref for ref in resolve_artifacts(cfg).stages}

    try:
        run_experiment(cfg, config_source="unit")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("SDF pretrain failure was not propagated")

    assert calls == ["contact_gen", "pretrain"]
    assert read_manifest(refs["contact_gen"].manifest_path)["status"] == "complete"
    pretrain_manifest = read_manifest(refs["pretrain"].manifest_path)
    assert pretrain_manifest["status"] == "failed"
    assert "NotImplementedError" in pretrain_manifest["metrics"]["error"]

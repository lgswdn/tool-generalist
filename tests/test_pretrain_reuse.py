from __future__ import annotations

from copy import deepcopy

from configs.config_exp import ExpCfg
from utils.artifacts.manifest import ArtifactManifest, write_manifest
from utils.artifacts.naming import _pretrain_model_hash_payload, encoder_artifact_name
from utils.config.hash import config_hash
from utils.artifacts.resolver import resolve_artifacts
from utils.experiment.runner import _stage_action
from utils.experiment.stages import all_stages


def test_pretrain_reuse_skips_local_contact_and_pretrain_stages():
    cfg = ExpCfg(name="reuse_unit")
    cfg.contact_gen.enabled = True
    cfg.pretrain.enabled = True
    cfg.rl.enabled = True
    cfg.pretrain_reuse = "configs/experiments/multitools_new.py"

    stages = {stage.name: stage for stage in all_stages(cfg)}

    assert stages["contact_gen"].required is False
    assert stages["pretrain"].required is False
    assert stages["rl"].required is True
    cfg.validate()


def test_pretrain_reuse_is_not_part_of_artifact_hash():
    cfg_a = ExpCfg(name="reuse_hash_unit")
    cfg_a.rl.enabled = True
    cfg_a.pretrain_reuse = "configs/experiments/multitools_new.py"

    cfg_b = deepcopy(cfg_a)
    cfg_b.pretrain_reuse = "configs/experiments/other_multitool.py"

    artifacts_a = resolve_artifacts(cfg_a, timestamp="20260101T000000Z")
    artifacts_b = resolve_artifacts(cfg_b, timestamp="20260101T000000Z")

    assert artifacts_a.experiment.config_hash == artifacts_b.experiment.config_hash
    assert artifacts_a.experiment.directory == artifacts_b.experiment.directory
    assert [(s.stage, s.config_hash, s.directory) for s in artifacts_a.stages] == [
        (s.stage, s.config_hash, s.directory) for s in artifacts_b.stages
    ]
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_inactive_icp_backend_does_not_change_tce_pretrain_hash():
    cfg = ExpCfg(name="inactive_backend_hash_unit")
    cfg.model.encoder_backend = "tce"

    payload = _pretrain_model_hash_payload(cfg)

    assert "icp" not in payload
    assert "tce" in payload
    assert "p2v" in payload

    cfg.model.encoder_backend = "icp"
    assert "icp" in _pretrain_model_hash_payload(cfg)


def test_inactive_unicorn_defaults_do_not_change_non_unicorn_hashes():
    cfg = ExpCfg(name="inactive_unicorn_hash_unit")
    cfg.pretrain.enabled = True
    cfg.model.encoder_backend = "tce"

    baseline_config_hash = config_hash(cfg)
    baseline_encoder_name = encoder_artifact_name(cfg)

    changed = deepcopy(cfg)
    changed.model.unicorn.num_patches += 1
    changed.pretrain.unicorn.num_patches += 1
    changed.pretrain.optimizer.sam_rho = 0.2
    changed.pretrain.tasks.contact = True

    assert config_hash(changed) == baseline_config_hash
    assert encoder_artifact_name(changed) == baseline_encoder_name


def test_unicorn_hashes_include_unicorn_settings():
    cfg = ExpCfg(name="active_unicorn_hash_unit")
    cfg.pretrain.enabled = True
    cfg.pretrain.mode = "unicorn_contact"
    cfg.pretrain.enabled_heads = ["contact"]
    cfg.model.encoder_backend = "unicorn"
    cfg.model.pretrained_encoder.adapter = "unicorn_strict"

    baseline_config_hash = config_hash(cfg)
    baseline_encoder_name = encoder_artifact_name(cfg)

    changed = deepcopy(cfg)
    changed.model.unicorn.num_patches += 1
    changed.pretrain.unicorn.num_patches += 1

    assert config_hash(changed) != baseline_config_hash
    assert encoder_artifact_name(changed) != baseline_encoder_name


def test_contact_stage_reuses_existing_outputs_when_top_manifest_is_incomplete(tmp_path):
    cfg = ExpCfg(name="contact_reuse_unit")
    cfg.general.artifact_root = str(tmp_path)
    cfg.contact_gen.enabled = True
    cfg.pretrain.enabled = False
    cfg.rl.enabled = False

    contact_ref = next(ref for ref in resolve_artifacts(cfg).stages if ref.stage == "contact_gen")
    output_dir = contact_ref.directory / "tool"
    output_dir.mkdir(parents=True)
    (output_dir / "sample.pt.manifest.json").write_text("{}", encoding="utf-8")
    write_manifest(
        contact_ref.directory,
        ArtifactManifest(
            artifact_type="contact",
            artifact_name=contact_ref.artifact_name,
            exp_cfg_name=cfg.name,
            config_hash=contact_ref.config_hash,
            status="running",
        ),
    )

    assert _stage_action(cfg, contact_ref) == "reused"

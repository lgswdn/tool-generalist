from __future__ import annotations

from copy import deepcopy

from configs.config_exp import ExpCfg
from utils.artifacts.manifest import ArtifactManifest, write_manifest
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

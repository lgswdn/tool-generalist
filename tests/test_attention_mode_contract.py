from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_tce_multitask_training_propagates_attention_mode_explicitly():
    source = _source("pretrain/train.py")

    assert "vit_attention_mode=cfg.vit_attention_mode" in source
    assert 'vit_attention_contract": _VIT_ATTENTION_CONTRACT' in source
    assert "expected_vit_attention_mode=cfg.vit_attention_mode" in source


def test_attention_mode_has_no_runtime_or_model_fallback():
    model = _source("pretrain/model.py")
    actor = _source("rsl_rl/modules/actor_critic_tg.py")
    actor_bimanual = _source("rsl_rl/modules/actor_critic_tg_bimanual.py")
    evaluation = _source("scripts/eval_objects.py")

    assert "vit_attention_mode: str | None = None" in model
    assert "vit_attention_mode: str | None = None" in actor
    assert "vit_attention_mode: str | None = None" in actor_bimanual
    assert 'policy.setdefault("vit_attention_mode"' not in evaluation
    assert "Runtime spec is missing required policy_params.vit_attention_mode" in evaluation


def test_rl_rejects_ambiguous_or_mismatched_attention_checkpoint():
    actor = _source("rsl_rl/modules/actor_critic_tg.py")

    assert "TCE checkpoint predates explicit attention propagation" in actor
    assert "TCE checkpoint attention mismatch" in actor
    assert "expected_vit_attention_mode=vit_attention_mode" in actor
    assert 'expected_vit_attention_mode == "joint_self"' in actor

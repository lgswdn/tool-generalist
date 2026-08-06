def test_unicorn_ours_contact_hidden_experiment_uses_tce_checkpoint_and_head():
    from configs.experiments.panda_general_unicorn_ours_contact_hidden_full_yes_5k import (
        EXP_CFG,
        UNICORN_OURS_PRETRAIN_CHECKPOINT,
    )

    EXP_CFG.validate()
    assert EXP_CFG.model.encoder_backend == "tce"
    assert EXP_CFG.model.tce.rl_token_source == "contact_head_hidden"
    assert EXP_CFG.model.pretrained_encoder.checkpoint_path == UNICORN_OURS_PRETRAIN_CHECKPOINT
    assert "unicorn_pretrain_ours_generated_gripper" in UNICORN_OURS_PRETRAIN_CHECKPOINT
    assert EXP_CFG.pretrain.enabled is False
    assert EXP_CFG.rl.freeze_encoder is True
    assert EXP_CFG.rl.ppo.max_iterations == 5000
    assert EXP_CFG.rl.action.scale == 0.06

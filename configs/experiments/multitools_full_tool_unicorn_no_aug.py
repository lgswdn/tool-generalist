"""Single-arm full-tool/full-object RL using the no-augmentation UniCORN encoder."""

from copy import deepcopy

from configs.experiments.multitools_full_tool_diff_post import EXP_CFG as _BASE_EXP_CFG


UNICORN_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_no_aug/contact_gen_full_tool/"
    "unicorn_contact_no_aug_unicorn_contact/"
    "4191daadb22dce0efb6a6112463fdde0f8ae30393eced4dc188df2753f796740/"
    "best.pt"
)


EXP_CFG = deepcopy(_BASE_EXP_CFG)

EXP_CFG.name = "multitools_full_tool_unicorn_no_aug"
EXP_CFG.general.name = "multitools_full_tool_unicorn_no_aug"
EXP_CFG.model.name = "unicorn_contact_no_aug"

EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None

EXP_CFG.model.encoder_backend = "unicorn"
EXP_CFG.model.pretrained_encoder.name = "unicorn"
EXP_CFG.model.pretrained_encoder.adapter = "unicorn_strict"
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_ENCODER_CHECKPOINT

EXP_CFG.rl.enabled = True
EXP_CFG.rl.name = "multitools_full_tool_unicorn_no_aug"
EXP_CFG.rl.actor_critic_class = "ActorCriticTGUnicorn"
EXP_CFG.rl.launch.run_name = "multitools_full_tool_unicorn_no_aug"

EXP_CFG.model.policy_fusion.reuse_pretrain_pose_cross_attn = False

"""UniCORN-ours TCE RL using its pretrained contact-head hidden patch tokens."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


UNICORN_OURS_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_contact_hidden_full_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_PRETRAIN_CHECKPOINT
EXP_CFG.model.tce.rl_token_source = "contact_head_hidden"
configure_full_yes_comparison(EXP_CFG)

"""Cross-embodiment UniCORN-ours RL on DGN objects for 10,000 iterations."""

from configs.panda_comparison_common import configure_dgn_10k_comparison
from configs.panda_experiment_common import cross_embodiment_gripper_unicorn_rl_cfg


UNICORN_OURS_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)


EXP_CFG = cross_embodiment_gripper_unicorn_rl_cfg(
    "ce_unicorn_ours_dgn_10k",
    ours_tce=True,
)
EXP_CFG.num_gpus = 8
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_PRETRAIN_CHECKPOINT
configure_dgn_10k_comparison(EXP_CFG)

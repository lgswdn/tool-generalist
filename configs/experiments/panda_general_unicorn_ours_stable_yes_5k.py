"""UniCORN-ours RL on stable goals and the small unique YES set."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


SMALL_YES_MANIFEST = (
    "../object_selections/panda_general_small_yes_unique.json"
)
UNICORN_OURS_PRETRAIN_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_stable_yes_5k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_PRETRAIN_CHECKPOINT

# Match the PointNet stable-YES comparison contract.
configure_full_yes_comparison(EXP_CFG)
EXP_CFG.general.rl_objects_manifest = SMALL_YES_MANIFEST
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 1.0
EXP_CFG.rl.reward.stable_success_dwell_steps = 1

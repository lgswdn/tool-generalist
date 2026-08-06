"""GG fine-tuning from the prior 15k policy with real Franka finger speed."""

from configs.panda_comparison_common import configure_gg_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


PARENT_POLICY_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_unicorn_ours_gg_resume_to_15k/no-contact/TCE/"
    "panda_general_unicorn_ours_gg_resume_to_15k/20260718T013822Z/"
    "model_best.pt"
)
ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)


EXP_CFG = generated_gripper_unicorn_rl_cfg(
    "panda_general_unicorn_ours_gg_real_speed",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = ENCODER_CHECKPOINT
configure_gg_comparison(EXP_CFG)

# Initialize the full actor-critic policy but deliberately start a fresh PPO
# optimizer/schedule because the parallel-gripper dynamics changed.
EXP_CFG.rl.init_checkpoint = PARENT_POLICY_CHECKPOINT
EXP_CFG.rl.resume_checkpoint = None

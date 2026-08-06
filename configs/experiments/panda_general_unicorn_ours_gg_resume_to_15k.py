"""Resume the completed Unicorn-ours GG run from iteration 4,962 to 15,000."""

from configs.panda_comparison_common import GG_MAX_ITERATIONS, configure_gg_comparison
from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


RESUME_ITERATION = 4962
RESUME_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_unicorn_ours_gg_from_full_yes_5k/no-contact/TCE/"
    "panda_general_unicorn_ours_gg_from_full_yes_5k/20260717T133717Z/"
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
    "panda_general_unicorn_ours_gg_resume_to_15k",
    ours_tce=True,
)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = ENCODER_CHECKPOINT
configure_gg_comparison(EXP_CFG)
EXP_CFG.rl.init_checkpoint = None
EXP_CFG.rl.resume_checkpoint = RESUME_CHECKPOINT
# OnPolicyRunner interprets max_iterations as additional iterations on resume.
EXP_CFG.rl.ppo.max_iterations = GG_MAX_ITERATIONS - RESUME_ITERATION

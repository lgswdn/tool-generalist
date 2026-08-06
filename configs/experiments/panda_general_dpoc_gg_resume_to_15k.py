"""Resume the completed DPOC GG run from iteration 4,931 to 15,000."""

from configs.panda_comparison_common import GG_MAX_ITERATIONS, configure_gg_comparison
from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


RESUME_ITERATION = 4931
RESUME_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_dpoc_gg_from_full_yes_5k/no-contact/TCE/"
    "panda_general_dpoc_gg_from_full_yes_5k/20260717T133448Z/"
    "model_best.pt"
)
ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "002002d13240f5618b67ce62952e1e18c95cba23269c2ee2f628e8e381ff74d9/"
    "best.pt"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg(
    "panda_general_dpoc_gg_resume_to_15k"
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

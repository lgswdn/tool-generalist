"""Generated-gripper DPOC curriculum RL resumed from the stable-only checkpoint."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


STABLE_RESUME_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_diff_post_oc_stable/contact_gen_generated_gripper/TCE/"
    "panda_general_diff_post_oc_stable/20260706T172715Z/model_6000.pt"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_curr_from_stable")
EXP_CFG.rl.curriculum.enabled = True
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 10000
EXP_CFG.rl.curriculum.start_stable_pose_probability = 1.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.3
EXP_CFG.rl.resume_checkpoint = STABLE_RESUME_CHECKPOINT

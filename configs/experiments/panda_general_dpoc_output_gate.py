"""Generated-gripper DPOC 50/50 targets with output-gated stable/curr experts."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


STABLE_EXPERT_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_diff_post_oc_stable/contact_gen_generated_gripper/TCE/"
    "panda_general_diff_post_oc_stable/20260706T172715Z/model_6000.pt"
)
CURR_EXPERT_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_dpoc_curr/contact_gen_generated_gripper/TCE/"
    "panda_general_dpoc_curr/20260708T050019Z/model_best.pt"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_output_gate")
EXP_CFG.rl.actor_critic_class = "ActorCriticTGOutputGate"
EXP_CFG.rl.output_gate_expert_a_checkpoint = STABLE_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_expert_b_checkpoint = CURR_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_freeze_experts = True
EXP_CFG.rl.output_gate_hidden_dims = (256, 64)
EXP_CFG.rl.output_gate_initial_expert_a_weight = 0.8
EXP_CFG.rl.output_gate_per_action = False
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.5
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.5

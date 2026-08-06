"""Generated-gripper GG targets with output-gated stable/curr experts."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "070c98e77b135e637bdeb857f81886e7d1473df2e9438c782dcce4a79eedd779/"
    "best.pt"
)
STABLE_EXPERT_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_diff_post_oc_stable/contact_gen_generated_gripper/TCE/"
    "panda_general_diff_post_oc_stable/20260706T172715Z/model_6000.pt"
)
CURR_EXPERT_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_dpoc_curr/contact_gen_generated_gripper/TCE/"
    "panda_general_dpoc_curr/20260708T050019Z/model_best.pt"
    #"/mnt/project/world_model/tool_generalist/artifacts/RL/"
    #"panda_general_diff_post_oc/contact_gen_generated_gripper/TCE/"
    #"panda_general_diff_post_oc/20260706T172627Z/model_7500.pt"
)
GRASPGEN_FAILURE_OBJECTS_MANIFEST = (
    "../object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_conf_gt_0p9_listed_scales.json"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_output_gate_gg")
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = DPOC_ENCODER_CHECKPOINT
EXP_CFG.general.rl_objects_manifest = GRASPGEN_FAILURE_OBJECTS_MANIFEST
EXP_CFG.rl.actor_critic_class = "ActorCriticTGOutputGate"
EXP_CFG.rl.output_gate_expert_a_checkpoint = STABLE_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_expert_b_checkpoint = CURR_EXPERT_CHECKPOINT
EXP_CFG.rl.output_gate_freeze_experts = True
EXP_CFG.rl.output_gate_hidden_dims = (256, 64)
EXP_CFG.rl.output_gate_initial_expert_a_weight = 0.8
EXP_CFG.rl.output_gate_per_action = False
EXP_CFG.rl.init_checkpoint = None
EXP_CFG.rl.resume_checkpoint = None
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.5
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.5

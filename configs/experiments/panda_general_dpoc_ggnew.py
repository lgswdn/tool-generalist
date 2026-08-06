"""Generated-gripper GG RL using the newer generated-gripper DPOC encoder."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "002002d13240f5618b67ce62952e1e18c95cba23269c2ee2f628e8e381ff74d9/"
    "best.pt"
)
GRASPGEN_FAILURE_OBJECTS_MANIFEST = (
    "../object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_conf_gt_0p9_listed_scales.json"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_ggnew")
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = DPOC_ENCODER_CHECKPOINT
EXP_CFG.general.rl_objects_manifest = GRASPGEN_FAILURE_OBJECTS_MANIFEST
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.0

"""Generated-gripper GG RL using the TCE contact-pretrained encoder."""

from configs.panda_experiment_common import generated_gripper_unicorn_rl_cfg


UNICORN_OURS_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)
GRASPGEN_FAILURE_OBJECTS_MANIFEST = (
    "../object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_conf_gt_0p9_listed_scales.json"
)


EXP_CFG = generated_gripper_unicorn_rl_cfg("panda_general_unicorn_gg", ours_tce=True)
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = UNICORN_OURS_ENCODER_CHECKPOINT
EXP_CFG.general.rl_objects_manifest = GRASPGEN_FAILURE_OBJECTS_MANIFEST
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.0

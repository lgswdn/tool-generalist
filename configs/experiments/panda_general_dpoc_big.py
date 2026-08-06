"""Generated-gripper object-centered diff-post RL with large unstable-pose objects."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "070c98e77b135e637bdeb857f81886e7d1473df2e9438c782dcce4a79eedd779/"
    "best.pt"
)
UNGRASPABLE_OBJECTS_MANIFEST = (
    "/mnt/project/world_model/tool_generalist/grasp_result_dgn_full_yes/"
    "conclusions/ungraspable_pairs.json"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dpoc_big")
EXP_CFG.contact_gen.enabled = False
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.pretrain_reuse = None
EXP_CFG.model.pretrained_encoder.checkpoint_path = DPOC_ENCODER_CHECKPOINT
EXP_CFG.general.rl_objects_manifest = UNGRASPABLE_OBJECTS_MANIFEST
EXP_CFG.rl.domain_randomization.object.scale.enabled = False
EXP_CFG.rl.curriculum.enabled = False
EXP_CFG.rl.curriculum.start_step = 0
EXP_CFG.rl.curriculum.end_step = 0
EXP_CFG.rl.curriculum.start_stable_pose_probability = 0.0
EXP_CFG.rl.curriculum.end_stable_pose_probability = 0.0

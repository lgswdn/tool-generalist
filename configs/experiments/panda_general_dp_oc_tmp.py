"""Temporary generated-gripper OC diff-post RL on low-scale failed objects."""

from configs.panda_experiment_common import generated_gripper_diff_post_rl_cfg


FAILED_LT_016_OBJECTS = "configs/object_selections/panda_general_dp_oc_tmp_failed_lt_0.16.json"
RESUME_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_diff_post_oc/no-contact/TCE/panda_general_diff_post_oc/"
    "20260701T070048Z/model_32000.pt"
)


EXP_CFG = generated_gripper_diff_post_rl_cfg("panda_general_dp_oc_tmp")
EXP_CFG.general.rl_objects_manifest = FAILED_LT_016_OBJECTS
EXP_CFG.rl.resume_checkpoint = RESUME_CHECKPOINT

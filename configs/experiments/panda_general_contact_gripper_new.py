"""Contact-only generation over full_yes using the gripper_new asset set."""

from configs.panda_experiment_common import generated_gripper_diff_post_pretrain_cfg


EXP_CFG = generated_gripper_diff_post_pretrain_cfg()
EXP_CFG.name = "panda_general_contact_gripper_new"
EXP_CFG.general.name = EXP_CFG.name
EXP_CFG.paths_yaml = "configs/paths/generated_gripper_contact_new.yaml"
EXP_CFG.contact_gen.name = "contact_gen_gripper_new"
EXP_CFG.contact_gen.regenerate = True
EXP_CFG.pretrain.enabled = False
EXP_CFG.pretrain.retrain = False
EXP_CFG.rl.enabled = False

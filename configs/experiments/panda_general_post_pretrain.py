"""Object-centered postcontact-only pretrain using the original generated-gripper contacts."""

from configs.panda_comparison_common import configure_post_contact_reuse
from configs.panda_experiment_common import generated_gripper_post_pretrain_cfg


EXP_CFG = generated_gripper_post_pretrain_cfg()
EXP_CFG.name = "panda_general_post_pretrain"
EXP_CFG.general.name = EXP_CFG.name
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.pretrain.wandb_project = "ungrasp"
EXP_CFG.pretrain.wandb_run_name = EXP_CFG.name
configure_post_contact_reuse(EXP_CFG)

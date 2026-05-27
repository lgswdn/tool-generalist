"""UniCORN contact-patch pretraining on full-tool data without point-cloud augmentation."""

from configs.config_utils import clone_cfg
from configs.experiments.unicorn_pretrain import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = clone_cfg(_BASE_EXP_CFG)
EXP_CFG.name = "unicorn_pretrain_no_aug"
EXP_CFG.general.name = "unicorn_pretrain_no_aug"
EXP_CFG.pretrain.name = "unicorn_contact_no_aug"
EXP_CFG.pretrain.augment = False
EXP_CFG.pretrain.wandb_run_name = "all_tools_eef_new_no_aug"
EXP_CFG.num_gpus = 8
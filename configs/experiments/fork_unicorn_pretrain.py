"""UniCORN contact-patch pretraining on the existing fork contact artifact."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import UNICORN_CONTACT_CFG, clone_cfg


CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_multitool_new/"
    "ded4300acdcb31c55ee93f2e86d0f96a0ead8fc4edaae22f749eb9ecbe362e61"
)


EXP_CFG = ExpCfg(name="fork_unicorn_pretrain")
EXP_CFG.general.name = "fork_sdf"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef/tools_fork.json"
EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.name = "contact_gen_multitool_new"
EXP_CFG.contact_gen.enabled = False

EXP_CFG.model.name = "unicorn_contact"
EXP_CFG.model.encoder_backend = "unicorn"
EXP_CFG.model.pretrained_encoder.name = "unicorn"
EXP_CFG.model.pretrained_encoder.adapter = "unicorn_strict"

EXP_CFG.pretrain = clone_cfg(UNICORN_CONTACT_CFG)
EXP_CFG.pretrain.dataset_manifest = CONTACT_DATASET
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "fork_pretrain"
EXP_CFG.pretrain.wandb_run_name = "unicorn_contact"

EXP_CFG.rl.enabled = False
EXP_CFG.rl.launch.distributed = False

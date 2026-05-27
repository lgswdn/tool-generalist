"""UniCORN contact-patch pretraining on the eef_new full-tool contact artifact."""

from configs.config_exp import ExpCfg
from configs.config_pretrain import UNICORN_CONTACT_CFG, clone_cfg
from configs.experiments.multitools_full_tool_contact import FULL_YES_MANIFEST


CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_full_tool/"
    "281987b90b894c5a84c97b9b0c89bca2d8711036c52e2d2b3f7f0a65f7d94535"
)


EXP_CFG = ExpCfg(name="unicorn_pretrain")
EXP_CFG.general.name = "unicorn_pretrain"
EXP_CFG.general.tools_selected_json = "/mnt/project/world_model/tool_generalist/eef_new/tools_selected.json"
EXP_CFG.general.objects_manifest = FULL_YES_MANIFEST
EXP_CFG.num_gpus = 4

EXP_CFG.contact_gen.name = "contact_gen_full_tool"
EXP_CFG.contact_gen.enabled = False

EXP_CFG.model.name = "unicorn_contact"
EXP_CFG.model.encoder_backend = "unicorn"
EXP_CFG.model.pretrained_encoder.name = "unicorn"
EXP_CFG.model.pretrained_encoder.adapter = "unicorn_strict"

EXP_CFG.pretrain = clone_cfg(UNICORN_CONTACT_CFG)
EXP_CFG.pretrain.dataset_manifest = CONTACT_DATASET
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "unicorn_pretrain"
EXP_CFG.pretrain.wandb_run_name = "all_tools_eef_new"

EXP_CFG.rl.enabled = False
EXP_CFG.rl.launch.distributed = False

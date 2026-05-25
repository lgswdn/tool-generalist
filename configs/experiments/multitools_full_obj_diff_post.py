"""Full-DGN object-to-object contact dataset with diff+post pretraining."""

from copy import deepcopy

from configs.config_pretrain import DIFF_CFG, clone_cfg
from configs.experiments.multitools_full_obj_contact import EXP_CFG as _CONTACT_EXP_CFG


EXP_CFG = deepcopy(_CONTACT_EXP_CFG)
EXP_CFG.name = "multitools_full_obj_diff_post"
EXP_CFG.general.name = "multitools_full_obj_diff_post"
EXP_CFG.model.name = "multitool_full_obj_diff_post"
EXP_CFG.num_gpus = 8

# Use the already materialized object-as-tool files directly.  The object2object
# contact run has partial failures, so its top-level contact artifact is not
# marked complete even though many final .pt files are usable for pretraining.
EXP_CFG.contact_gen.enabled = False
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.contact_gen.shard_count = 1
EXP_CFG.contact_gen.shard_index = 0

EXP_CFG.pretrain = clone_cfg(DIFF_CFG)
EXP_CFG.pretrain.name = "diff_post"
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = True
EXP_CFG.pretrain.enabled_heads = ["diff", "postcontact"]
EXP_CFG.pretrain.tasks.sdf = False
EXP_CFG.pretrain.tasks.diffusion = True
EXP_CFG.pretrain.tasks.postcontact = True
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "multitools_full_obj_pretrain"
EXP_CFG.pretrain.wandb_run_name = "diff_post"
EXP_CFG.pretrain.dataset_manifest = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_full_obj_as_tool/"
    "defdafd09677e86629006c0c3a158dba3499bc96535009b6f94003548731cebd"
)
EXP_CFG.pretrain.condition_normalization = True
EXP_CFG.pretrain.condition_norm_sample_files = 64
EXP_CFG.pretrain.epochs = 20
EXP_CFG.pretrain.optimizer.learning_rate = 3e-4
EXP_CFG.pretrain.optimizer.min_learning_rate = 3e-5

EXP_CFG.rl.enabled = False
EXP_CFG.rl.launch.distributed = False

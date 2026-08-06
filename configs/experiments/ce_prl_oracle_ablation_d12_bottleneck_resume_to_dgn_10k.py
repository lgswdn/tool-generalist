"""Give the completed rank-10 bottleneck teacher 5k more DGN iterations."""

from copy import deepcopy

from configs.experiments.ce_prl_oracle_rebuild_bottleneck_dgn_5k import (
    EXP_CFG as PARENT_EXP_CFG,
)
from configs.oracle_pointnet_rebuild_common import SOURCE_PRETRAIN_CHECKPOINT
from configs.panda_comparison_common import completed_parent_checkpoint
from configs.panda_experiment_common import GENERATED_GRIPPER_NEW_PATHS_YAML


PARENT = "ce_prl_oracle_rebuild_d12_bottleneck_dgn_5k"
NAME = "ce_prl_oracle_ablation_d12_bottleneck_resume_to_dgn_10k"

EXP_CFG = deepcopy(PARENT_EXP_CFG)
EXP_CFG.name = NAME
EXP_CFG.general.name = NAME
EXP_CFG.rl.name = NAME
EXP_CFG.rl.launch.run_name = NAME
EXP_CFG.rl.init_checkpoint = None
EXP_CFG.rl.resume_checkpoint = completed_parent_checkpoint(
    PARENT,
    contact_name="no-contact",
    encoder_family="TCE",
    expected_bottleneck_rank=10,
    expected_pretrained_encoder_checkpoint=str(SOURCE_PRETRAIN_CHECKPOINT),
    expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
    expected_max_iterations=5_000,
    expected_num_gpus=8,
    checkpoint_filename="model_last.pt",
)
# OnPolicyRunner interprets max_iterations as additional iterations on resume.
EXP_CFG.rl.ppo.max_iterations = 5_000
EXP_CFG.rl.launch.wandb_project = "ungrasp"

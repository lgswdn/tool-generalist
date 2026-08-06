"""Combined grippers: D4 kinematic encoder, paper contacts, DGN 5k."""

from configs.panda_comparison_common import configure_full_yes_comparison
from configs.panda_experiment_common import general_d4_full_rl_cfg


EXP_CFG = general_d4_full_rl_cfg(
    "ce_general_d4_full_paper_kinematic_dgn_5k",
    contact_quality="paper",
    architecture="kinematic",
)
configure_full_yes_comparison(EXP_CFG)

# Keep this run fully local: disable W&B for both encoder pretraining and RL.
EXP_CFG.general.wandb.enabled = False
EXP_CFG.general.wandb.mode = "disabled"
EXP_CFG.pretrain.logger = "none"
EXP_CFG.pretrain.wandb_mode = "disabled"
EXP_CFG.rl.launch.logger = "tensorboard"

# Use two ranks so the cross-embodiment environment keeps its exact 50/50
# generated-gripper / one-DoF-gripper assignment.
EXP_CFG.num_gpus = 2
EXP_CFG.rl.launch.distributed = True

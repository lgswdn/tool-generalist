"""Resume rich scratch point-cloud PointNet RL from iteration 3,460 to 5,000."""

from copy import deepcopy

from configs.experiments.panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k import (
    EXP_CFG as BASE_EXP_CFG,
)


RESUME_ITERATION = 3460
TARGET_ITERATION = 5000
RESUME_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k/"
    "no-contact/oracle_pointcloud_pointnet/"
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k/"
    "20260720T070646Z/model_best.pt"
)


EXP_CFG = deepcopy(BASE_EXP_CFG)
EXP_CFG.name = "panda_general_oracle_pointcloud_pointnet_rich_scratch_resume_to_5k"
EXP_CFG.general.name = EXP_CFG.name
EXP_CFG.rl.name = EXP_CFG.name
EXP_CFG.rl.init_checkpoint = None
EXP_CFG.rl.resume_checkpoint = RESUME_CHECKPOINT
# OnPolicyRunner interprets this as additional iterations after loading.
EXP_CFG.rl.ppo.max_iterations = TARGET_ITERATION - RESUME_ITERATION

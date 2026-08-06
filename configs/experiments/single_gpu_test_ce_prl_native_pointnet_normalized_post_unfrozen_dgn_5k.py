"""Single-GPU test clone of the unfrozen normalized native-PointNet DGN5k run."""

from copy import deepcopy

from configs.experiments.ce_prl_native_pointnet_normalized_post_unfrozen_dgn_5k import (
    EXP_CFG as BASE_EXP_CFG,
)


NAME = "single_gpu_test_ce_prl_native_pointnet_normalized_post_unfrozen_dgn_5k"

EXP_CFG = deepcopy(BASE_EXP_CFG)
EXP_CFG.name = NAME
EXP_CFG.general.name = NAME
EXP_CFG.pretrain.name = NAME
EXP_CFG.pretrain.wandb_run_name = NAME
EXP_CFG.rl.name = NAME
EXP_CFG.rl.launch.run_name = NAME
EXP_CFG.num_gpus = 1
EXP_CFG.rl.env.generated_parallel_finger_velocity_limit_m_s = 2.61

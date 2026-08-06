"""Official Panda-gripper RL using TCE contact-pretrained encoder weights."""

from configs.panda_experiment_common import official_panda_unicorn_rl_cfg


EXP_CFG = official_panda_unicorn_rl_cfg("panda_gripper_unicorn_ours", ours_tce=True)

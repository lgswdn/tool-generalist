"""Full-DGN selected-tool contact dataset with diff pretraining and RL."""

from copy import deepcopy

from configs.config_pretrain import DIFF_CFG, clone_cfg
from configs.experiments.multitools_full_tool_contact import EXP_CFG as _CONTACT_EXP_CFG


EXP_CFG = deepcopy(_CONTACT_EXP_CFG)
EXP_CFG.name = "multitools_full_tool_diff"
EXP_CFG.general.name = "multitools_full_tool_diff"
EXP_CFG.model.name = "multitool_full_tool_diff_only"
EXP_CFG.num_gpus = 8

EXP_CFG.contact_gen.enabled = True
EXP_CFG.contact_gen.regenerate = False
EXP_CFG.contact_gen.shard_count = 1
EXP_CFG.contact_gen.shard_index = 0

EXP_CFG.pretrain = clone_cfg(DIFF_CFG)
EXP_CFG.pretrain.enabled = True
EXP_CFG.pretrain.retrain = True
EXP_CFG.pretrain.logger = "wandb"
EXP_CFG.pretrain.wandb_project = "multitools_full_tool_pretrain"
EXP_CFG.pretrain.wandb_run_name = "diff_only"
EXP_CFG.pretrain.epochs = 20
EXP_CFG.pretrain.optimizer.learning_rate = 3e-4
EXP_CFG.pretrain.optimizer.min_learning_rate = 3e-5

EXP_CFG.rl.enabled = True
EXP_CFG.rl.name = "multitools_full_tool_diff"
EXP_CFG.rl.launch.distributed = True
EXP_CFG.rl.env.num_envs = 1024
EXP_CFG.rl.launch.logger = "wandb"
EXP_CFG.rl.launch.wandb_project = "all_tools_full_tool"
EXP_CFG.rl.launch.run_name = "multitools_full_tool_diff"
EXP_CFG.rl.table.enabled = True
EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]
EXP_CFG.rl.domain_randomization.ground.material.enabled = False
EXP_CFG.rl.ppo.save_interval = 500
EXP_CFG.rl.ppo.entropy_coef = 0.006
EXP_CFG.rl.separate_actor_critic_fusion = True
EXP_CFG.rl.reward.object_goal_tracking_term_weight = 3
EXP_CFG.rl.reward.object_goal_tracking_fine_term_weight = 6

"""CPU-only UniCORN pretrain smoke configuration."""

from copy import deepcopy

from configs.experiments.fork_unicorn_pretrain import EXP_CFG as _BASE_EXP_CFG


EXP_CFG = deepcopy(_BASE_EXP_CFG)
EXP_CFG.name = "fork_unicorn_pretrain_smoke"
EXP_CFG.num_gpus = 0

EXP_CFG.pretrain.name = "unicorn_contact_smoke"
EXP_CFG.pretrain.max_files = 2
EXP_CFG.pretrain.batch.batch_size = 4
EXP_CFG.pretrain.batch.num_workers = 0
EXP_CFG.pretrain.epochs = 1
EXP_CFG.pretrain.logger = "none"
EXP_CFG.pretrain.wandb_project = None
EXP_CFG.pretrain.wandb_run_name = None
EXP_CFG.pretrain.device = "cpu"

EXP_CFG.rl.enabled = False
EXP_CFG.rl.launch.distributed = False

"""Example dry-run experiment config."""

from configs.config_exp import ExpCfg


EXP_CFG = ExpCfg(name="example")
EXP_CFG.general.name = "example"
EXP_CFG.contact_gen.enabled = True

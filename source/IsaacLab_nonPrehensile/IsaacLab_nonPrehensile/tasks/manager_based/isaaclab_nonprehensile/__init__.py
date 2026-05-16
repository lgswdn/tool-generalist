import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="tool-sdf-v0",
    entry_point=f"{__name__}.env_tool:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_tool:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:TGPPORunnerCfg",
    },
)

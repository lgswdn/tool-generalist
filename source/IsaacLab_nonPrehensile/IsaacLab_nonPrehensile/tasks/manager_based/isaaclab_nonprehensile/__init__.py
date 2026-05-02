import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="tool-v0",
    entry_point=f"{__name__}.env_tool:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_tool:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:NonPrehensilePPORunnerCfg",
    },
)

gym.register(
    id="tool-momentum-v0",
    entry_point=f"{__name__}.env_tool_momentum:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_tool_momentum:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:MomentumPPORunnerCfg",
    },
)

gym.register(
    id="tool-sdf-v0",
    entry_point=f"{__name__}.env_tool:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_tool:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:SDFPPORunnerCfg",
    },
)

gym.register(
    id="tool-point2vec-v0",
    entry_point=f"{__name__}.env_tool:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_tool:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:Point2VecPPORunnerCfg",
    },
)

# Auto-register SDF experiment variants from SDF_VARIANTS dict.
# Each variant is registered as "tool-sdf-<suffix>" with overridden policy/runner params.
from .agents.config.rsl_rl_ppo_cfg import SDF_VARIANTS, make_sdf_variant

for _suffix, _overrides in SDF_VARIANTS.items():
    _VariantCfg, _gym_id = make_sdf_variant(_suffix, _overrides)
    # Inject the class into the agents.config module so the entry_point string resolves
    setattr(agents.config.rsl_rl_ppo_cfg, _VariantCfg.__name__, _VariantCfg)
    gym.register(
        id=_gym_id,
        entry_point=f"{__name__}.env_tool:NonPrehensileEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_tool:NonPrehensileEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:{_VariantCfg.__name__}",
        },
    )

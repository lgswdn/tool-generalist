from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import field


@configclass
class ToolUnicornActorCriticCfg:
    """Config for Tool Unicorn Actor-Critic with 3 clouds (object + tool + merged)."""

    class_name: str = "ActorCriticToolUnicorn"

    # Point cloud layout
    pc_point_dim: int = 3
    object_pc_num_points: int = 512
    tool_pc_num_points: int = 512
    merged_pc_num_points: int = 512

    # Unicorn encoder settings
    unicorn_cfg: dict = field(default_factory=dict)
    unicorn_ckpt: str = '/mnt/afs/zhuwenxuan/project/inp/checkpoints/unicorn'
    freeze_unicorn: bool = True        # Freeze pretrained encoder
    encoder_strict_load: bool = True

    # Fusion MLP
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    fusion_use_norm: bool = True
    fusion_norm_type: str | None = "layer"

    # Actor head
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    actor_use_norm: bool = True
    actor_norm_type: str | None = "layer"
    actor_output_activation: bool = False

    # Critic head (larger capacity)
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128,64])
    critic_use_norm: bool = True
    critic_norm_type: str | None = "layer"

    # SD-Cross settings
    use_sd_cross: bool = True
    sd_num_query: int = 16
    sd_num_query_object: int = 4
    sd_emb_dim: int = 128
    sd_cat_query: bool = False
    sd_cat_ctx: bool = True

    # Activation / noise
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class NonPrehensilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the tool-generalist non-prehensile task."""

    # Training parameters
    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    # Logging / experiment identifiers
    experiment_name = "franka_nonprehensile"

    # Observation normalization
    empirical_normalization = False

    # Policy network
    policy = ToolUnicornActorCriticCfg()

    # PPO algorithm hyper-parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,         
        use_clipped_value_loss=True,
        clip_param=0.3,                
        entropy_coef=0.006,               
        num_learning_epochs=8,      
        num_mini_batches=8,            
        learning_rate=5.0e-5,        
        schedule="adaptive",  
        gamma=0.99,               
        lam=0.95,                 
        desired_kl=0.016, 
        max_grad_norm=1.0,
    )


# =============================================================================
# Momentum (7D) configuration — for tool-momentum-v0
# =============================================================================

@configclass
class MomentumActorCriticCfg:
    """Config for ActorCriticMomentum with 7D point clouds (xyz + mass + velocity).

    Observation layout (env_tool_momentum):
        object_cloud (512*7=3584) + tool_cloud_as_ee (512*7=3584) + extra_state (46)
    The tool cloud is treated as EE (end-effector) in the Momentum encoder.
    """

    class_name: str = "ActorCriticMomentum"

    # Point cloud / state layout
    point_dim: int = 7
    num_points: int = 512
    num_obstacles: int = 0        # no obstacles
    num_ee_points: int = 512      # tool cloud treated as EE
    robot_state_dim: int = 14

    # Momentum encoder settings — propagate layout to encoder internals
    momentum_cfg: dict = field(default_factory=lambda: {
        "num_points_per_object": 512,
        "num_obstacles": 0,
        "num_ee_points": 512,
    })
    momentum_ckpt: str | None = '/mnt/afs/zhuwenxuan/project/inp/checkpoints/point_encoder_action_global_step_044950.pt'
    freeze_momentum: bool = True
    encoder_strict_load: bool = False

    # StateDependentCrossFeatNet settings (matches reference)
    use_learnable_query_tokens: bool = False  # Reference uses SD-cross, not learnable queries
    sd_num_query: int = 16
    sd_num_query_object: int | None = 8
    sd_emb_dim: int = 128
    sd_cat_query: bool = False
    sd_cat_ctx: bool = True
    sd_query_keys: tuple | None = None  # Default: ("extra_state",)

    # Learnable query tokens settings (when use_learnable_query_tokens=True)
    num_query_object_tokens: int | None = None
    num_query_tokens: int = 16
    cross_attn_heads: int = 4
    cross_attn_layers: int = 1
    cross_attn_ff_dim: int | None = None
    cross_attn_dropout: float = 0.0

    # Actor / Critic heads (matches reference)
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])

    # Activation / noise
    activation: str = "gelu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class MomentumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the 7D momentum-based tool-generalist task."""

    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    experiment_name = "franka_nonprehensile_momentum"

    empirical_normalization = False

    policy = MomentumActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.3,
        entropy_coef=0.006,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
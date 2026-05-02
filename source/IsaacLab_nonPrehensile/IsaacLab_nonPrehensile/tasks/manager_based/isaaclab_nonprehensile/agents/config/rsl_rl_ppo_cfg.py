from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import field


POINT2VEC_CKPT_PATH = "/mnt/project/world_model/tool_generalist/model/pre_point2vec-epoch.799-step.64800.ckpt"


@configclass
class MultiICPActorCriticCfg:
    """Config for Multi-ICP Actor-Critic with object cloud + tool cloud (as obstacle).

    Observation layout (env_tool):
        object_cloud (512*3=1536) | tool_cloud (512*3=1536) | hand_state (9) | rest (robot_state+prev_action+rel_goal+phys_params)
    Uses ActorCriticMultiICP_HandState: hand_state is passed to ICP encoder, rest goes to SD-Cross.
    """

    class_name: str = "ActorCriticMultiICP_HandState"

    # Point cloud layout: 1 object + 1 tool-as-obstacle
    num_obstacles: int = 1
    num_large_obstacles: int = 1

    # ICP pretrained weights
    icp_weights_path: str | None = '/mnt/afs/zhuwenxuan/project/inp/512-32-balanced-SAM-wd-5e-05-920'
    freeze_icp: bool = True

    icp_point_dim: int = 3
    icp_num_points: int = 512

    # Network architecture
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    fusion_use_norm: bool = True
    fusion_norm_type: str = "layer"

    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    actor_use_norm: bool = True
    actor_norm_type: str = "layer"
    actor_output_activation: bool = False

    critic_hidden_dims: list[int] = field(default_factory=lambda: [128])
    critic_use_norm: bool = True
    critic_norm_type: str = "layer"

    # SD-Cross settings
    use_sd_cross: bool = True
    sd_num_query: int = 16
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
    policy = MultiICPActorCriticCfg()

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
        object_cloud (512*7=3584) + tool_obstacle (512*7=3584) + tool_ee (512*7=3584) + extra_state (46)
    The tool cloud appears in both the obstacle and EE slots.
    """

    class_name: str = "ActorCriticMomentum"

    # Point cloud / state layout
    point_dim: int = 7
    num_points: int = 512
    num_obstacles: int = 1        # tool cloud in obstacle slot
    num_ee_points: int = 512      # tool cloud also as EE
    robot_state_dim: int = 14

    # Momentum encoder settings — propagate layout to encoder internals
    momentum_cfg: dict = field(default_factory=lambda: {
        "num_points_per_object": 512,
        "num_obstacles": 1,
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

    # Actor / Critic heads
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128])

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

@configclass
class Point2VecActorCriticCfg:
    """Config for Point2Vec Actor-Critic on object + tool xyz point clouds.

    Observation layout (env_tool):
        object_cloud (512*3=1536) | tool_cloud (512*3=1536) | extra_state
    """

    class_name: str = "ActorCriticPoint2Vec"

    # Point cloud / state layout
    point_dim: int = 3  # Only coordinates supported (3D)
    num_points: int = 512
    num_obstacles: int = 1

    # Point2Vec encoder settings
    point2vec_ckpt_path: str | None = POINT2VEC_CKPT_PATH
    freeze_point2vec: bool = True

    # Tokenizer settings
    tokenizer_num_groups: int = 64
    tokenizer_group_size: int = 32
    tokenizer_group_radius: float | None = None

    # Encoder settings
    encoder_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    encoder_dropout: float = 0.0
    encoder_attention_dropout: float = 0.05
    encoder_drop_path_rate: float = 0.25
    encoder_add_pos_at_every_layer: bool = True

    # Feature aggregation
    use_max_pooling: bool = True
    use_mean_pooling: bool = True

    # Data transformations
    train_transformations: list[str] = field(default_factory=lambda: ["center", "unit_sphere"])
    val_transformations: list[str] = field(default_factory=lambda: ["center", "unit_sphere"])

    # StateDependentCrossFeatNet settings
    use_sd_cross: bool = True
    sd_num_query: int = 16
    sd_emb_dim: int = 128
    sd_cat_query: bool = False
    sd_cat_ctx: bool = True

    # Actor / Critic heads
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])

    # Activation / noise
    activation: str = "gelu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class Point2VecPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Point2Vec-encoder-based training."""

    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    experiment_name = "franka_nonprehensile_point2vec"

    empirical_normalization = False

    policy = Point2VecActorCriticCfg()

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
# SDF Encoder configuration — for tool-sdf-v0
# =============================================================================

@configclass
class SDFActorCriticCfg:
    """Config for Actor-Critic using SDFPointCloudEncoder (joint ViT encoder).

    Observation layout:
        object_cloud (512*3=1536) | tool_cloud (512*3=1536) | extra_state
    Uses ActorCriticSDF: joint ViT encoder processes tool + object together.
    """

    class_name: str = "ActorCriticSDF"

    # Point cloud settings
    num_points: int = 512
    point_dim: int = 3
    patch_size: int = 32

    # Encoder architecture
    encoder_channel: int = 128
    vit_depth: int = 4
    vit_heads: int = 4

    # Encoder weights (pretrained from SDF pretraining)
    encoder_weights_path: str | None = "/path/to/best.pt"
    freeze_encoder: bool = True

    # StateDependentCrossFeatNet settings
    use_learnable_query_tokens: bool = False
    sd_num_query: int = 16
    sd_num_query_object: int | None = 8
    sd_emb_dim: int = 128
    sd_cat_query: bool = False
    sd_cat_ctx: bool = True
    sd_query_keys: tuple | None = None

    # Learnable query tokens settings (when use_learnable_query_tokens=True)
    num_query_object_tokens: int | None = None
    num_query_tokens: int = 16
    cross_attn_heads: int = 4
    cross_attn_layers: int = 1
    cross_attn_ff_dim: int | None = None
    cross_attn_dropout: float = 0.0

    # Actor / Critic heads
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128])

    # Activation / noise
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class SDFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for SDF-encoder-based task.

    Uses the same NonPrehensileEnv from env_tool.py but with ActorCriticSDF policy.
    """

    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    experiment_name = "franka_nonprehensile_sdf"

    empirical_normalization = False

    policy = SDFActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.3,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=16,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


# =============================================================================
# SDF Variant Registry — experiment variants with shared encoder
# =============================================================================
# Each entry: "suffix" -> {policy overrides} or {"policy": {...}, "runner": {...}}
# Registered as gym task "tool-sdf-<suffix>"
#
# Usage:
#   python train.py --task tool-sdf-frozen-v0
#   python train.py --task tool-sdf-learnable-query-v0
#   python train.py --task tool-sdf-finetune-v0
#
# To add a new experiment, just add an entry here — no new classes needed.
# =============================================================================

SDF_VARIANTS: dict[str, dict] = {
    "teardrop-point-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_point/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_point",
        },
    },
    "teardrop-patch-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_patch/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_patch",
        },
    },
    "teardrop-movement-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_movement/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_movement",
        },
    },
    "teardrop-movement-patch-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_movement_patch/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_movement_patch",
        },
    },
    "teardrop-diffusion-patch-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_sdf_diff/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_diffusion",
        },
    },
    "teardrop-joint-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/teardrop_joint/best.pt",
        },
        "runner": {
            "experiment_name": "teardrop_sdf_joint",
        },
    },
    "multitool-patch-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/tool_sdf_patch/best.pt",
        },
        "runner": {
            "experiment_name": "multitool_sdf_patch",
        }
    },
    "multitool-point-v0": {
        "policy": {
            "encoder_weights_path": "/mnt/project/world_model/tool_generalist/model/encoder/tool_sdf_point/best.pt",
        },
        "runner": {
            "experiment_name": "multitool_sdf_point",
        }
    }
}


def make_sdf_variant(suffix: str, overrides: dict):
    """Create a (RunnerCfg class, gym_id) pair from overrides on the base SDF config.

    ``overrides`` can be:
      - A flat dict of SDFActorCriticCfg field overrides (shorthand)
      - {"policy": {...}, "runner": {...}} for full control

    Returns (RunnerCfgClass, gym_id_string).
    """
    policy_ov = overrides.get("policy", overrides if "runner" not in overrides else {})
    runner_ov = overrides.get("runner", {})

    # --- Build policy config ---
    policy_cfg = SDFActorCriticCfg()
    for k, v in policy_ov.items():
        setattr(policy_cfg, k, v)

    # --- Build runner config class (dynamic @configclass) ---
    runner_attrs = {
        "policy": policy_cfg,
        "experiment_name": runner_ov.get(
            "experiment_name", f"franka_sdf_{suffix.replace('-', '_')}"
        ),
    }
    # Apply any other runner-level overrides
    for k, v in runner_ov.items():
        if k != "experiment_name":
            runner_attrs[k] = v

    cls_name = f"SDFPPORunnerCfg_{'_'.join(suffix.split('-'))}"
    VariantRunnerCfg = configclass(
        type(cls_name, (SDFPPORunnerCfg,), runner_attrs)
    )

    gym_id = f"tool-sdf-{suffix}"
    return VariantRunnerCfg, gym_id

"""Model config declarations for the new experiment framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


PRETRAINED_ENCODER_ADAPTERS = (
    "tce_strict",
    "point2vec_native",
    "icp_legacy",
    "unicorn_strict",
    "oracle_none",
    "oracle_pointmesh_pointnet_strict",
    "oracle_pointcloud_pointnet_strict",
    "oracle_pointcloud_pointnet_pretrain_strict",
    "oracle_pointcloud_pointnet_normalized_pretrain_strict",
    "oracle_pointcloud_pointnet_rl_encoder_strict",
    "oracle_pointcloud_patch_oracle_strict",
    "patch_distance_pointnet_strict",
)


@dataclass
class EncoderCfg:
    name: str = "none"
    encoder_type: str = "none"
    output_dim: int = 0
    checkpoint_path: Optional[str] = None
    trainable: bool = True


@dataclass
class KinematicConditioningCfg:
    enabled: bool = False
    state_fractions: tuple[float, float, float] = (0.0, 0.5, 1.0)
    attention_layers: int = 1
    delta_std: float = 0.15


@dataclass
class TCECfg(EncoderCfg):
    name: str = "TCE"
    encoder_type: str = "TCE"
    output_dim: int = 128
    num_points: int = 512
    patch_size: int = 32
    encoder_channel: int = 128
    vit_depth: int = 12
    vit_heads: int = 4
    vit_attention_mode: str = "joint_self"
    rl_token_source: str = "encoder"
    encoder_token_pca_rank: int = 128
    encoder_token_pca_path: Optional[str] = None
    encoder_token_bottleneck_rank: int = 128
    encoder_token_bottleneck_pca_path: Optional[str] = None
    kinematic_conditioning: KinematicConditioningCfg = field(
        default_factory=KinematicConditioningCfg
    )


@dataclass
class P2VCfg(EncoderCfg):
    name: str = "P2V"
    encoder_type: str = "Point2Vec"
    output_dim: int = 384
    num_points: int = 512
    token_dim: int = 384
    tokenizer_num_groups: int = 128
    tokenizer_group_size: int = 32
    tokenizer_group_radius: Optional[float] = None
    encoder_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    encoder_dropout: float = 0.0
    encoder_attention_dropout: float = 0.0
    encoder_drop_path_rate: float = 0.2
    encoder_add_pos_at_every_layer: bool = True
    train_transformations: list[str] = field(default_factory=lambda: ["unit_sphere"])
    val_transformations: list[str] = field(default_factory=lambda: ["unit_sphere"])
    checkpoint_path: Optional[str] = None


@dataclass
class ICPCfg(EncoderCfg):
    name: str = "ICP"
    encoder_type: str = "ICP"
    output_dim: int = 128
    num_points: int = 512
    checkpoint_path: Optional[str] = None


@dataclass
class UnicornCfg(EncoderCfg):
    name: str = "UniCORN"
    encoder_type: str = "UniCORN"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    vit_depth: int = 4
    vit_heads: int = 4
    rl_token_source: str = "encoder"
    checkpoint_path: Optional[str] = None


@dataclass
class PatchDistancePointNetCfg(EncoderCfg):
    """XYZ-only PointNet pretrained on distance to its discrete patch points."""

    name: str = "patch_distance_pointnet"
    encoder_type: str = "patch_distance_pointnet"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    point_scale_m: float = 0.05
    query_count: int = 24
    supervised_patches_per_cloud: int = 8
    query_min_offset_m: float = 0.0005
    query_max_offset_m: float = 0.03
    distance_scale_m: float = 0.03
    patch_center_scale_m: float = 0.30
    checkpoint_path: Optional[str] = None


@dataclass
class OraclePatchCfg(EncoderCfg):
    """Explicit patch-distance representation with no pretrained weights."""

    name: str = "oracle_patch"
    encoder_type: str = "oracle_patch"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    include_contact_feature: bool = True
    contact_eps: float = 0.002
    center_scale_m: float = 0.30
    distance_scale_m: float = 0.10
    patch_relative_scale_m: float = 0.05
    log_distance_resolution_m: float = 0.005
    log_distance_cap_m: float = 0.05
    normalization_clip: float = 5.0


@dataclass
class OraclePointMeshPointNetCfg(EncoderCfg):
    """Patchwise PointNet over privileged ``(x, y, z, unsigned distance)``."""

    name: str = "oracle_pointmesh_pointnet"
    encoder_type: str = "oracle_pointmesh_pointnet"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    coordinate_scale_m: float = 0.30
    distance_scale_m: float = 0.10
    normalization_clip: float = 5.0


@dataclass
class OraclePointCloudPointNetCfg(EncoderCfg):
    """Fast patchwise PointNet using nearest opposite point-cloud geometry."""

    name: str = "oracle_pointcloud_pointnet"
    encoder_type: str = "oracle_pointcloud_pointnet"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    nearest_frame_batch_size: int = 64
    # "fast11" is the fitted probe contract.  "rich21" adds cheap local
    # distance/displacement/contact-scale features for learning directly in RL.
    feature_mode: str = "fast11"
    # False keeps the fitted probe's input normalization but initializes every
    # learned PointNet/projection weight from scratch for a controlled RL ablation.
    load_fitted_weights: bool = True
    # The fitted probe reconstructs a rank-10 Unicorn token.  Scratch RL can
    # instead produce a direct 128D patch token with no 10D bottleneck.
    use_rank10_bottleneck: bool = True
    # "patches": 32 pooled patch tokens; "points": 1024 unpooled point tokens.
    token_mode: str = "patches"
    # "fast11_probe_v1" is the fixed mean/std contract isolated by the
    # normalization ablation. "identity" preserves legacy native experiments.
    input_normalization: str = "identity"


@dataclass
class OraclePointCloudPatchOracleCfg(EncoderCfg):
    """Deep analytic patch MLP over nearest opposite point-cloud geometry."""

    name: str = "oracle_pointcloud_patch_oracle"
    encoder_type: str = "oracle_pointcloud_patch_oracle"
    output_dim: int = 128
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    nearest_frame_batch_size: int = 64


@dataclass
class ConcertoCfg(EncoderCfg):
    name: str = "Concerto"
    encoder_type: str = "Concerto"
    output_dim: int = 768
    num_points: int = 512
    feature_dim: int = 768
    checkpoint_path: Optional[str] = None


@dataclass
class PretrainedEncoderCfg:
    name: str = "tce"
    checkpoint_path: Optional[str] = None
    schema: str = "pretrain_checkpoint_v1"
    adapter: str = "tce_strict"


@dataclass
class PolicyFusionCfg:
    sd_num_query: int = 8
    query_dim: int = 128
    relative_translation_query_tokens: int = 2
    reuse_pretrain_pose_cross_attn: bool = False
    context_dim: Optional[int] = None
    cross_attn_heads: int = 4
    cross_attn_layers: int = 2
    # "joint" lets every layer attend to all tokens. "tool_then_object"
    # routes layer 1 to gripper/tool tokens and layer 2 to object tokens.
    cross_attn_token_order: str = "joint"
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])


@dataclass
class ModelCfg:
    name: str = "model_default"
    encoder_backend: str = "tce"
    tce: TCECfg = field(default_factory=TCECfg)
    p2v: P2VCfg = field(default_factory=P2VCfg)
    icp: ICPCfg = field(default_factory=ICPCfg)
    unicorn: UnicornCfg = field(default_factory=UnicornCfg)
    patch_distance_pointnet: PatchDistancePointNetCfg = field(
        default_factory=PatchDistancePointNetCfg
    )
    oracle_patch: OraclePatchCfg = field(default_factory=OraclePatchCfg)
    oracle_pointmesh_pointnet: OraclePointMeshPointNetCfg = field(
        default_factory=OraclePointMeshPointNetCfg
    )
    oracle_pointcloud_pointnet: OraclePointCloudPointNetCfg = field(
        default_factory=OraclePointCloudPointNetCfg
    )
    oracle_pointcloud_patch_oracle: OraclePointCloudPatchOracleCfg = field(
        default_factory=OraclePointCloudPatchOracleCfg
    )
    pretrained_encoder: PretrainedEncoderCfg = field(default_factory=PretrainedEncoderCfg)
    policy_fusion: PolicyFusionCfg = field(default_factory=PolicyFusionCfg)

    @property
    def num_points(self) -> int:
        return self.encoder.num_points

    @num_points.setter
    def num_points(self, value: int) -> None:
        self.encoder.num_points = value

    @property
    def patch_size(self) -> int:
        return self.tce.patch_size

    @patch_size.setter
    def patch_size(self, value: int) -> None:
        self.tce.patch_size = value

    @property
    def encoder_channel(self) -> int:
        return self.tce.encoder_channel

    @encoder_channel.setter
    def encoder_channel(self, value: int) -> None:
        self.tce.encoder_channel = value

    @property
    def vit_depth(self) -> int:
        return self.tce.vit_depth

    @vit_depth.setter
    def vit_depth(self, value: int) -> None:
        self.tce.vit_depth = value

    @property
    def vit_heads(self) -> int:
        return self.tce.vit_heads

    @vit_heads.setter
    def vit_heads(self, value: int) -> None:
        self.tce.vit_heads = value

    @property
    def encoder(self) -> EncoderCfg:
        backend = self._normalized_encoder_backend()
        if backend == "tce":
            return self.tce
        if backend == "point2vec":
            return self.p2v
        if backend == "icp":
            return self.icp
        if backend == "unicorn":
            return self.unicorn
        if backend == "patch_distance_pointnet":
            return self.patch_distance_pointnet
        if backend == "oracle_patch":
            return self.oracle_patch
        if backend == "oracle_pointmesh_pointnet":
            return self.oracle_pointmesh_pointnet
        if backend == "oracle_pointcloud_pointnet":
            return self.oracle_pointcloud_pointnet
        if backend == "oracle_pointcloud_patch_oracle":
            return self.oracle_pointcloud_patch_oracle
        raise ValueError(f"Unsupported ModelCfg.encoder_backend: {self.encoder_backend!r}")

    @encoder.setter
    def encoder(self, value: EncoderCfg) -> None:
        if isinstance(value, TCECfg):
            self.tce = value
            self.encoder_backend = "tce"
            return
        if isinstance(value, P2VCfg):
            self.p2v = value
            self.encoder_backend = "point2vec"
            return
        if isinstance(value, ICPCfg):
            self.icp = value
            self.encoder_backend = "icp"
            return
        if isinstance(value, UnicornCfg):
            self.unicorn = value
            self.encoder_backend = "unicorn"
            return
        if isinstance(value, PatchDistancePointNetCfg):
            self.patch_distance_pointnet = value
            self.encoder_backend = "patch_distance_pointnet"
            return
        if isinstance(value, OraclePatchCfg):
            self.oracle_patch = value
            self.encoder_backend = "oracle_patch"
            return
        if isinstance(value, OraclePointMeshPointNetCfg):
            self.oracle_pointmesh_pointnet = value
            self.encoder_backend = "oracle_pointmesh_pointnet"
            return
        if isinstance(value, OraclePointCloudPointNetCfg):
            self.oracle_pointcloud_pointnet = value
            self.encoder_backend = "oracle_pointcloud_pointnet"
            return
        if isinstance(value, OraclePointCloudPatchOracleCfg):
            self.oracle_pointcloud_patch_oracle = value
            self.encoder_backend = "oracle_pointcloud_patch_oracle"
            return
        raise ValueError(
            "ModelCfg.encoder must be a TCECfg, P2VCfg, ICPCfg, UnicornCfg, "
            "PatchDistancePointNetCfg, "
            "OraclePatchCfg, OraclePointMeshPointNetCfg, or "
            "OraclePointCloudPointNetCfg, or OraclePointCloudPatchOracleCfg"
        )

    @property
    def concerto(self) -> ConcertoCfg:
        return ConcertoCfg()

    @property
    def actor_critic_class(self) -> str:
        if self._normalized_encoder_backend() == "point2vec":
            return "ActorCriticPoint2Vec"
        if self._normalized_encoder_backend() == "icp":
            return "ActorCriticICP"
        if self._normalized_encoder_backend() == "unicorn":
            return "ActorCriticTGUnicorn"
        return "ActorCriticTG"

    def _normalized_encoder_backend(self) -> str:
        value = self.encoder_backend.strip().lower()
        if value in {"tce", "tg"}:
            return "tce"
        if value in {"point2vec", "p2v"}:
            return "point2vec"
        if value in {"icp", "corn"}:
            return "icp"
        if value in {"unicorn", "universal_corn"}:
            return "unicorn"
        if value in {"patch_distance_pointnet", "patch_sdf_pointnet"}:
            return "patch_distance_pointnet"
        if value in {"oracle_patch", "oracle_sdf"}:
            return "oracle_patch"
        if value in {"oracle_pointmesh_pointnet", "oracle_unsigned_pointnet"}:
            return "oracle_pointmesh_pointnet"
        if value in {"oracle_pointcloud_pointnet", "oracle_fast_pointnet"}:
            return "oracle_pointcloud_pointnet"
        if value in {"oracle_pointcloud_patch_oracle", "oracle_fast_patch_oracle"}:
            return "oracle_pointcloud_patch_oracle"
        return value

    @property
    def hidden_dims(self) -> list[int]:
        return self.policy_fusion.fusion_hidden_dims

    @hidden_dims.setter
    def hidden_dims(self, value: list[int]) -> None:
        self.policy_fusion.fusion_hidden_dims = value

    @property
    def action_dim(self) -> None:
        return None

    @property
    def observation_dim(self) -> None:
        return None

    @property
    def physics_dim(self) -> None:
        return None

    def resolved_action_dim(self, fallback: int | None = None) -> int | None:
        return fallback

    def resolved_physics_dim(self, fallback: int | None = None) -> int | None:
        return fallback

    def resolved_observation_dim(self, fallback: int | None = None) -> int | None:
        return fallback

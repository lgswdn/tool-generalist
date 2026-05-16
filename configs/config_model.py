"""Model config declarations for the new experiment framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


PRETRAINED_ENCODER_ADAPTERS = ("tce_strict", "point2vec_native")


@dataclass
class EncoderCfg:
    name: str = "none"
    encoder_type: str = "none"
    output_dim: int = 0
    checkpoint_path: Optional[str] = None
    trainable: bool = True


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
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])


@dataclass
class ModelCfg:
    name: str = "model_default"
    encoder_backend: str = "tce"
    tce: TCECfg = field(default_factory=TCECfg)
    p2v: P2VCfg = field(default_factory=P2VCfg)
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
        raise ValueError("ModelCfg.encoder must be a TCECfg or P2VCfg")

    @property
    def concerto(self) -> ConcertoCfg:
        return ConcertoCfg()

    @property
    def actor_critic_class(self) -> str:
        if self._normalized_encoder_backend() == "point2vec":
            return "ActorCriticPoint2Vec"
        return "ActorCriticTG"

    def _normalized_encoder_backend(self) -> str:
        value = self.encoder_backend.strip().lower()
        if value in {"tce", "tg"}:
            return "tce"
        if value in {"point2vec", "p2v"}:
            return "point2vec"
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

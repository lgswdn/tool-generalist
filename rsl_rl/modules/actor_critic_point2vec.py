# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic network using Point2Vec point cloud encoder with StateDependentCrossFeatNet fusion."""

from __future__ import annotations

from typing import Optional
import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet

from rsl_rl.point2vec.modules.pointnet import PointcloudTokenizer
from rsl_rl.point2vec.modules.transformer import TransformerEncoder
from rsl_rl.point2vec.utils import transforms
from rsl_rl.point2vec.utils.checkpoint import extract_model_checkpoint


class ActorCriticPoint2Vec(nn.Module):
    """
    Actor-Critic network using Point2Vec as point cloud encoder.
    
    Architecture:
        1. Point cloud encoding with Point2Vec (tokenizer + encoder)
        2. Mean/Max pooling to aggregate features
        3. StateDependentCrossFeatNet fusion with extra state
        4. Fusion MLP
        5. Actor and Critic heads
    """
    
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        point_dim: int = 3,  # Only coordinates supported for now
        num_points: int = 512,
        num_obstacles: int = 1,
        # Point2Vec settings
        point2vec_ckpt_path: Optional[str] = None,
        freeze_point2vec: bool = True,
        # Tokenizer settings
        tokenizer_num_groups: int = 128,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: Optional[float] = None,
        # Encoder settings (will be loaded from checkpoint if available)
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        # Feature aggregation
        use_max_pooling: bool = True,
        use_mean_pooling: bool = True,
        # Data transformations
        train_transformations: list = ["center", "unit_sphere"],
        val_transformations: list = ["center", "unit_sphere"],
        # StateDependentCrossFeatNet settings
        use_sd_cross: bool = True,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        # MLP settings
        fusion_hidden_dims=(256, 128, 64),
        actor_hidden_dims=(64,),
        critic_hidden_dims=(64,),
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"ActorCriticPoint2Vec.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_obstacles = num_obstacles
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_point2vec = freeze_point2vec
        self.use_sd_cross = use_sd_cross
        self.use_max_pooling = use_max_pooling
        self.use_mean_pooling = use_mean_pooling

        # Point2Vec only supports 3D coordinates for now
        if point_dim != 3:
            raise ValueError(
                f"Point2Vec currently only supports point_dim=3 (coordinates only), got {point_dim}"
            )

        # Calculate observation layout
        object_pc_dim = self.num_points * self.point_dim
        obstacle_pc_dim = num_obstacles * self.num_points * self.point_dim
        self.pc_dim = object_pc_dim + obstacle_pc_dim

        print(f"[ActorCriticPoint2Vec] Point cloud layout:")
        print(f"  - Object points: {self.num_points} (dim: {object_pc_dim})")
        print(
            f"  - Obstacle points: {num_obstacles} x {self.num_points} = {num_obstacles * self.num_points} (dim: {obstacle_pc_dim})"
        )
        print(f"  - Total point cloud dim: {self.pc_dim}")
        print(f"  - Input feature dim per point: {self.point_dim} (coordinates only)")

        # Extra state dimension
        self.extra_state_dim = num_actor_obs - self.pc_dim
        print(f"  - Extra state dim: {self.extra_state_dim}")

        activation_fn = resolve_nn_activation(activation)

        # ----------------------------------------------------------------------
        # Point2Vec encoder setup
        # ----------------------------------------------------------------------
        print(f"[ActorCriticPoint2Vec] Initializing Point2Vec encoder...")

        # Build data transformations
        def build_transformation(name: str) -> transforms.Transform:
            if name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "scale":
                return transforms.PointcloudScaling(min=0.8, max=1.2)
            elif name == "rotate":
                return transforms.PointcloudRotation(dims=[1], deg=None)
            elif name == "translate":
                return transforms.PointcloudTranslation(0.2)
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        # Positional encoding
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        # Tokenizer
        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        # Encoder
        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        # Load pretrained checkpoint if provided
        if point2vec_ckpt_path is not None:
            # Resolve path (handle both relative and absolute paths)
            resolved_path = os.path.abspath(os.path.expanduser(point2vec_ckpt_path))
            if os.path.exists(resolved_path):
                print(f"[ActorCriticPoint2Vec] Loading pretrained checkpoint from {resolved_path}")
                self._load_pretrained_checkpoint(resolved_path)
            else:
                print(f"[ActorCriticPoint2Vec] Warning: Checkpoint path does not exist: {resolved_path}")
                print(f"[ActorCriticPoint2Vec] Using random initialization instead")
        else:
            print(f"[ActorCriticPoint2Vec] No pretrained checkpoint provided, using random initialization")

        # Freeze encoder if specified
        if freeze_point2vec:
            for p in self.tokenizer.parameters():
                p.requires_grad = False
            for p in self.positional_encoding.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.tokenizer.eval()
            self.positional_encoding.eval()
            self.encoder.eval()
            print(f"[ActorCriticPoint2Vec] Point2Vec encoder frozen")

        # Encoder output dimension
        self.encoder_feat_dim = encoder_dim
        print(f"[ActorCriticPoint2Vec] Encoder output dimension: {self.encoder_feat_dim}")

        # ----------------------------------------------------------------------
        # Feature fusion setup (StateDependentCrossFeatNet or simple concat)
        # ----------------------------------------------------------------------
        # Calculate aggregated feature dimension
        point2vec_feature_dim = 0
        if self.use_max_pooling:
            point2vec_feature_dim += self.encoder_feat_dim
        if self.use_mean_pooling:
            point2vec_feature_dim += self.encoder_feat_dim
        
        if point2vec_feature_dim == 0:
            raise ValueError("At least one of use_max_pooling or use_mean_pooling must be True")
        
        if self.use_sd_cross:
            # Use StateDependentCrossFeatNet for feature fusion
            print(f"[ActorCriticPoint2Vec] Using StateDependentCrossFeatNet for fusion")
            
            sd_cfg = StateDependentCrossFeatNet.Config(
                dim_in=(1, point2vec_feature_dim),  # Aggregated features: [B, 1, D]
                dim_out=sd_emb_dim,
                query_keys=("rest",),  # Query from extra state
                num_query=sd_num_query,
                ctx_dim=self.extra_state_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=sd_cat_ctx,
            )
            
            self.state_cross = StateDependentCrossFeatNet(sd_cfg)
            
            # Calculate sd_cross output dimension
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += self.extra_state_dim
            
            fusion_input_dim = sd_out_dim
            
            print(f"  - SD num_query: {sd_num_query}")
            print(f"  - SD emb_dim: {sd_emb_dim}")
            print(f"  - SD cat_query: {sd_cat_query}")
            print(f"  - SD cat_ctx: {sd_cat_ctx}")
            print(f"  - SD output dim: {sd_out_dim}")
        else:
            # Simple concatenation
            fusion_input_dim = point2vec_feature_dim + self.extra_state_dim
            print(f"[ActorCriticPoint2Vec] Using simple concatenation for fusion")
            print(f"  - Point2Vec feature dim: {point2vec_feature_dim}")
            print(f"  - Extra state dim: {self.extra_state_dim}")
            print(f"  - Fusion input dim: {fusion_input_dim}")

        # Build fusion MLP
        self.fusion_mlp = self._build_fusion_mlp(
            fusion_input_dim, fusion_hidden_dims, activation_fn
        )
        
        if fusion_hidden_dims:
            fusion_out_dim = int(fusion_hidden_dims[-1])
        else:
            fusion_out_dim = int(fusion_input_dim)

        # Build Actor and Critic heads
        self.actor = self._build_mlp(
            fusion_out_dim, actor_hidden_dims, activation_fn, num_actions
        )
        self.critic = self._build_mlp(
            fusion_out_dim, critic_hidden_dims, activation_fn, 1
        )

        print(f"[ActorCriticPoint2Vec] Network dimensions:")
        print(f"  - Fusion input: {fusion_input_dim}")
        print(f"  - Fusion output: {fusion_out_dim}")
        print(f"  - Actor hidden: {actor_hidden_dims}")
        print(f"  - Critic hidden: {critic_hidden_dims}")
        print(f"  - Actor output: {num_actions}")

        # Action distribution parameters
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticPoint2Vec] Initialization complete")

    # --------------------------------------------------------------------------
    # Utility builders
    # --------------------------------------------------------------------------
    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dims, activation, output_dim: Optional[int] = None
    ):
        """Build MLP with optional output layer."""
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation)
            prev_dim = hidden

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers) if layers else nn.Identity()

    @staticmethod
    def _build_fusion_mlp(input_dim: int, hidden_dims, activation):
        """Build fusion MLP (no output layer)."""
        if not hidden_dims:
            return nn.Identity()
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation)
            prev_dim = hidden
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------------
    # Checkpoint loading
    # --------------------------------------------------------------------------
    def _load_pretrained_checkpoint(self, ckpt_path: str):
        """Load pretrained Point2Vec checkpoint."""
        checkpoint = extract_model_checkpoint(ckpt_path)
        
        # Filter checkpoint keys to only include encoder components
        model_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("tokenizer."):
                model_dict[k] = v
            elif k.startswith("positional_encoding."):
                model_dict[k] = v
            elif k.startswith("encoder."):
                model_dict[k] = v
        
        missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
        print(f"[ActorCriticPoint2Vec] Loaded checkpoint:")
        print(f"  - Missing keys: {len(missing_keys)} keys")
        print(f"  - Unexpected keys: {len(unexpected_keys)} keys")
        if missing_keys:
            print(f"    Missing: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"    Missing: {missing_keys}")
        if unexpected_keys:
            print(f"    Unexpected: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"    Unexpected: {unexpected_keys}")

    # --------------------------------------------------------------------------
    # Observation parsing
    # --------------------------------------------------------------------------
    def _split_observations(self, obs: torch.Tensor):
        """
        Split observations into point cloud and extra state.
        
        Point cloud layout: [object_cloud, obstacle_clouds, ee_cloud]
        Each point has self.point_dim features (3 for coordinates)
        """
        batch_size = obs.shape[0]

        # Extract point clouds
        offset = 0
        pc_parts = []

        # Object cloud
        object_pc = obs[:, offset : offset + self.num_points * self.point_dim]
        pc_parts.append(object_pc.view(batch_size, self.num_points, self.point_dim))
        offset += self.num_points * self.point_dim

        # Obstacle clouds
        if self.num_obstacles > 0:
            obstacle_dim = self.num_obstacles * self.num_points * self.point_dim
            obstacle_pc = obs[:, offset : offset + obstacle_dim]
            pc_parts.append(
                obstacle_pc.view(
                    batch_size, self.num_obstacles * self.num_points, self.point_dim
                )
            )
            offset += obstacle_dim

        # Concatenate to unified point cloud
        pointcloud = torch.cat(pc_parts, dim=1)  # [B, total_points, point_dim]

        # Extract extra state (all remaining observations)
        extra_state = obs[:, offset:]  # [B, extra_state_dim]

        return pointcloud, extra_state

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------
    def _encode_with_point2vec(self, pointcloud: torch.Tensor, is_training: bool = True):
        """
        Encode point clouds with Point2Vec and extract features.
        
        Args:
            pointcloud: [B, N, 3] point cloud coordinates
            is_training: Whether in training mode (affects data transformations)
            
        Returns:
            features: [B, T, feat_dim] encoded token features
        """
        # Apply data transformations
        if is_training:
            pointcloud = self.train_transformations(pointcloud)
        else:
            pointcloud = self.val_transformations(pointcloud)
        
        # Tokenize: convert point cloud to tokens
        tokens, centers = self.tokenizer(pointcloud)  # (B, T, C), (B, T, 3)
        
        # Positional encoding
        pos_embeddings = self.positional_encoding(centers)  # (B, T, C)
        
        # Encode with Transformer
        with torch.set_grad_enabled(not self.freeze_point2vec):
            output = self.encoder(
                tokens, pos_embeddings, return_hidden_states=False
            )
        
        # Extract features from last hidden state
        features = output.last_hidden_state  # (B, T, feat_dim)
        
        return features

    def _get_features(self, observations: torch.Tensor):
        """
        Extract features from observations.
        
        Pipeline:
            1. Split observations into point cloud and extra state
            2. Encode with Point2Vec
            3. Aggregate features (max/mean pooling)
            4. StateDependentCrossFeatNet fusion (or simple concat)
            5. Fusion MLP
            
        Returns:
            fused_features: [B, fusion_out_dim]
        """
        # Split observations
        pointcloud, extra_state = self._split_observations(observations)
        
        # Encode with Point2Vec
        is_training = self.training and not self.freeze_point2vec
        encoder_features = self._encode_with_point2vec(pointcloud, is_training=is_training)  # [B, T, D]
        
        # Aggregate features: max and/or mean pooling
        aggregated_features = []
        if self.use_max_pooling:
            max_features = encoder_features.max(dim=1).values  # [B, D]
            aggregated_features.append(max_features)
        if self.use_mean_pooling:
            mean_features = encoder_features.mean(dim=1)  # [B, D]
            aggregated_features.append(mean_features)
        
        point2vec_features = torch.cat(aggregated_features, dim=-1)  # [B, aggregated_D]
        
        # Prepare for sd_cross or concat
        if self.use_sd_cross:
            # SD-Cross expects [B, seq_len, D], so unsqueeze to [B, 1, D]
            x = point2vec_features.unsqueeze(1)  # [B, 1, D]
            
            # Build context for sd_cross
            sd_ctx = {'rest': extra_state}
            
            # Apply StateDependentCrossFeatNet
            base_features = self.state_cross(x, ctx=sd_ctx)
            
            # Apply fusion MLP
            fused_features = self.fusion_mlp(base_features)
        else:
            # Simple concatenation
            fusion_input = torch.cat([point2vec_features, extra_state], dim=-1)
            
            # Apply fusion MLP
            fused_features = self.fusion_mlp(fusion_input)
        
        return fused_features

    # --------------------------------------------------------------------------
    # Actor / Critic interface
    # --------------------------------------------------------------------------
    def update_distribution(self, observations: torch.Tensor):
        """Update action distribution based on observations."""
        features = self._get_features(observations)
        mean = self.actor(features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        """Sample action from current policy."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        """Deterministic action (mean) for inference."""
        features = self._get_features(observations)
        return self.actor(features)

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        """Evaluate value function."""
        features = self._get_features(critic_observations)
        return self.critic(features)

    def reset(self, dones=None):
        """Stateless policy; nothing to reset."""
        pass

    def train(self, mode=True):
        """
        Set training mode, but keep Point2Vec encoder in eval mode if frozen.
        """
        super().train(mode)
        
        # Keep Point2Vec encoder in eval mode if it's frozen
        if self.freeze_point2vec:
            self.tokenizer.eval()
            self.positional_encoding.eval()
            self.encoder.eval()
        
        return self

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

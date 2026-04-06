# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic network using Concerto point cloud encoder with StateDependentCrossFeatNet fusion."""

from __future__ import annotations

from typing import Any, Dict, Optional
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet

import concerto


class ActorCriticConcerto(nn.Module):
    """
    Actor-Critic network using Concerto as point cloud encoder.
    
    Architecture:
        1. Point cloud encoding with Concerto
        2. Mean pooling to aggregate features
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
        point_dim: int = 3,  # Concerto uses coord (3) + color (3), can add normal (3) -> 9
        num_points: int = 512,
        num_obstacles: int = 1,
        # Concerto settings
        concerto_model: str = "concerto_large",
        # concerto_base concerto_small concerto_tiny
        concerto_repo_id: str = "Pointcept/Concerto",
        concerto_ckpt: Optional[str] = None,
        freeze_concerto: bool = True,
        enable_flash: bool = False,
        enc_patch_size: Optional[list] = None,
        use_color: Optional[bool] = None,  # Auto-infer from point_dim if None
        use_normal: Optional[bool] = None,  # Auto-infer from point_dim if None
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
                f"ActorCriticConcerto.__init__ got unexpected arguments (ignored): "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if concerto is None:
            raise ImportError(
                "concerto is not installed. Please install it to use ActorCriticConcerto."
            )

        self.point_dim = point_dim
        self.num_points = num_points
        self.num_obstacles = num_obstacles
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type
        self.freeze_concerto = freeze_concerto
        self.use_sd_cross = use_sd_cross

        # Auto-infer use_color and use_normal from point_dim if not specified
        # point_dim can be: 3 (coord only), 6 (coord + color), 9 (coord + color + normal)
        if use_color is None:
            # Auto-infer: if point_dim >= 6, assume color is used
            use_color = (point_dim >= 6)
        if use_normal is None:
            # Auto-infer: if point_dim >= 9, assume normal is used
            use_normal = (point_dim >= 9)
        
        self.use_color = use_color
        self.use_normal = use_normal

        # Validate point_dim matches expected dimension
        expected_dim = 3  # coord
        if use_color:
            expected_dim += 3
        if use_normal:
            expected_dim += 3
        
        if point_dim != expected_dim:
            print(
                f"[ActorCriticConcerto] Warning: point_dim={point_dim} doesn't match expected dimension {expected_dim} "
                f"(coord=3, color={'3' if use_color else '0'}, normal={'3' if use_normal else '0'}). "
                f"Using point_dim={point_dim} from input. "
                f"Note: Concerto supports input without color/normal by setting them to zeros."
            )
        
        self.input_feat_dim = point_dim

        # Calculate observation layout
        object_pc_dim = self.num_points * self.point_dim
        obstacle_pc_dim = num_obstacles * self.num_points * self.point_dim
        self.pc_dim = object_pc_dim + obstacle_pc_dim

        print(f"[ActorCriticConcerto] Point cloud layout:")
        print(f"  - Object points: {self.num_points} (dim: {object_pc_dim})")
        print(
            f"  - Obstacle points: {num_obstacles} x {self.num_points} = {num_obstacles * self.num_points} (dim: {obstacle_pc_dim})"
        )
        print(f"  - Total point cloud dim: {self.pc_dim}")
        print(f"  - Input feature dim per point: {self.input_feat_dim}")
        print(f"  - Using color: {self.use_color}, Using normal: {self.use_normal}")
        print(f"  - Note: Concerto supports input without color/normal (they will be set to zeros)")

        # Extra state dimension
        self.extra_state_dim = num_actor_obs - self.pc_dim
        print(f"  - Extra state dim: {self.extra_state_dim}")

        activation_fn = resolve_nn_activation(activation)

        # ----------------------------------------------------------------------
        # Concerto encoder setup
        # ----------------------------------------------------------------------
        print(f"[ActorCriticConcerto] Loading Concerto model: {concerto_model}")
        
        custom_config = {}
        if enc_patch_size is not None:
            custom_config["enc_patch_size"] = enc_patch_size
        else:
            custom_config["enc_patch_size"] = [1024 for _ in range(5)]  # default
        custom_config["enable_flash"] = enable_flash

        ckpt = concerto.load(
            concerto_model,
            repo_id=concerto_repo_id,
            custom_config=custom_config,
            download_root="/mnt/home/zhengyixin/IsaacLab_nonPrehensile/ckpts/concerto",
            ckpt_only=True,
        )
        print(f"[ActorCriticConcerto] Loaded config from {concerto_repo_id}")


        concerto_cfg = ckpt["config"]
        enc_channels = concerto_cfg["enc_channels"]
        
        # Concerto 输出维度：直接使用 enc_channels[-1]
        self.encoder_feat_dim = int(enc_channels[-1])
        self.patch_token_dim = self.encoder_feat_dim
        
        print(
            f"[ActorCriticConcerto] Concerto output dimension: {self.encoder_feat_dim}"
        )

        # Now load the full model
        if concerto_ckpt is not None and os.path.exists(concerto_ckpt):
            # Use checkpoint path directly as name
            self.concerto_model = concerto.load(
                concerto_ckpt,  # Use checkpoint path directly as name
                custom_config=custom_config,
            )
            print(f"[ActorCriticConcerto] Loaded Concerto model from {concerto_ckpt}")
        else:
            self.concerto_model = concerto.load(
                concerto_model,
                repo_id=concerto_repo_id,
                custom_config=custom_config,
            )
            print(f"[ActorCriticConcerto] Loaded Concerto model from {concerto_repo_id}")

        self.concerto_model = self.concerto_model.cpu()

        # Freeze encoder if specified
        if freeze_concerto:
            for p in self.concerto_model.parameters():
                p.requires_grad = False
            self.concerto_model.eval()
            print(f"[ActorCriticConcerto] Concerto encoder frozen")

        # Load transform pipeline
        self.transform = concerto.transform.default()
        print(f"[ActorCriticConcerto] Loaded Concerto transform pipeline")

        # ----------------------------------------------------------------------
        # Feature fusion setup (StateDependentCrossFeatNet or simple concat)
        # ----------------------------------------------------------------------
        concerto_feature_dim = self.encoder_feat_dim
        
        if self.use_sd_cross:
            # Use StateDependentCrossFeatNet for feature fusion
            print(f"[ActorCriticConcerto] Using StateDependentCrossFeatNet for fusion")
            
            sd_cfg = StateDependentCrossFeatNet.Config(
                dim_in=(1, concerto_feature_dim),  # Concerto features after mean pooling: [B, 1, D]
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
            fusion_input_dim = concerto_feature_dim + self.extra_state_dim
            print(f"[ActorCriticConcerto] Using simple concatenation for fusion")
            print(f"  - Concerto feature dim: {concerto_feature_dim}")
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

        print(f"[ActorCriticConcerto] Network dimensions:")
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

        print(f"[ActorCriticConcerto] Initialization complete")

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
    # Observation parsing
    # --------------------------------------------------------------------------
    def _split_observations(self, obs: torch.Tensor):
        """
        Split observations into point cloud and extra state.
        
        Point cloud layout: [object_cloud, obstacle_clouds]
        Each point has self.point_dim features
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

    def _prepare_concerto_input(self, pointcloud: torch.Tensor):
        """
        Prepare point cloud data for Concerto model.
        
        Args:
            pointcloud: [B, N, point_dim] where point_dim can be:
                - 3: coord only
                - 6: coord + color
                - 9: coord + color + normal
            
        Returns:
            List of point dictionaries ready for Concerto
            
        Note: Concerto supports input without color/normal by setting them to zeros.
        This is the recommended approach as stated in Concerto's documentation.
        """
        batch_size = pointcloud.shape[0]
        point_dicts = []

        for i in range(batch_size):
            pc = pointcloud[i]  # [N, point_dim]
            
            # Extract coord (always first 3 dimensions)
            coord = pc[:, :3].cpu().numpy()
            
            # Extract color and normal based on point_dim and configuration
            # If point_dim < 6, no color in input; if point_dim < 9, no normal in input
            offset = 3
            
            if self.use_color and pc.shape[1] >= 6:
                # Color is available in input
                color = pc[:, offset:offset+3].cpu().numpy()
                offset += 3
            else:
                # No color in input or not using color - set to zeros (Concerto supports this)
                color = np.zeros_like(coord)
            
            if self.use_normal and pc.shape[1] >= 9:
                # Normal is available in input
                normal = pc[:, offset:offset+3].cpu().numpy()
                offset += 3
            else:
                # No normal in input or not using normal - set to zeros (Concerto supports this)
                normal = np.zeros_like(coord)
            
            point_dict = {
                "coord": coord,
                "color": color,
                "normal": normal,
            }
            
            # Apply transform
            point_dict = self.transform(point_dict)
            
            point_dicts.append(point_dict)

        return point_dicts

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------
    def _encode_with_concerto(self, point_dicts, device=None):
        """
        Encode point clouds with Concerto and extract features.
        
        Args:
            point_dicts: List of point dictionaries
            device: Target device for tensors (if None, will try to infer from point_dicts or model)
            
        Returns:
            features: [B, N_patches, feat_dim] encoded features
            masks: [B, N_patches] padding masks
        """
        batch_features = []
        
        # Get device if not provided (use model device as default)
        if device is None:
            device = next(self.concerto_model.parameters()).device
        
        # Move tensors to device
        for point_dict in point_dicts:
            for key, value in point_dict.items():
                if isinstance(value, torch.Tensor):
                    point_dict[key] = value.to(device)
            
            # Forward through Concerto
            with torch.set_grad_enabled(not self.freeze_concerto):
                point = self.concerto_model(point_dict)
            
            # Extract features directly from Concerto output (no upsampling)
            feat = point.feat  # [N_downsampled, feat_dim]
            batch_features.append(feat)
        
        # Stack features (note: features may have different lengths due to downsampling)
        # Pad to the maximum length
        max_len = max(f.shape[0] for f in batch_features)
        feat_dim = batch_features[0].shape[1]
        
        padded_features = torch.zeros(
            len(batch_features), max_len, feat_dim,
            device=batch_features[0].device,
            dtype=batch_features[0].dtype
        )
        masks = torch.ones(len(batch_features), max_len, dtype=torch.bool, device=batch_features[0].device)
        
        for i, feat in enumerate(batch_features):
            length = feat.shape[0]
            padded_features[i, :length] = feat
            masks[i, :length] = False  # False means not masked (valid)
        
        return padded_features, masks

    def _get_features(self, observations: torch.Tensor):
        """
        Extract features from observations.
        
        Pipeline:
            1. Split observations into point cloud and extra state
            2. Prepare Concerto input
            3. Encode with Concerto
            4. Mean pooling to aggregate features
            5. StateDependentCrossFeatNet fusion (or simple concat)
            6. Fusion MLP
            
        Returns:
            fused_features: [B, fusion_out_dim]
        """
        # Split observations
        pointcloud, extra_state = self._split_observations(observations)
        
        # Get device from observations (already on correct device in distributed training)
        device = observations.device
        
        # Prepare Concerto input
        point_dicts = self._prepare_concerto_input(pointcloud)
        
        # Encode with Concerto
        encoder_tokens, mask = self._encode_with_concerto(point_dicts, device=device)  # [B, N_patches, feat_dim]
        
        # Mean pooling: aggregate over the sequence dimension, considering mask
        # mask: True means masked (invalid), False means valid
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        valid_mask = ~mask_expanded  # [B, N, 1], True means valid
        
        # Set masked positions to 0 and compute mean over valid positions
        masked_tokens = encoder_tokens * valid_mask.float()  # [B, N, D]
        sum_tokens = masked_tokens.sum(dim=1)  # [B, D]
        count_valid = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1], avoid division by zero
        concerto_features = sum_tokens / count_valid  # [B, D]
        
        # Prepare for sd_cross or concat
        if self.use_sd_cross:
            # SD-Cross expects [B, seq_len, D], so unsqueeze to [B, 1, D]
            x = concerto_features.unsqueeze(1)  # [B, 1, D]
            
            # Build context for sd_cross
            sd_ctx = {'rest': extra_state}
            
            # Apply StateDependentCrossFeatNet
            base_features = self.state_cross(x, ctx=sd_ctx)
            
            # Apply fusion MLP
            fused_features = self.fusion_mlp(base_features)
        else:
            # Simple concatenation
            fusion_input = torch.cat([concerto_features, extra_state], dim=-1)
            
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
        Set training mode, but keep Concerto encoder in eval mode if frozen.
        """
        super().train(mode)
        
        # Keep Concerto encoder in eval mode if it's frozen
        if hasattr(self, 'concerto_model') and self.freeze_concerto:
            self.concerto_model.eval()
        
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


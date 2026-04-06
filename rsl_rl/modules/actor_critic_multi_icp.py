# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

from rsl_rl.modules.models.rl.net.icp import ICPNet 
from rsl_rl.modules.models.rl.net.sd_cross import StateDependentCrossFeatNet

class ActorCriticMultiICPBase(nn.Module):
    """
    Base class for ActorCritic networks with multiple point cloud inputs.
    
    This class contains all common functionality shared between:
    - ActorCriticMultiICP: hand_state passed to SD-Cross (not ICP)
    - ActorCriticMultiICP_HandState: hand_state passed to ICP encoder
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_obstacles=2,
        num_large_obstacles=0,
        icp_point_dim=3,
        icp_num_points=512,
        icp_weights_path=None,
        freeze_icp=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        fusion_hidden_dims=None,
        fusion_use_norm=True,
        fusion_norm_type="layer",
        actor_use_norm=True,
        actor_norm_type="layer",
        actor_output_activation=False,
        critic_use_norm=True,
        critic_norm_type="layer",
        use_sd_cross: bool = True,
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"{self.__class__.__name__}.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # Save configuration
        self.num_obstacles = num_obstacles
        self.num_large_obstacles = num_large_obstacles
        self.num_small_obstacles = num_obstacles - num_large_obstacles
        self.icp_point_dim = icp_point_dim
        self.icp_num_points = icp_num_points
        
        # Calculate dimensions - subclasses will override nonpc_obs_dim calculation
        object_pc_dim = icp_num_points * icp_point_dim
        obstacles_pc_dim = num_obstacles * icp_num_points * icp_point_dim
        self.nonpc_obs_dim = self._calculate_nonpc_obs_dim(num_actor_obs, object_pc_dim, obstacles_pc_dim)
        
        # Print observation dimensions - subclasses can override
        self._print_observation_dims(object_pc_dim, obstacles_pc_dim, num_actor_obs)
        
        # Initialize ICP encoder (subclass-specific)
        self.icp_encoder = self._create_icp_encoder(icp_weights_path, freeze_icp, "shared_pointcloud")
        
        # Get ICP feature dimension
        icp_feature_dim = self.icp_encoder.cfg.dim_out
        
        # Calculate total number of tokens
        icp_num_patches = self.icp_num_points // self.icp_encoder.cfg.patch_size
        total_num_tokens = (1 + num_obstacles) * icp_num_patches
        
        # Create learnable type tokens
        self.object_type_token = nn.Parameter(torch.zeros(1, 1, icp_feature_dim))
        self.large_obstacle_type_token = nn.Parameter(torch.zeros(1, 1, icp_feature_dim))
        self.small_obstacle_type_token = nn.Parameter(torch.zeros(1, 1, icp_feature_dim))
        
        # Initialize type tokens
        nn.init.normal_(self.object_type_token, std=0.02)
        nn.init.normal_(self.large_obstacle_type_token, std=0.02)
        nn.init.normal_(self.small_obstacle_type_token, std=0.02)
        
        print(f"[{self.__class__.__name__}] Type tokens initialized:")
        print(f"  - Object type token: {self.object_type_token.shape}")
        print(f"  - Large obstacle type token: {self.large_obstacle_type_token.shape}")
        print(f"  - Small obstacle type token: {self.small_obstacle_type_token.shape}")
        
        # Activation function
        activation = resolve_nn_activation(activation)
        
        # Choose fusion backend
        self.use_sd_cross = use_sd_cross
        
        # Set default fusion hidden dimensions
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        
        if self.use_sd_cross:
            if sd_query_keys is None:
                sd_query_keys = ("rest",)
            
            sd_ctx_dim = self.nonpc_obs_dim
            
            # Calculate number of query tokens for each part
            # First 1/2 attend to object only, rest attend to all tokens
            num_query_object = max(1, sd_num_query // 2)  # At least 1 query token for object
            num_query_all = sd_num_query - num_query_object
            
            # Calculate number of object tokens (only object, no obstacles)
            object_num_tokens = icp_num_patches  # Only object tokens
            
            print(f'[{self.__class__.__name__}] icp_feature_dim: {icp_feature_dim}, total_num_tokens: {total_num_tokens}, object_num_tokens: {object_num_tokens}, sd_emb_dim: {sd_emb_dim}')
            print(f'  - Query tokens: {num_query_object} for object-only, {num_query_all} for all tokens')
            
            # StateDependentCrossFeatNet for object-only attention (first 1/4 query tokens)
            if num_query_object > 0:
                sd_cfg_object = StateDependentCrossFeatNet.Config(
                    dim_in=(object_num_tokens, icp_feature_dim),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_object,
                    ctx_dim=sd_ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,  # Don't cat ctx here, will cat at the end
                )
                self.state_cross_encoder_object = StateDependentCrossFeatNet(sd_cfg_object)
            else:
                self.state_cross_encoder_object = None
            
            # StateDependentCrossFeatNet for all tokens attention (remaining query tokens)
            if num_query_all > 0:
                sd_cfg_all = StateDependentCrossFeatNet.Config(
                    dim_in=(total_num_tokens, icp_feature_dim),
                    dim_out=sd_emb_dim,
                    query_keys=tuple(sd_query_keys),
                    num_query=num_query_all,
                    ctx_dim=sd_ctx_dim,
                    emb_dim=sd_emb_dim,
                    cat_query=sd_cat_query,
                    cat_ctx=False,  # Don't cat ctx here, will cat at the end
                )
                self.state_cross_encoder_all = StateDependentCrossFeatNet(sd_cfg_all)
            else:
                self.state_cross_encoder_all = None
            
            # Calculate output dimension
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim
            
            fusion_input_dim = sd_out_dim
            print(f"[{self.__class__.__name__}] Using StateDependentCrossFeatNet for fusion (differentiated attention).")
            print(f"  - SD output dim: {sd_out_dim}")
            print(f"  - Fusion input dim: {fusion_input_dim}")
            
            # Store configuration for later use
            self.num_query_object = num_query_object
            self.num_query_all = num_query_all
            self.object_num_tokens = object_num_tokens
            self.sd_cat_ctx = sd_cat_ctx  # Store for later use in _get_fused_features
        else:
            fusion_input_dim = self.nonpc_obs_dim + (total_num_tokens * icp_feature_dim)
        
        # Build fusion MLP
        self.feature_fusion = self._build_fusion_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            activation=activation,
            use_norm=fusion_use_norm,
            norm_type=fusion_norm_type
        )
        
        mlp_input_dim_a = fusion_hidden_dims[-1]
        mlp_input_dim_c = fusion_hidden_dims[-1]
        
        # Policy network (Actor)
        self.actor = self._build_actor_critic_mlp(
            input_dim=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
            use_norm=actor_use_norm,
            norm_type=actor_norm_type,
            is_actor=True,
            output_activation=actor_output_activation
        )
        
        # Value network (Critic)
        self.critic = self._build_actor_critic_mlp(
            input_dim=mlp_input_dim_c,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
            use_norm=critic_use_norm,
            norm_type=critic_norm_type,
            is_actor=False
        )
        
        print(f"[{self.__class__.__name__}] Network architecture:")
        print(f"  - Shared ICP Encoder: {self.icp_encoder}")
        print(f"  - Feature Fusion MLP: {self.feature_fusion}")
        if self.use_sd_cross:
            if hasattr(self, 'state_cross_encoder_object') and self.state_cross_encoder_object is not None:
                print(f"  - StateDependentCrossFeatNet (object-only): {self.state_cross_encoder_object}")
            if hasattr(self, 'state_cross_encoder_all') and self.state_cross_encoder_all is not None:
                print(f"  - StateDependentCrossFeatNet (all tokens): {self.state_cross_encoder_all}")
        print(f"  - Actor MLP: {self.actor}")
        print(f"  - Critic MLP: {self.critic}")
        print(f"  - ICP feature dim: {icp_feature_dim}")
        print(f"  - Fusion output dim: {fusion_hidden_dims[-1]}")
        print(f"  - Fusion normalization: {fusion_norm_type if fusion_use_norm else 'None'}")
        print(f"  - Actor normalization: {actor_norm_type if actor_use_norm else 'None'}")
        print(f"  - Actor output activation: {'Tanh' if actor_output_activation else 'None'}")
        print(f"  - Critic normalization: {critic_norm_type if critic_use_norm else 'None'}")
        
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        self.distribution = None
        Normal.set_default_validate_args(False)

    # Abstract methods to be implemented by subclasses
    def _calculate_nonpc_obs_dim(self, num_actor_obs, object_pc_dim, obstacles_pc_dim):
        """Calculate non-point-cloud observation dimension. Override in subclasses."""
        raise NotImplementedError
    
    def _print_observation_dims(self, object_pc_dim, obstacles_pc_dim, num_actor_obs):
        """Print observation dimensions. Override in subclasses."""
        raise NotImplementedError
    
    def _create_icp_encoder(self, icp_weights_path, freeze_icp, cloud_type):
        """Create ICP encoder. Override in subclasses to set different keys."""
        raise NotImplementedError
    
    def _split_obs(self, obs):
        """Split observations. Override in subclasses."""
        raise NotImplementedError
    
    def _extract_point_clouds_and_context(self, observations):
        """Extract point clouds and context. Override in subclasses."""
        raise NotImplementedError
    
    def _expand_context_for_batch(self, context, batch_size, num_total_clouds):
        """Expand context for batch processing. Override in subclasses."""
        raise NotImplementedError

    # Common methods shared by all subclasses
    def _build_fusion_mlp(self, input_dim, hidden_dims, activation, use_norm, norm_type):
        """Build fusion MLP with optional normalization."""
        fusion_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_norm and norm_type is not None:
                if norm_type == "layer":
                    fusion_layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch":
                    fusion_layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")
            
            fusion_layers.append(activation)
            prev_dim = hidden_dim
            
        return nn.Sequential(*fusion_layers)

    def _build_actor_critic_mlp(self, input_dim, hidden_dims, output_dim, activation, use_norm, norm_type, is_actor, output_activation=True):
        """Build actor or critic MLP with optional normalization."""
        layers = []
        prev_dim = input_dim
        
        # First layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        if use_norm and norm_type is not None:
            if norm_type == "layer":
                layers.append(nn.LayerNorm(hidden_dims[0]))
            elif norm_type == "batch":
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")
        layers.append(activation)
        prev_dim = hidden_dims[0]
        
        # Hidden layers
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
                if is_actor and output_activation:
                    layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if use_norm and norm_type is not None:
                    if norm_type == "layer":
                        layers.append(nn.LayerNorm(hidden_dims[i + 1]))
                    elif norm_type == "batch":
                        layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                    else:
                        raise ValueError(f"Unknown normalization type: {norm_type}")
                layers.append(activation)
                prev_dim = hidden_dims[i + 1]
        
        return nn.Sequential(*layers)

    def _get_fused_features(self, observations):
        """Get fused features - common implementation with subclass-specific context handling."""
        object_cloud, obstacles_clouds, context, regular_obs = self._extract_point_clouds_and_context(observations)
        
        batch_size = object_cloud.size(0)
        
        # Combine object and obstacles
        all_clouds = torch.cat([object_cloud.unsqueeze(1), obstacles_clouds], dim=1)
        num_total_clouds = 1 + self.num_obstacles
        all_clouds_flat = all_clouds.view(batch_size * num_total_clouds, self.icp_num_points, self.icp_point_dim)
        
        # Expand context for batch processing (subclass-specific)
        expanded_context = self._expand_context_for_batch(context, batch_size, num_total_clouds)
        
        # Forward through ICP encoder
        with torch.no_grad() if not self.icp_encoder.training else torch.enable_grad():
            _, all_icp_feats_flat = self.icp_encoder(all_clouds_flat, expanded_context)
        
        # Reshape and add type tokens
        num_patches = all_icp_feats_flat.size(1)
        icp_feature_dim = all_icp_feats_flat.size(2)
        all_icp_feats_per_cloud = all_icp_feats_flat.view(batch_size, num_total_clouds, num_patches, icp_feature_dim)
        
        # Add type tokens
        all_icp_feats_per_cloud[:, 0, :, :] += self.object_type_token
        if self.num_large_obstacles > 0:
            all_icp_feats_per_cloud[:, 1:1+self.num_large_obstacles, :, :] += self.large_obstacle_type_token
        if self.num_small_obstacles > 0:
            all_icp_feats_per_cloud[:, 1+self.num_large_obstacles:, :, :] += self.small_obstacle_type_token
        
        # Merge tokens
        all_icp_feats = all_icp_feats_per_cloud.view(batch_size, num_total_clouds * num_patches, icp_feature_dim)
        
        if self.use_sd_cross:
            sd_ctx = {'rest': regular_obs}
            
            # Extract object-only tokens (first cloud)
            object_icp_feats = all_icp_feats_per_cloud[:, 0, :, :]  # [B, num_patches, icp_feature_dim]
            
            # Differentiated attention: first 1/4 query tokens attend to object only, rest attend to all
            base_features_parts = []
            
            # First 1/4 query tokens attend to object-only tokens
            if self.num_query_object > 0 and self.state_cross_encoder_object is not None:
                object_features = self.state_cross_encoder_object(object_icp_feats, ctx=sd_ctx)
                base_features_parts.append(object_features)
            
            # Remaining query tokens attend to all tokens
            if self.num_query_all > 0 and self.state_cross_encoder_all is not None:
                all_features = self.state_cross_encoder_all(all_icp_feats, ctx=sd_ctx)
                base_features_parts.append(all_features)
            
            # Concatenate all features
            base_features = torch.cat(base_features_parts, dim=-1) if base_features_parts else torch.empty(batch_size, 0, device=all_icp_feats.device)
            
            # base_features already contains regular_obs if sd_cat_ctx=True (handled inside StateDependentCrossFeatNet)
            # Note: In this implementation, we set cat_ctx=False in StateDependentCrossFeatNet config,
            # so we manually add it here if sd_cat_ctx=True
            if hasattr(self, 'sd_cat_ctx') and self.sd_cat_ctx:
                base_features = torch.cat([base_features, regular_obs], dim=-1)
            
            fused_features = self.feature_fusion(base_features)
            return fused_features
        else:
            all_icp_feats_flat_concat = all_icp_feats.flatten(start_dim=1)
            raw_features = torch.cat([regular_obs, all_icp_feats_flat_concat], dim=-1)
            fused_features = self.feature_fusion(raw_features)
            return fused_features

    def _get_actor_features(self, observations):
        return self._get_fused_features(observations)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        features = self._get_actor_features(observations)
        mean = self.actor(features)
        
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        if torch.any(std < 0):
            print(f"[ERROR] Negative std detected!")
            print(f"  std min: {std.min().item():.6f}, max: {std.max().item():.6f}")
            print(f"  mean min: {mean.min().item():.6f}, max: {mean.max().item():.6f}")
            print(f"  noise_std_type: {self.noise_std_type}")
            if self.noise_std_type == "scalar":
                print(f"  self.std min: {self.std.min().item():.6f}, max: {self.std.max().item():.6f}")
            elif self.noise_std_type == "log":
                print(f"  self.log_std min: {self.log_std.min().item():.6f}, max: {self.log_std.max().item():.6f}")
                print(f"  exp(log_std) min: {torch.exp(self.log_std).min().item():.6f}, max: {torch.exp(self.log_std).max().item():.6f}")
            raise RuntimeError("Negative standard deviation detected!")
        
        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        fused_features = self._get_fused_features(observations)
        actions_mean = self.actor(fused_features)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        fused_features = self._get_fused_features(critic_observations)
        value = self.critic(fused_features)
        return value

    def train(self, mode=True):
        super().train(mode)
        frozen = not any(param.requires_grad for param in self.icp_encoder.parameters())
        if frozen:
            self.icp_encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class ActorCriticMultiICP(ActorCriticMultiICPBase):
    """
    ActorCritic network with multiple point clouds.
    hand_state is passed to SD-Cross (not ICP encoder).
    """
    
    def _calculate_nonpc_obs_dim(self, num_actor_obs, object_pc_dim, obstacles_pc_dim):
        return num_actor_obs - object_pc_dim - obstacles_pc_dim
    
    def _print_observation_dims(self, object_pc_dim, obstacles_pc_dim, num_actor_obs):
        print(f"[ActorCriticMultiICP] Observation dimensions:")
        print(f"  - Object point cloud: {object_pc_dim} ({self.icp_num_points} points × {self.icp_point_dim}D)")
        print(f"  - Obstacles point cloud: {obstacles_pc_dim} ({self.num_obstacles} obstacles × {self.icp_num_points} points × {self.icp_point_dim}D)")
        print(f"  - Regular observations (including hand_state): {self.nonpc_obs_dim}")
        print(f"  - Total: {num_actor_obs}")
    
    def _create_icp_encoder(self, icp_weights_path, freeze_icp, cloud_type):
        """Create ICP encoder without hand_state context."""
        default_cfg = ICPNet.Config(
            dim_in=(self.icp_num_points, self.icp_point_dim),
            dim_out=128,
            keys={},  # No hand_state context
            headers=['collision'],
            num_query=1,
            patch_size=32,
            encoder_channel=128,
            pos_embed_type='mlp',
            group_type='fps',
            patch_type='mlp',
            patch_overlap=1.0,
            p_drop=0.0,
            freeze_encoder=False,
            use_adapter=False,
            adapter_dim=64,
            tune_last_layer=False,
            late_late_fusion=False,
            output_attn=False,
            output_hidden=False,
            activate_header=False,
            pre_ln_bias=True,
            ignore_zero=False,
            use_vq=False,
            train_last_ln=True,
            header_inputs=None,
            use_v2_module=False
        )
        
        default_cfg.encoder.num_hidden_layers = 2
        default_cfg.encoder.layer.hidden_size = 128
        default_cfg.encoder.layer.num_attention_heads = 3
        
        icp_encoder = ICPNet(default_cfg)
        
        if icp_weights_path is not None:
            print(f"[ActorCriticMultiICP] Loading ICP weights for {cloud_type} from: {icp_weights_path}")
            icp_encoder.load(filename=icp_weights_path, verbose=True)
            print(f"  [{cloud_type}] ICP Successfully loaded all weights using ICPNet.load()!")
    
        if freeze_icp:
            for p in icp_encoder.parameters():
                p.requires_grad = False
            icp_encoder.eval()
            print(f"  [{cloud_type}] ICP encoder frozen and set to eval mode")
        
        return icp_encoder
    
    def _split_obs(self, obs):
        """Split observations - hand_state included in regular_obs."""
        object_pc_dim = self.icp_num_points * self.icp_point_dim
        obstacles_pc_dim = self.num_obstacles * self.icp_num_points * self.icp_point_dim
        
        object_cloud_flat = obs[:, :object_pc_dim]
        object_cloud = object_cloud_flat.view(-1, self.icp_num_points, self.icp_point_dim)
        
        obstacles_start = object_pc_dim
        obstacles_end = obstacles_start + obstacles_pc_dim
        obstacles_cloud_flat = obs[:, obstacles_start:obstacles_end]
        obstacles_clouds = obstacles_cloud_flat.view(
            -1, self.num_obstacles, self.icp_num_points, self.icp_point_dim
        )
        
        regular_obs = obs[:, obstacles_end:]
        
        return object_cloud, obstacles_clouds, regular_obs
    
    def _extract_point_clouds_and_context(self, observations):
        """Extract point clouds - no context for ICP."""
        if isinstance(observations, torch.Tensor):
            object_cloud, obstacles_clouds, regular_obs = self._split_obs(observations)
            context = {}
        else:
            raise ValueError("observations must be a tensor for concatenated mode")
        
        return object_cloud, obstacles_clouds, context, regular_obs
    
    def _expand_context_for_batch(self, context, batch_size, num_total_clouds):
        """No expansion needed - context is empty."""
        return context


class ActorCriticMultiICP_HandState(ActorCriticMultiICPBase):
    """
    ActorCritic network with multiple point clouds.
    hand_state is passed to ICP encoder (not SD-Cross).
    """
    
    def _calculate_nonpc_obs_dim(self, num_actor_obs, object_pc_dim, obstacles_pc_dim):
        return num_actor_obs - object_pc_dim - obstacles_pc_dim - 9  # 9 for hand_state
    
    def _print_observation_dims(self, object_pc_dim, obstacles_pc_dim, num_actor_obs):
        print(f"[ActorCriticMultiICP_HandState] Observation dimensions:")
        print(f"  - Object point cloud: {object_pc_dim} ({self.icp_num_points} points × {self.icp_point_dim}D)")
        print(f"  - Obstacles point cloud: {obstacles_pc_dim} ({self.num_obstacles} obstacles × {self.icp_num_points} points × {self.icp_point_dim}D)")
        print(f"  - Hand state: 9D (passed to ICP encoder)")
        print(f"  - Regular observations (excluding hand_state): {self.nonpc_obs_dim}")
        print(f"  - Total: {num_actor_obs}")
    
    def _create_icp_encoder(self, icp_weights_path, freeze_icp, cloud_type):
        """Create ICP encoder with hand_state context."""
        default_cfg = ICPNet.Config(
            dim_in=(self.icp_num_points, self.icp_point_dim),
            dim_out=128,
            keys={'hand_state': 9},  # hand_state as context
            headers=['collision'],
            num_query=1,
            patch_size=32,
            encoder_channel=128,
            pos_embed_type='mlp',
            group_type='fps',
            patch_type='mlp',
            patch_overlap=1.0,
            p_drop=0.0,
            freeze_encoder=False,
            use_adapter=False,
            adapter_dim=64,
            tune_last_layer=False,
            late_late_fusion=False,
            output_attn=False,
            output_hidden=False,
            activate_header=False,
            pre_ln_bias=True,
            ignore_zero=False,
            use_vq=False,
            train_last_ln=True,
            header_inputs=None,
            use_v2_module=False
        )
        
        default_cfg.encoder.num_hidden_layers = 2
        default_cfg.encoder.layer.hidden_size = 128
        default_cfg.encoder.layer.num_attention_heads = 3
        
        icp_encoder = ICPNet(default_cfg)
        
        if icp_weights_path is not None:
            print(f"[ActorCriticMultiICP] Loading ICP weights for {cloud_type} from: {icp_weights_path}")
            icp_encoder.load(filename=icp_weights_path, verbose=True)
            print(f"  [{cloud_type}] ICP Successfully loaded all weights using ICPNet.load()!")
    
        if freeze_icp:
            for p in icp_encoder.parameters():
                p.requires_grad = False
            icp_encoder.eval()
            print(f"  [{cloud_type}] ICP encoder frozen and set to eval mode")
        
        return icp_encoder
    
    def _split_obs(self, obs):
        """Split observations - hand_state separated from regular_obs."""
        object_pc_dim = self.icp_num_points * self.icp_point_dim
        obstacles_pc_dim = self.num_obstacles * self.icp_num_points * self.icp_point_dim
        
        object_cloud_flat = obs[:, :object_pc_dim]
        object_cloud = object_cloud_flat.view(-1, self.icp_num_points, self.icp_point_dim)
        
        obstacles_start = object_pc_dim
        obstacles_end = obstacles_start + obstacles_pc_dim
        obstacles_cloud_flat = obs[:, obstacles_start:obstacles_end]
        obstacles_clouds = obstacles_cloud_flat.view(
            -1, self.num_obstacles, self.icp_num_points, self.icp_point_dim
        )
        
        hand_state_start = obstacles_end
        hand_state_end = hand_state_start + 9
        hand_state = obs[:, hand_state_start:hand_state_end]
        
        regular_obs = obs[:, hand_state_end:]
        
        return object_cloud, obstacles_clouds, hand_state, regular_obs
    
    def _extract_point_clouds_and_context(self, observations):
        """Extract point clouds - hand_state in context for ICP."""
        if isinstance(observations, torch.Tensor):
            object_cloud, obstacles_clouds, hand_state, regular_obs = self._split_obs(observations)
            context = {'hand_state': hand_state}
        else:
            raise ValueError("observations must be a tensor for concatenated mode")
        
        return object_cloud, obstacles_clouds, context, regular_obs
    
    def _expand_context_for_batch(self, context, batch_size, num_total_clouds):
        """Expand hand_state context for all point clouds."""
        expanded_context = {}
        if 'hand_state' in context:
            hand_state = context['hand_state']  # [B, 9]
            expanded_hand_state = hand_state.unsqueeze(1).expand(-1, num_total_clouds, -1)  # [B, num_clouds, 9]
            expanded_hand_state = expanded_hand_state.reshape(batch_size * num_total_clouds, -1)  # [B*num_clouds, 9]
            expanded_context['hand_state'] = expanded_hand_state
        return expanded_context 
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

class ActorCriticICP(nn.Module):
    """
    ActorCritic network integrated with ICPNet as point cloud encoder
    
    Input observations contain:
    - Point cloud data (point_cloud)
    - Context information (context) - such as hand states, etc.
    - Other regular observations (proprioceptive, etc.)
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,          # Actor network total input dimension (including point cloud)
        num_critic_obs,         # Critic network input dimension  
        num_actions,            # Action dimension
        icp_point_dim=3,        # Point cloud feature dimension per point
        icp_num_points=512,     # Number of points in point cloud
        icp_weights_path=None,  # Path to pretrained ICP weights
        freeze_icp=True,        # Whether to freeze ICP parameters
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        fusion_hidden_dims=None,  # Feature fusion MLP hidden dimensions (applied after SD-cross or concat)
        fusion_use_norm=True,     # Whether to use normalization in fusion MLP
        fusion_norm_type="layer", # Normalization type: "layer", "batch", or None
        actor_use_norm=True,      # Whether to use normalization in actor MLP
        actor_norm_type="batch",  # Normalization type for actor: "layer", "batch", or None
        actor_output_activation=False,  # Whether to use activation function on actor output (tanh)
        critic_use_norm=False,     # Whether to use normalization in critic MLP
        critic_norm_type=None, # Normalization type for critic: "layer", "batch", or None
        # StateDependentCrossFeatNet settings
        use_sd_cross: bool = True,  # default True; no need to change in most cases
        sd_num_query: int = 16,
        sd_emb_dim: int = 128,
        sd_cat_query: bool = False,
        sd_cat_ctx: bool = True,
        sd_query_keys=None,  # default ("hand_state", "rest") if None
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticICP.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # Save configuration
        self.icp_point_dim = icp_point_dim
        self.icp_num_points = icp_num_points
        
        # Calculate dimensions
        # The rest obs (non-pointcloud, excluding hand_state which goes to context)
        self.nonpc_obs_dim = num_actor_obs - (icp_point_dim * icp_num_points) - 9
        
        # Initialize ICPNet point cloud encoder
        if icp_weights_path is not None:
            # Create ICPNet with default configuration and load pretrained weights
            print(f"Loading ICP from pretrained weights: {icp_weights_path}")
            
            # Configuration based on pretrained weights analysis
            default_cfg = ICPNet.Config(
                dim_in=(icp_num_points, icp_point_dim),
                dim_out=128,  # From patch_encoder output dimension
                keys={'hand_state': 9},  # From tokenize weights: hand_state.linear.weight: [128, 9]
                headers=['collision'],   # From header weights: collision header exists
                num_query=1,
                patch_size=32,  # From patch_encoder input: 96/3 = 32
                encoder_channel=128,  # From encoder weights: layer.0.attention.Wqkv.weight: [384, 128]
                pos_embed_type='mlp',  # From pos_embed weights structure
                group_type='fps',
                patch_type='mlp',  # From patch_encoder weights structure
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
            
            # Set encoder configuration based on weights analysis
            default_cfg.encoder.num_hidden_layers = 2  # From weights: Layer numbers: [0, 1]
            default_cfg.encoder.layer.hidden_size = 128  # Match encoder_channel
            default_cfg.encoder.layer.num_attention_heads = 3  # From weights: 384/128 = 3 heads
            
            self.icp_encoder = ICPNet(default_cfg)
            
            # Load pretrained weights with strict checking
            print("Loading ICP weights with strict=True to ensure configuration match...")
            self.icp_encoder.load(filename=icp_weights_path, verbose=True)
            
            # Verify that all weights were loaded correctly
            print("ICP encoder configuration verification:")
            print(f"  - dim_out: {self.icp_encoder.cfg.dim_out}")
            print(f"  - encoder_channel: {self.icp_encoder.cfg.encoder_channel}")
            print(f"  - num_hidden_layers: {self.icp_encoder.cfg.encoder.num_hidden_layers}")
            print(f"  - hidden_size: {self.icp_encoder.cfg.encoder.layer.hidden_size}")
            print(f"  - num_attention_heads: {self.icp_encoder.cfg.encoder.layer.num_attention_heads}")
            print(f"  - keys: {self.icp_encoder.cfg.keys}")
            print(f"  - headers: {self.icp_encoder.cfg.headers}")
            
            # Freeze parameters (default behavior for pretrained models)
            if freeze_icp:
                print("Freezing ICP encoder parameters...")
                for param in self.icp_encoder.parameters():
                    param.requires_grad = False
                self.icp_encoder.eval()  # Set to evaluation mode
                print("ICP encoder frozen and set to eval mode")
                
            print("ICP encoder loaded from pretrained weights!")
        else:
            # Initialize with default configuration if no pretrained weights
            default_cfg = ICPNet.Config(
                dim_in=(icp_num_points, icp_point_dim),
                dim_out=128,  # Match pretrained weights configuration
                keys={'hand_state': 9},  # Include hand_state for consistency
                headers=['collision'],   # Include collision header for consistency
                num_query=1,
                patch_size=32,
                encoder_channel=128,  # Match pretrained weights configuration
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
            
            # Set encoder configuration to match pretrained weights
            default_cfg.encoder.num_hidden_layers = 2
            default_cfg.encoder.layer.hidden_size = 128
            default_cfg.encoder.layer.num_attention_heads = 3
            
            self.icp_encoder = ICPNet(default_cfg)
            
            # Freeze ICP parameters if specified
            if freeze_icp:
                print("Freezing ICP encoder parameters...")
                for param in self.icp_encoder.parameters():
                    param.requires_grad = False
                self.icp_encoder.eval()  # Set to evaluation mode
                print("ICP encoder frozen and set to eval mode")
        
        # Activation function
        activation = resolve_nn_activation(activation)
        
        # ICPNet output feature dimension - get from actual encoder config or use default
        # Since we have headers (collision), use dim_out which is 128
        icp_feature_dim = self.icp_encoder.cfg.dim_out
        
        print(f"ICP encoder cfg.dim_out: {self.icp_encoder.cfg.dim_out}")
        print(f"ICP encoder cfg.encoder.layer.hidden_size: {self.icp_encoder.cfg.encoder.layer.hidden_size}")
        print(f"Using ICP feature dim: {icp_feature_dim}")
        
        # Choose fusion backend: StateDependentCrossFeatNet or classic concat
        self.use_sd_cross = use_sd_cross
        
        # Set default fusion hidden dimensions
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        
        if self.use_sd_cross:
            # Default query keys: only the remaining observations (rest)
            if sd_query_keys is None:
                sd_query_keys = ("rest",)

            # Context dimension = only remaining observation dim (exclude hand_state)
            sd_ctx_dim = self.nonpc_obs_dim

            # Note: StateDependentCrossFeatNet only uses the last entry of dim_in as embed dim
            sd_cfg = StateDependentCrossFeatNet.Config(
                dim_in=(1, icp_feature_dim),
                dim_out=sd_emb_dim,
                query_keys=tuple(sd_query_keys),
                num_query=sd_num_query,
                ctx_dim=sd_ctx_dim,
                emb_dim=sd_emb_dim,
                cat_query=sd_cat_query,
                cat_ctx=sd_cat_ctx,
            )

            self.state_cross = StateDependentCrossFeatNet(sd_cfg)

            # Compute sd_cross output dim
            sd_out_dim = sd_num_query * sd_emb_dim
            if sd_cat_query:
                sd_out_dim += sd_num_query * sd_emb_dim
            if sd_cat_ctx:
                sd_out_dim += sd_ctx_dim

            fusion_input_dim = sd_out_dim
            print(f"Using StateDependentCrossFeatNet for fusion. SD output dim: {sd_out_dim}, Fusion output dim: {fusion_hidden_dims[-1]}")
        else:
            # Classic concat -> MLP fusion
            fusion_input_dim = self.nonpc_obs_dim + icp_feature_dim

        # Build fusion MLP (shared logic for both fusion types)
        self.feature_fusion = self._build_fusion_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            activation=activation,
            use_norm=fusion_use_norm,
            norm_type=fusion_norm_type
        )

        mlp_input_dim_a = fusion_hidden_dims[-1]
        mlp_input_dim_c = fusion_hidden_dims[-1]
        
        # Policy network (Actor) with optional normalization
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

        # Value network (Critic) with optional normalization
        self.critic = self._build_actor_critic_mlp(
            input_dim=mlp_input_dim_c,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
            use_norm=critic_use_norm,
            norm_type=critic_norm_type,
            is_actor=False
        )

        print(f"Feature Fusion MLP: {self.feature_fusion}")
        if self.use_sd_cross:
            print(f"StateDependentCrossFeatNet: {self.state_cross}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"ICP feature dim: {icp_feature_dim}")
        if not self.use_sd_cross:
            print(f"Concat+Fusion input dim: {fusion_input_dim}")
        print(f"Fusion output dim: {fusion_hidden_dims[-1]}")
        print(f"Fusion normalization: {fusion_norm_type if fusion_use_norm else 'None'}")
        print(f"Actor normalization: {actor_norm_type if actor_use_norm else 'None'}")
        print(f"Actor output activation: {'Tanh' if actor_output_activation else 'None'}")
        print(f"Critic normalization: {critic_norm_type if critic_use_norm else 'None'}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _build_fusion_mlp(self, input_dim, hidden_dims, activation, use_norm, norm_type):
        """
        Build fusion MLP with optional normalization.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_norm: Whether to use normalization
            norm_type: Normalization type ("layer", "batch", or None)
            
        Returns:
            nn.Sequential: Fusion MLP
        """
        fusion_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add normalization if enabled
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
        """
        Build actor or critic MLP with optional normalization.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            use_norm: Whether to use normalization
            norm_type: Normalization type ("layer", "batch", or None)
            is_actor: Whether this is actor (True) or critic (False)
            output_activation: Whether to use activation function on output layer (only for actor)
            
        Returns:
            nn.Sequential: Actor or critic MLP
        """
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
                # Last layer
                layers.append(nn.Linear(hidden_dims[i], output_dim))
                if is_actor and output_activation:
                    layers.append(nn.Tanh())  # Actor uses tanh for bounded actions (optional)
                # Critic doesn't need activation on output
            else:
                # Intermediate layers
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

    def _split_obs(self, obs):
        """
        Split observations into point cloud, hand_state, and regular observations
        Layout: [point_cloud_flat, hand_state(9), other_obs...]
        
        Args:
            obs: (batch, total_obs_dim) - flattened observations
            
        Returns:
            pc: (batch, num_points, point_dim) - point cloud
            hand_state: (batch, 9) - hand state
            rest: (batch, remaining_obs_dim) - remaining regular observations
        """
        # obs: (batch, total_obs_dim)
        pc_flat = obs[:, :self.icp_num_points * self.icp_point_dim]
        pc = pc_flat.view(-1, self.icp_num_points, self.icp_point_dim)
        
        # Extract hand_state (9 dimensions after point cloud)
        hand_state_start = self.icp_num_points * self.icp_point_dim
        hand_state_end = hand_state_start + 9
        hand_state = obs[:, hand_state_start:hand_state_end]
        
        # The rest of observations
        rest = obs[:, hand_state_end:]
        
        return pc, hand_state, rest

    def _extract_point_cloud_and_context(self, observations):
        """
        Extract point cloud and context information from flattened observation tensor
        
        Args:
            observations: Flattened observation tensor (batch, total_obs_dim)
            
        Returns:
            point_cloud: Point cloud tensor (batch, num_points, 3)
            context: Context dictionary with hand_state
            regular_obs: Regular observation tensor
        """
        if isinstance(observations, torch.Tensor):
            # Split flattened observations
            point_cloud, hand_state, regular_obs = self._split_obs(observations)
            context = {'hand_state': hand_state}  # Put hand_state in context for ICP
        else:
            raise ValueError("observations must be a tensor for flattened observation mode")
            
        return point_cloud, context, regular_obs

    def _get_fused_features(self, observations):
        """
        Get fused features for both actor and critic network
        
        Args:
            observations: Observation dictionary
            
        Returns:
            features: Fused feature tensor (batch, fusion_hidden_dim)
        """
        # Extract point cloud and context
        point_cloud, context, regular_obs = self._extract_point_cloud_and_context(observations)
        
        # Encode point cloud using ICPNet (if enabled and available)
        with torch.no_grad() if not self.icp_encoder.training else torch.enable_grad():
            icp_output, icp_features = self.icp_encoder(point_cloud, context)
        
        if self.use_sd_cross:
            # Fuse with StateDependentCrossFeatNet first
            # If icp_features is [B, D], upcast to [B, 1, D]
            if len(icp_features.shape) == 2:
                x = icp_features.unsqueeze(1)
            else:
                x = icp_features

            # Build context for query: only use rest (remaining observations)
            sd_ctx = {
                'rest': regular_obs,
            }

            base_features = self.state_cross(x, ctx=sd_ctx)
            fused_features = self.feature_fusion(base_features)
            return fused_features
        else:
            # Classic MLP fusion: if icp_features is a sequence, mean pool first
            if len(icp_features.shape) > 2:
                icp_features = icp_features.mean(dim=1)

            # Concatenate regular observations and ICP features
            if regular_obs.shape[-1] > 0:
                if icp_features.shape[-1] > 0:
                    raw_features = torch.cat([regular_obs, icp_features], dim=-1)
                else:
                    raw_features = regular_obs
            else:
                # If no regular obs, pad with zeros to match expected input size
                batch_size = icp_features.shape[0] if icp_features.shape[-1] > 0 else point_cloud.shape[0]
                device = icp_features.device if icp_features.shape[-1] > 0 else point_cloud.device
                if icp_features.shape[-1] > 0:
                    zeros = torch.zeros(batch_size, self.feature_fusion[0].in_features - icp_features.shape[-1], 
                                      device=device)
                    raw_features = torch.cat([zeros, icp_features], dim=-1)
                else:
                    raw_features = torch.zeros(batch_size, self.feature_fusion[0].in_features, device=device)

            # Apply feature fusion
            fused_features = self.feature_fusion(raw_features)
            return fused_features

    def _get_actor_features(self, observations):
        """
        Get input features for actor network
        
        Args:
            observations: Observation dictionary
            
        Returns:
            features: Fused feature tensor (batch, feature_dim)
        """
        return self._get_fused_features(observations)

    @staticmethod
    # not used at the moment
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
        """
        Update action distribution
        
        Args:
            observations: Observation dictionary containing point cloud and other info
        """
        # Get fused features
        features = self._get_actor_features(observations)
        
        # Compute mean
        mean = self.actor(features)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """
        Sample actions
        
        Args:
            observations: Observation dictionary
            
        Returns:
            actions: Sampled actions
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """
        Calculate log probabilities of actions
        
        Args:
            actions: Action tensor
            
        Returns:
            log_probs: Log probabilities
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        Inference mode, return deterministic actions (mean)
        
        Args:
            observations: Observation dictionary
            
        Returns:
            actions_mean: Action mean
        """
        fused_features = self._get_fused_features(observations)
        actions_mean = self.actor(fused_features)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluate state value
        
        Args:
            critic_observations: Critic network observation input (should be same format as actor obs)
            
        Returns:
            value: State value
        """
        # Use fused features for critic as well
        fused_features = self._get_fused_features(critic_observations)
        value = self.critic(fused_features)
        return value

    def train(self, mode=True):
        """
        Set training mode, but keep ICP encoder in eval mode if frozen
        """
        super().train(mode)
        
        # Keep ICP encoder in eval mode if it's frozen
        if hasattr(self, 'icp_encoder'):
            frozen = not any(param.requires_grad for param in self.icp_encoder.parameters())
            if frozen:
                self.icp_encoder.eval()
        
        return self

    def load_state_dict(self, state_dict, strict=True):
        """
        Load model parameters
        
        Args:
            state_dict: State dictionary
            strict: Whether to strictly match key names
            
        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

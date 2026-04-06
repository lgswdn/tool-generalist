# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        fuser_hidden_dims=None,  # New parameter for feature fusion
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Input dimensions
        input_dim_a = num_actor_obs
        input_dim_c = num_critic_obs

        # Fuser MLP for feature fusion (optional)
        self.use_fuser = fuser_hidden_dims is not None and len(fuser_hidden_dims) > 0
        if self.use_fuser:
            print(f"Using fuser with hidden dims: {fuser_hidden_dims}")
            # Build fuser network for actor observations
            fuser_layers_a = [nn.Linear(input_dim_a, fuser_hidden_dims[0]), activation]
            for i in range(len(fuser_hidden_dims) - 1):
                fuser_layers_a.append(nn.Linear(fuser_hidden_dims[i], fuser_hidden_dims[i + 1]))
                fuser_layers_a.append(activation)
            self.fuser_a = nn.Sequential(*fuser_layers_a)
            
            # Build fuser network for critic observations (if different from actor)
            if input_dim_c != input_dim_a:
                fuser_layers_c = [nn.Linear(input_dim_c, fuser_hidden_dims[0]), activation]
                for i in range(len(fuser_hidden_dims) - 1):
                    fuser_layers_c.append(nn.Linear(fuser_hidden_dims[i], fuser_hidden_dims[i + 1]))
                    fuser_layers_c.append(activation)
                self.fuser_c = nn.Sequential(*fuser_layers_c)
            else:
                # Share the same fuser if dimensions are the same
                self.fuser_c = self.fuser_a
            
            mlp_input_dim_a = fuser_hidden_dims[-1]
            mlp_input_dim_c = fuser_hidden_dims[-1]
            
            print(f"Fuser Actor: {self.fuser_a}")
            if self.fuser_c is not self.fuser_a:
                print(f"Fuser Critic: {self.fuser_c}")
            else:
                print("Fuser Critic: (shared with actor)")
        else:
            self.fuser_a = None
            self.fuser_c = None
            mlp_input_dim_a = input_dim_a
            mlp_input_dim_c = input_dim_c

        # Policy (Actor)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                actor_layers.append(resolve_nn_activation('tanh'))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function (Critic)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

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
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _get_actor_features(self, observations):
        """Process observations through optional fuser for actor"""
        if self.use_fuser:
            return self.fuser_a(observations)
        else:
            return observations

    def _get_critic_features(self, observations):
        """Process observations through optional fuser for critic"""
        if self.use_fuser:
            return self.fuser_c(observations)
        else:
            return observations

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
        # Get processed features through optional fuser
        features = self._get_actor_features(observations)
        
        # compute mean
        mean = self.actor(features)
        
        # Debug: Check for NaN or Inf in actor output
        if torch.any(torch.isnan(mean)):
            print(f"WARNING: Actor output contains NaN! mean stats: min={mean.min():.4f}, max={mean.max():.4f}")
            mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.any(torch.isinf(mean)):
            print(f"WARNING: Actor output contains Inf! mean stats: min={mean.min():.4f}, max={mean.max():.4f}")
            mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Debug: Check std values
        if torch.any(torch.isnan(std)) or torch.any(torch.isinf(std)):
            print(f"WARNING: Standard deviation contains NaN or Inf! std stats: min={std.min():.4f}, max={std.max():.4f}")
            std = torch.clamp(std, 1e-8, 10.0)
        
        # Clamp standard deviation to reasonable range
        std = torch.clamp(std, 1e-8, 10.0)
        
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        log_probs = self.distribution.log_prob(actions).sum(dim=-1)
        
        # Debug: Check for NaN or Inf in log probabilities
        if torch.any(torch.isnan(log_probs)):
            print(f"WARNING: Log probabilities contain NaN! log_probs stats: min={log_probs.min():.4f}, max={log_probs.max():.4f}")
            log_probs = torch.nan_to_num(log_probs, nan=-1e4)
        if torch.any(torch.isinf(log_probs)):
            print(f"WARNING: Log probabilities contain Inf! log_probs stats: min={log_probs.min():.4f}, max={log_probs.max():.4f}")
            log_probs = torch.nan_to_num(log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)
        
        return log_probs

    def act_inference(self, observations):
        # Get processed features through optional fuser
        features = self._get_actor_features(observations)
        actions_mean = self.actor(features)
        
        # Debug: Check for NaN or Inf in actor output
        if torch.any(torch.isnan(actions_mean)):
            print(f"WARNING: Actor inference output contains NaN! actions_mean stats: min={actions_mean.min():.4f}, max={actions_mean.max():.4f}")
            actions_mean = torch.nan_to_num(actions_mean, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.any(torch.isinf(actions_mean)):
            print(f"WARNING: Actor inference output contains Inf! actions_mean stats: min={actions_mean.min():.4f}, max={actions_mean.max():.4f}")
            actions_mean = torch.nan_to_num(actions_mean, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clamp actions to reasonable range
        actions_clamped = torch.clamp(actions_mean, -50, 50)
        if not torch.equal(actions_mean, actions_clamped):
            print(f"WARNING: Actor inference output clamped! Original range: [{actions_mean.min():.4f}, {actions_mean.max():.4f}]")
        
        return actions_clamped

    def evaluate(self, critic_observations, **kwargs):
        # Get processed features through optional fuser
        features = self._get_critic_features(critic_observations)
        value = self.critic(features)
        
        # Debug: Check for NaN or Inf in critic output
        nan_count = torch.sum(torch.isnan(value)).item()
        inf_count = torch.sum(torch.isinf(value)).item()
        
        if nan_count > 0:
            print(f"ERROR: Critic output contains {nan_count} NaN values!")
            print(f"NaN locations: {torch.where(torch.isnan(value))}")
            value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
            print(f"Replaced NaN values with 0.0")
        
        if inf_count > 0:
            print(f"ERROR: Critic output contains {inf_count} Inf values!")
            print(f"Inf locations: {torch.where(torch.isinf(value))}")
            value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
            print(f"Replaced Inf values with ±1e4")
        
        # Clamp values to prevent explosion while preserving reasonable range
        value_original = value.clone()
        value = torch.clamp(value, -1e4, 1e4)
        
        # Debug: Check if clamping occurred
        clamping_occurred = not torch.equal(value_original, value)
        if clamping_occurred:
            clamp_count = torch.sum(torch.ne(value_original, value)).item()
            print(f"WARNING: Critic output clamped! {clamp_count} values were outside [-1e4, 1e4]")
            print(f"Original range: [{value_original.min():.4f}, {value_original.max():.4f}]")
            print(f"Clamped range: [{value.min():.4f}, {value.max():.4f}]")
        
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
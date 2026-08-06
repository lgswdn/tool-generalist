from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_tg import ActorCriticTG
from rsl_rl.modules.tg_policy_common import build_context_vector, build_mlp, initialize_action_noise
from rsl_rl.utils import resolve_nn_activation


class ActorCriticTGOutputGate(nn.Module):
    """Two TG experts with trainable final gates over post-fusion features plus context."""

    is_recurrent = False
    tracks_encoder_feature_calls = False
    supports_cached_features = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        expert_a_checkpoint: str,
        expert_b_checkpoint: str,
        output_gate_freeze_experts: bool = True,
        output_gate_hidden_dims: tuple[int, ...] | list[int] = (64,),
        output_gate_initial_expert_a_weight: float = 0.8,
        output_gate_per_action: bool = False,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **expert_kwargs,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.noise_std_type = noise_std_type
        self.freeze_encoder = True
        self.output_gate_freeze_experts = bool(output_gate_freeze_experts)
        self.output_gate_per_action = bool(output_gate_per_action)

        expert_kwargs = dict(expert_kwargs)
        expert_kwargs["activation"] = activation
        expert_kwargs["init_noise_std"] = init_noise_std
        expert_kwargs["noise_std_type"] = noise_std_type

        self.expert_a = ActorCriticTG(num_actor_obs, num_critic_obs, num_actions, **expert_kwargs)
        self.expert_b = ActorCriticTG(num_actor_obs, num_critic_obs, num_actions, **expert_kwargs)
        self._load_expert(self.expert_a, expert_a_checkpoint, "expert_a")
        self._load_expert(self.expert_b, expert_b_checkpoint, "expert_b")

        if self.output_gate_freeze_experts:
            self._freeze_expert(self.expert_a)
            self._freeze_expert(self.expert_b)

        gate_hidden_dims = tuple(int(dim) for dim in output_gate_hidden_dims)
        if len(gate_hidden_dims) == 0 or any(dim <= 0 for dim in gate_hidden_dims):
            raise ValueError("output_gate_hidden_dims must contain positive hidden dimensions")
        if not 0.0 < float(output_gate_initial_expert_a_weight) < 1.0:
            raise ValueError("output_gate_initial_expert_a_weight must be in (0, 1)")

        activation_fn = resolve_nn_activation(activation)
        actor_gate_dim = self.num_actions if self.output_gate_per_action else 1
        gate_input_dim = 2 * int(self.expert_a.fusion_out_dim) + int(self.expert_a.context_dim)
        self.actor_gate = build_mlp(gate_input_dim, gate_hidden_dims, activation_fn, actor_gate_dim)
        self.critic_gate = build_mlp(gate_input_dim, gate_hidden_dims, activation_fn, 1)
        self._init_gate(self.actor_gate, output_gate_initial_expert_a_weight)
        self._init_gate(self.critic_gate, output_gate_initial_expert_a_weight)

        initialize_action_noise(
            self,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            noise_std_type=self.noise_std_type,
        )
        self._last_actor_gate: torch.Tensor | None = None
        self._last_critic_gate: torch.Tensor | None = None

    @staticmethod
    def _load_expert(expert: ActorCriticTG, checkpoint_path: str, label: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError(f"{label} checkpoint has no model_state_dict: {checkpoint_path}")
        expert.load_state_dict(state, strict=True)

    @staticmethod
    def _freeze_expert(expert: ActorCriticTG) -> None:
        for param in expert.parameters():
            param.requires_grad_(False)
        expert.eval()

    @staticmethod
    def _init_gate(gate: nn.Module, initial_expert_a_weight: float) -> None:
        final_linear = None
        for module in gate.modules():
            if isinstance(module, nn.Linear):
                final_linear = module
        if final_linear is None:
            return
        bias = math.log(float(initial_expert_a_weight) / (1.0 - float(initial_expert_a_weight)))
        with torch.no_grad():
            final_linear.weight.zero_()
            final_linear.bias.fill_(bias)

    def _action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def _context_from_obs(self, observations: torch.Tensor) -> torch.Tensor:
        return build_context_vector(self.expert_a._split_observations(observations))

    def _gate_input(
        self,
        observations: torch.Tensor,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        return self._gate_input_from_context(self._context_from_obs(observations), feature_a, feature_b)

    @staticmethod
    def _gate_input_from_context(
        ctx_vec: torch.Tensor,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([feature_a, feature_b, ctx_vec], dim=-1)

    def _expert_actor_features_actions(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.output_gate_freeze_experts:
            with torch.no_grad():
                feature_a = self.expert_a._get_features(observations, branch="actor")
                feature_b = self.expert_b._get_features(observations, branch="actor")
                action_a = self.expert_a.actor(feature_a)
                action_b = self.expert_b.actor(feature_b)
            return feature_a, feature_b, action_a, action_b
        feature_a = self.expert_a._get_features(observations, branch="actor")
        feature_b = self.expert_b._get_features(observations, branch="actor")
        return feature_a, feature_b, self.expert_a.actor(feature_a), self.expert_b.actor(feature_b)

    def _expert_critic_features_values(
        self,
        critic_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.output_gate_freeze_experts:
            with torch.no_grad():
                feature_a = self.expert_a._get_features(critic_observations, branch="critic")
                feature_b = self.expert_b._get_features(critic_observations, branch="critic")
                value_a = self.expert_a.critic(feature_a)
                value_b = self.expert_b.critic(feature_b)
            return feature_a, feature_b, value_a, value_b
        feature_a = self.expert_a._get_features(critic_observations, branch="critic")
        feature_b = self.expert_b._get_features(critic_observations, branch="critic")
        return feature_a, feature_b, self.expert_a.critic(feature_a), self.expert_b.critic(feature_b)

    def get_cached_encoder_features(self, observations: torch.Tensor):
        return self.expert_a._tokenize(observations)

    def _expert_actor_features_actions_from_cached(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.output_gate_freeze_experts:
            with torch.no_grad():
                feature_a = self.expert_a._features_from_tokens_context(all_tokens, ctx_vec, branch="actor")
                feature_b = self.expert_b._features_from_tokens_context(all_tokens, ctx_vec, branch="actor")
                action_a = self.expert_a.actor(feature_a)
                action_b = self.expert_b.actor(feature_b)
            return feature_a, feature_b, action_a, action_b
        feature_a = self.expert_a._features_from_tokens_context(all_tokens, ctx_vec, branch="actor")
        feature_b = self.expert_b._features_from_tokens_context(all_tokens, ctx_vec, branch="actor")
        return feature_a, feature_b, self.expert_a.actor(feature_a), self.expert_b.actor(feature_b)

    def _expert_critic_features_values_from_cached(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.output_gate_freeze_experts:
            with torch.no_grad():
                feature_a = self.expert_a._features_from_tokens_context(all_tokens, ctx_vec, branch="critic")
                feature_b = self.expert_b._features_from_tokens_context(all_tokens, ctx_vec, branch="critic")
                value_a = self.expert_a.critic(feature_a)
                value_b = self.expert_b.critic(feature_b)
            return feature_a, feature_b, value_a, value_b
        feature_a = self.expert_a._features_from_tokens_context(all_tokens, ctx_vec, branch="critic")
        feature_b = self.expert_b._features_from_tokens_context(all_tokens, ctx_vec, branch="critic")
        return feature_a, feature_b, self.expert_a.critic(feature_a), self.expert_b.critic(feature_b)

    def _mixed_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        feature_a, feature_b, action_a, action_b = self._expert_actor_features_actions(observations)
        gate_input = self._gate_input(observations, feature_a, feature_b)
        gate = torch.sigmoid(self.actor_gate(gate_input))
        self._last_actor_gate = gate.detach()
        return gate * action_a + (1.0 - gate) * action_b

    def _mixed_action_mean_from_cached(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        feature_a, feature_b, action_a, action_b = self._expert_actor_features_actions_from_cached(all_tokens, ctx_vec)
        gate_input = self._gate_input_from_context(ctx_vec, feature_a, feature_b)
        gate = torch.sigmoid(self.actor_gate(gate_input))
        self._last_actor_gate = gate.detach()
        return gate * action_a + (1.0 - gate) * action_b

    def update_distribution(self, observations: torch.Tensor):
        mean = self._mixed_action_mean(observations)
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        return self._mixed_action_mean(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        feature_a, feature_b, value_a, value_b = self._expert_critic_features_values(critic_observations)
        gate_input = self._gate_input(critic_observations, feature_a, feature_b)
        gate = torch.sigmoid(self.critic_gate(gate_input))
        self._last_critic_gate = gate.detach()
        return gate * value_a + (1.0 - gate) * value_b

    def act_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        mean = self._mixed_action_mean_from_cached(all_tokens, ctx_vec)
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))
        return self.distribution.sample()

    def evaluate_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        feature_a, feature_b, value_a, value_b = self._expert_critic_features_values_from_cached(all_tokens, ctx_vec)
        gate_input = self._gate_input_from_context(ctx_vec, feature_a, feature_b)
        gate = torch.sigmoid(self.critic_gate(gate_input))
        self._last_critic_gate = gate.detach()
        return gate * value_a + (1.0 - gate) * value_b

    def act_inference_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        return self._mixed_action_mean_from_cached(all_tokens, ctx_vec)

    def reset(self, dones=None):
        pass

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actions_log_prob_from_cached_features(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def train(self, mode=True):
        super().train(mode)
        if self.output_gate_freeze_experts:
            self.expert_a.eval()
            self.expert_b.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def diagnostic_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        if self._last_actor_gate is not None:
            actor_gate = self._last_actor_gate.float()
            stats["gate_actor_expert_a_mean"] = float(actor_gate.mean().item())
            stats["gate_actor_expert_a_min"] = float(actor_gate.min().item())
            stats["gate_actor_expert_a_max"] = float(actor_gate.max().item())
        if self._last_critic_gate is not None:
            critic_gate = self._last_critic_gate.float()
            stats["gate_critic_expert_a_mean"] = float(critic_gate.mean().item())
        return stats

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

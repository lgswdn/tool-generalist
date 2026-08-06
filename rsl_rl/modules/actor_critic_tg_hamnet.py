from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_tg import ActorCriticTG


class HAMNetNodeRouter(nn.Module):
    """Predict a module mixture for every weight and bias tensor."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_parameter_sets: int,
        num_modules: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        dims = (int(input_dim), *(int(dim) for dim in hidden_dims))
        layers: list[nn.Module] = []
        for dim_in, dim_out in zip(dims, dims[1:]):
            layers.extend((nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.GELU()))
        layers.append(nn.Linear(dims[-1], int(num_parameter_sets) * int(num_modules)))
        self.network = nn.Sequential(*layers)
        self.num_parameter_sets = int(num_parameter_sets)
        self.num_modules = int(num_modules)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.network(context)
        logits = logits.view(
            context.shape[0], self.num_parameter_sets, self.num_modules
        )
        return F.softmax(logits, dim=-1)


class HAMNetModularMLP(nn.Module):
    """MLP whose layer parameters are synthesized from learned module banks."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        num_modules: int,
    ) -> None:
        super().__init__()
        dims = (int(input_dim), *(int(dim) for dim in hidden_dims), int(output_dim))
        self.weight_banks = nn.ParameterList()
        self.bias_banks = nn.ParameterList()
        for dim_in, dim_out in zip(dims, dims[1:]):
            weights = nn.Parameter(torch.empty(int(num_modules), dim_out, dim_in))
            biases = nn.Parameter(torch.zeros(int(num_modules), dim_out))
            for module_weight in weights:
                nn.init.orthogonal_(module_weight, gain=nn.init.calculate_gain("tanh"))
            self.weight_banks.append(weights)
            self.bias_banks.append(biases)

    @property
    def num_parameter_sets(self) -> int:
        return 2 * len(self.weight_banks)

    def forward(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        if modulation.shape[1] != self.num_parameter_sets:
            raise ValueError(
                "HAMNet modulation count does not match the modular MLP: "
                f"got {modulation.shape[1]}, expected {self.num_parameter_sets}"
            )
        for layer_index, (weights, biases) in enumerate(
            zip(self.weight_banks, self.bias_banks)
        ):
            weight_mix = modulation[:, 2 * layer_index]
            bias_mix = modulation[:, 2 * layer_index + 1]
            x = torch.einsum("bm,moi,bi->bo", weight_mix, weights, x)
            x = x + torch.einsum("bm,mo->bo", bias_mix, biases)
            if layer_index + 1 < len(self.weight_banks):
                x = torch.tanh(x)
        return x


class ActorCriticTGHAMNet(ActorCriticTG):
    """Original TG encoder/fusion with HAMNet-style modular hypernetwork heads."""

    def __init__(
        self,
        *args,
        hamnet_num_modules: int = 4,
        hamnet_hidden_dims: Sequence[int] = (256, 128, 128, 64),
        hamnet_router_hidden_dims: Sequence[int] = (256, 256),
        separate_actor_critic_fusion: bool = False,
        **kwargs,
    ) -> None:
        if not separate_actor_critic_fusion:
            raise ValueError(
                "ActorCriticTGHAMNet requires separate_actor_critic_fusion=True"
            )
        if int(hamnet_num_modules) < 2:
            raise ValueError("ActorCriticTGHAMNet requires at least two modules")
        super().__init__(
            *args,
            separate_actor_critic_fusion=separate_actor_critic_fusion,
            **kwargs,
        )
        head_kwargs = {
            "input_dim": self.fusion_out_dim,
            "hidden_dims": tuple(hamnet_hidden_dims),
            "num_modules": int(hamnet_num_modules),
        }
        self.actor = HAMNetModularMLP(output_dim=self.num_actions, **head_kwargs)
        self.critic = HAMNetModularMLP(output_dim=1, **head_kwargs)
        self._actor_parameter_sets = self.actor.num_parameter_sets
        num_parameter_sets = (
            self._actor_parameter_sets + self.critic.num_parameter_sets
        )
        self.hamnet_router = HAMNetNodeRouter(
            input_dim=2 * self.token_dim + self.context_dim,
            num_parameter_sets=num_parameter_sets,
            num_modules=int(hamnet_num_modules),
            hidden_dims=tuple(hamnet_router_hidden_dims),
        )

    def _modulation(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
        *,
        branch: str,
    ) -> torch.Tensor:
        routing_context = torch.cat(
            (all_tokens.mean(dim=1), all_tokens.amax(dim=1), ctx_vec),
            dim=-1,
        )
        modulation = self.hamnet_router(routing_context)
        if branch == "actor":
            return modulation[:, : self._actor_parameter_sets]
        if branch == "critic":
            return modulation[:, self._actor_parameter_sets :]
        raise ValueError(f"Unknown ActorCriticTGHAMNet branch: {branch!r}")

    def _head_inputs(
        self,
        observations: torch.Tensor,
        *,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_tokens, ctx_vec = self._tokenize(observations)
        features = self._features_from_tokens_context(
            all_tokens, ctx_vec, branch=branch
        )
        return features, self._modulation(all_tokens, ctx_vec, branch=branch)

    def update_distribution(self, observations: torch.Tensor):
        features, modulation = self._head_inputs(observations, branch="actor")
        mean = self.actor(features, modulation)
        self.distribution = Normal(
            mean, torch.clamp(self._action_std(mean), min=1e-6)
        )

    def act_inference(self, observations: torch.Tensor):
        features, modulation = self._head_inputs(observations, branch="actor")
        return self.actor(features, modulation)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        features, modulation = self._head_inputs(
            critic_observations, branch="critic"
        )
        return self.critic(features, modulation)

    def act_from_cached_features(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
    ):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        modulation = self._modulation(all_tokens, ctx_vec, branch="actor")
        mean = self.actor(features, modulation)
        self.distribution = Normal(
            mean, torch.clamp(self._action_std(mean), min=1e-6)
        )
        return self.distribution.sample()

    def evaluate_from_cached_features(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
    ):
        features = self._features_from_tokens_context(
            all_tokens, ctx_vec, branch="critic"
        )
        modulation = self._modulation(all_tokens, ctx_vec, branch="critic")
        return self.critic(features, modulation)

    def act_inference_from_cached_features(
        self,
        all_tokens: torch.Tensor,
        ctx_vec: torch.Tensor,
    ):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        modulation = self._modulation(all_tokens, ctx_vec, branch="actor")
        return self.actor(features, modulation)

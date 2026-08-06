from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_tg import ActorCriticTG


class SoftModularHead(nn.Module):
    """Soft-Module gated head adapted to PPO actor/critic outputs."""

    def __init__(
        self,
        *,
        input_dim: int,
        task_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_modules: int = 4,
        module_hidden: int = 128,
        gating_hidden: int = 128,
        num_gating_layers: int = 1,
        cond_ob: bool = True,
        add_bn: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_modules = int(num_modules)
        self.cond_ob = bool(cond_ob)
        if self.num_layers < 1:
            raise ValueError("sm_num_layers must be >= 1")
        if self.num_modules < 1:
            raise ValueError("sm_num_modules must be >= 1")

        self.base = nn.Linear(int(input_dim), int(module_hidden))
        self.embedding_base = nn.Linear(int(task_dim), int(module_hidden))

        module_input_dim = int(module_hidden)
        self.layer_modules = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.ModuleList()
            for _ in range(self.num_modules):
                if add_bn:
                    layer.append(
                        nn.Sequential(
                            nn.BatchNorm1d(module_input_dim),
                            nn.Linear(module_input_dim, int(module_hidden)),
                            nn.BatchNorm1d(int(module_hidden)),
                        )
                    )
                else:
                    layer.append(nn.Linear(module_input_dim, int(module_hidden)))
            self.layer_modules.append(layer)
            module_input_dim = int(module_hidden)

        gating_layers: list[nn.Module] = []
        gating_input_dim = int(module_hidden)
        for _ in range(int(num_gating_layers)):
            gating_layers.append(nn.Linear(gating_input_dim, int(gating_hidden)))
            gating_layers.append(nn.ELU())
            gating_input_dim = int(gating_hidden)
        self.gating_mlp = nn.Sequential(*gating_layers) if gating_layers else nn.Identity()
        self.gating_weight_0 = (
            nn.Linear(gating_input_dim, self.num_modules * self.num_modules)
            if self.num_layers > 1
            else None
        )
        self.gating_weight_cond = nn.ModuleList()
        self.gating_weights = nn.ModuleList()
        for layer_idx in range(max(self.num_layers - 2, 0)):
            self.gating_weight_cond.append(
                nn.Linear((layer_idx + 1) * self.num_modules * self.num_modules, gating_input_dim)
            )
            self.gating_weights.append(nn.Linear(gating_input_dim, self.num_modules * self.num_modules))
        self.gating_weight_cond_last = (
            nn.Linear((self.num_layers - 1) * self.num_modules * self.num_modules, gating_input_dim)
            if self.num_layers > 1
            else None
        )
        self.gating_last = nn.Linear(gating_input_dim, self.num_modules)
        self.output = nn.Linear(int(module_hidden), int(output_dim))

    def forward(self, x: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        base = F.elu(self.base(x))
        embedding = self.embedding_base(task_embedding)
        if self.cond_ob:
            embedding = embedding * base
        embedding = self.gating_mlp(F.elu(embedding))

        flat_weights = []
        if self.num_layers > 1:
            weight_shape = (-1, self.num_modules, self.num_modules)
            raw_weight = self.gating_weight_0(F.elu(embedding))
            weight = F.softmax(raw_weight.view(weight_shape), dim=-1)
            flat_weights.append(weight.reshape(weight.shape[0], -1))

            for gating_weight, gating_weight_cond in zip(self.gating_weights, self.gating_weight_cond):
                cond = gating_weight_cond(torch.cat(flat_weights, dim=-1))
                cond = F.elu(cond * embedding)
                raw_weight = gating_weight(cond)
                weight = F.softmax(raw_weight.view(weight_shape), dim=-1)
                flat_weights.append(weight.reshape(weight.shape[0], -1))

            cond_last = self.gating_weight_cond_last(torch.cat(flat_weights, dim=-1))
            cond_last = F.elu(cond_last * embedding)
        else:
            cond_last = F.elu(embedding)
        last_weight = F.softmax(self.gating_last(cond_last), dim=-1)

        module_outputs = torch.stack(
            [module(base) for module in self.layer_modules[0]],
            dim=-2,
        )
        weights = [flat.view(-1, self.num_modules, self.num_modules) for flat in flat_weights]
        for layer_idx, weight in enumerate(weights):
            next_outputs = []
            for module_idx, module in enumerate(self.layer_modules[layer_idx + 1]):
                module_input = (module_outputs * weight[:, module_idx, :].unsqueeze(-1)).sum(dim=-2)
                next_outputs.append(module(F.elu(module_input)))
            module_outputs = torch.stack(next_outputs, dim=-2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(dim=-2)
        return self.output(F.elu(out))


class ActorCriticTGSM(ActorCriticTG):
    """TCE PPO actor-critic with separate Soft-Module actor and critic heads."""

    def __init__(
        self,
        *args,
        task_embedding_dim: int = 2,
        sm_num_layers: int = 2,
        sm_num_modules: int = 4,
        sm_module_hidden: int = 128,
        sm_gating_hidden: int = 128,
        sm_num_gating_layers: int = 1,
        sm_cond_ob: bool = True,
        sm_add_bn: bool = False,
        separate_actor_critic_fusion: bool = False,
        **kwargs,
    ) -> None:
        if not separate_actor_critic_fusion:
            raise ValueError("ActorCriticTGSM requires separate_actor_critic_fusion=True")
        if int(task_embedding_dim) <= 0:
            raise ValueError("ActorCriticTGSM requires task_embedding_dim > 0")
        super().__init__(
            *args,
            task_embedding_dim=task_embedding_dim,
            separate_actor_critic_fusion=separate_actor_critic_fusion,
            **kwargs,
        )
        self.sm_task_embedding_dim = int(task_embedding_dim)
        common = {
            "input_dim": self.fusion_out_dim,
            "task_dim": self.sm_task_embedding_dim,
            "num_layers": int(sm_num_layers),
            "num_modules": int(sm_num_modules),
            "module_hidden": int(sm_module_hidden),
            "gating_hidden": int(sm_gating_hidden),
            "num_gating_layers": int(sm_num_gating_layers),
            "cond_ob": bool(sm_cond_ob),
            "add_bn": bool(sm_add_bn),
        }
        self.actor = SoftModularHead(output_dim=self.num_actions, **common)
        self.critic = SoftModularHead(output_dim=1, **common)

    def _task_embedding_from_context(self, ctx_vec: torch.Tensor) -> torch.Tensor:
        start = ctx_vec.shape[-1] - self.physics_dim - self.sm_task_embedding_dim
        stop = ctx_vec.shape[-1] - self.physics_dim
        return ctx_vec[:, start:stop]

    def _features_and_task_from_obs(
        self,
        observations: torch.Tensor,
        *,
        branch: str = "actor",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_tokens, ctx_vec = self._tokenize(observations)
        features = self._features_from_tokens_context(all_tokens, ctx_vec, branch=branch)
        return features, self._task_embedding_from_context(ctx_vec)

    def update_distribution(self, observations: torch.Tensor):
        features, task_embedding = self._features_and_task_from_obs(observations)
        mean = self.actor(features, task_embedding)
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))

    def act_inference(self, observations: torch.Tensor):
        features, task_embedding = self._features_and_task_from_obs(observations)
        return self.actor(features, task_embedding)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        features, task_embedding = self._features_and_task_from_obs(critic_observations, branch="critic")
        return self.critic(features, task_embedding)

    def act_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        task_embedding = self._task_embedding_from_context(ctx_vec)
        mean = self.actor(features, task_embedding)
        self.distribution = Normal(mean, torch.clamp(self._action_std(mean), min=1e-6))
        return self.distribution.sample()

    def evaluate_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        features = self._features_from_tokens_context(all_tokens, ctx_vec, branch="critic")
        task_embedding = self._task_embedding_from_context(ctx_vec)
        return self.critic(features, task_embedding)

    def act_inference_from_cached_features(self, all_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        features = self._features_from_tokens_context(all_tokens, ctx_vec)
        task_embedding = self._task_embedding_from_context(ctx_vec)
        return self.actor(features, task_embedding)

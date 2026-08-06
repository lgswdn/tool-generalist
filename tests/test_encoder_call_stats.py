from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RolloutStorage


class DummyEncoderPolicy(nn.Module):
    is_recurrent = False
    tracks_encoder_feature_calls = True

    def __init__(self, *, use_cache: bool, obs_dim: int = 4, num_actions: int = 1):
        super().__init__()
        self.supports_cached_features = use_cache
        self.freeze_point2vec = use_cache
        self.num_actions = num_actions
        self.actor_weight = nn.Parameter(torch.zeros(num_actions))
        self.value_weight = nn.Parameter(torch.zeros(1))
        self.distribution = None
        self.reset_call_history()

    def reset_call_history(self):
        self.standard_actor_calls = 0
        self.standard_critic_calls = 0
        self.cached_encoder_observations = []
        self.cached_actor_calls = 0
        self.cached_critic_calls = 0

    def _set_distribution(self, batch_size: int):
        mean = self.actor_weight.expand(batch_size, self.num_actions)
        self.distribution = Normal(mean, torch.ones_like(mean))

    def act(self, observations: torch.Tensor, **kwargs):
        self.standard_actor_calls += 1
        self._set_distribution(observations.shape[0])
        return self.distribution.mean

    def act_inference(self, observations: torch.Tensor, **kwargs):
        return self.actor_weight.expand(observations.shape[0], self.num_actions)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        self.standard_critic_calls += 1
        return self.value_weight.expand(critic_observations.shape[0], 1) + critic_observations[:, :1]

    def get_actions_log_prob(self, actions: torch.Tensor, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_cached_encoder_features(self, observations: torch.Tensor):
        self.cached_encoder_observations.append(observations.detach().clone())
        return observations[:, :2].clone(), observations[:, :1].clone()

    def act_from_cached_features(self, encoder_features: torch.Tensor, extra_state: torch.Tensor):
        self.cached_actor_calls += 1
        self._set_distribution(encoder_features.shape[0])
        return self.distribution.mean

    def evaluate_from_cached_features(self, encoder_features: torch.Tensor, extra_state: torch.Tensor):
        self.cached_critic_calls += 1
        return self.value_weight.expand(encoder_features.shape[0], 1) + encoder_features[:, :1]

    def get_actions_log_prob_from_cached_features(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


class IncompleteCachedPolicy(nn.Module):
    is_recurrent = False
    supports_cached_features = True
    freeze_point2vec = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def get_cached_encoder_features(self, observations: torch.Tensor):
        return observations[:, :1], observations[:, :1]


def _stage_total(stats: dict, stage: str) -> int:
    return int(stats["stages"][stage].sum().item())


def _double_batch_augmentation(obs=None, actions=None, env=None, obs_type=None):
    if obs is not None:
        obs = obs.repeat(2, 1)
    if actions is not None:
        actions = actions.repeat(2, 1)
    return obs, actions


def _run_one_ppo_iteration(
    *,
    use_cache: bool,
    num_envs: int = 4,
    num_steps: int = 3,
    num_epochs: int = 2,
    symmetry_cfg: dict | None = None,
):
    obs_dim = 4
    num_actions = 1
    policy = DummyEncoderPolicy(use_cache=use_cache, obs_dim=obs_dim, num_actions=num_actions)
    ppo = PPO(
        policy,
        num_learning_epochs=num_epochs,
        num_mini_batches=3,
        desired_kl=None,
        device="cpu",
        symmetry_cfg=symmetry_cfg,
    )
    ppo.init_storage("rl", num_envs, num_steps, [obs_dim], [obs_dim], [num_actions])
    policy.reset_call_history()
    ppo.reset_encoder_call_stats()

    for step in range(num_steps):
        obs = torch.full((num_envs, obs_dim), float(step))
        critic_obs = obs + 0.5
        ppo.act(obs, critic_obs)
        ppo.process_env_step(
            rewards=torch.zeros(num_envs),
            dones=torch.zeros(num_envs, dtype=torch.bool),
            infos={},
        )

    ppo.compute_returns(torch.zeros(num_envs, obs_dim))
    ppo.update()
    return ppo.get_encoder_call_stats(), ppo.encoder_call_stats_summary(iteration=7)


def test_storage_feedforward_generator_can_include_local_env_ids():
    storage = RolloutStorage(
        "rl",
        num_envs=3,
        num_transitions_per_env=2,
        obs_shape=[1],
        privileged_obs_shape=[1],
        actions_shape=[1],
        device="cpu",
    )
    for step in range(storage.num_transitions_per_env):
        for env_id in range(storage.num_envs):
            storage.observations[step, env_id, 0] = env_id
            storage.privileged_observations[step, env_id, 0] = env_id

    default_batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert len(default_batch) == 14

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1, include_env_ids=True))
    obs_batch = batch[0]
    env_ids_batch = batch[-1]

    assert len(batch) == 15
    assert torch.equal(env_ids_batch.cpu(), obs_batch[:, 0].long().cpu())
    assert torch.equal(torch.bincount(env_ids_batch.cpu(), minlength=3), torch.tensor([2, 2, 2]))


def test_collect_timing_cache_off_uses_cached_helpers_without_storage_cache():
    num_envs = 3
    obs_dim = 4
    policy = DummyEncoderPolicy(use_cache=False, obs_dim=obs_dim)
    ppo = PPO(policy, desired_kl=None, device="cpu")
    ppo.init_storage("rl", num_envs, 1, [obs_dim], [obs_dim], [1])
    policy.reset_call_history()
    ppo.reset_encoder_call_stats()
    ppo.reset_collect_timing()

    obs = torch.zeros(num_envs, obs_dim)
    critic_obs = torch.ones(num_envs, obs_dim)
    ppo.act(obs, critic_obs)

    stats = ppo.get_encoder_call_stats()
    assert ppo._use_cached_features is False
    assert ppo.storage.encoder_features is None
    assert policy.standard_actor_calls == 0
    assert policy.standard_critic_calls == 0
    assert policy.cached_actor_calls == 1
    assert policy.cached_critic_calls == 1
    assert len(policy.cached_encoder_observations) == 2
    assert torch.equal(policy.cached_encoder_observations[0], obs)
    assert torch.equal(policy.cached_encoder_observations[1], critic_obs)
    assert _stage_total(stats, "collect_actor") == num_envs
    assert _stage_total(stats, "collect_critic") == num_envs
    assert _stage_total(stats, "collect_cache") == 0

    summary = ppo.collect_timing_summary(iteration=5, total_time=1.0)
    assert "[CollectTiming]" in summary
    assert "[rank 0/1]" in summary
    assert "[cache=off]" in summary
    assert "iter=5" in summary
    assert "total=1.000s" in summary
    assert "encoder=" in summary
    assert "actor_critic=" in summary
    assert "other=" in summary
    assert "[CollectOtherTiming]" in summary
    assert "env_step=" in summary
    assert "transfer_normalize=" in summary
    assert "process_env_step=" in summary
    assert "bookkeeping=" in summary
    assert "unaccounted=" in summary
    assert "%" in summary


def test_collect_timing_cache_on_caches_actor_and_critic_encoder_features():
    num_envs = 3
    obs_dim = 4
    policy = DummyEncoderPolicy(use_cache=True, obs_dim=obs_dim)
    ppo = PPO(policy, desired_kl=None, device="cpu")
    ppo.init_storage("rl", num_envs, 1, [obs_dim], [obs_dim], [1])
    policy.reset_call_history()
    ppo.reset_encoder_call_stats()
    ppo.reset_collect_timing()

    obs = torch.zeros(num_envs, obs_dim)
    critic_obs = torch.ones(num_envs, obs_dim)
    ppo.act(obs, critic_obs)

    stats = ppo.get_encoder_call_stats()
    assert ppo._use_cached_features is True
    assert policy.standard_actor_calls == 0
    assert policy.standard_critic_calls == 0
    assert policy.cached_actor_calls == 1
    assert policy.cached_critic_calls == 1
    assert len(policy.cached_encoder_observations) == 2
    assert torch.equal(policy.cached_encoder_observations[0], obs)
    assert torch.equal(policy.cached_encoder_observations[1], critic_obs)
    assert torch.equal(ppo.transition.values, critic_obs[:, :1])
    assert _stage_total(stats, "collect_cache") == 2 * num_envs
    assert _stage_total(stats, "collect_actor") == 0
    assert _stage_total(stats, "collect_critic") == 0
    assert "[cache=on]" in ppo.collect_timing_summary(iteration=6, total_time=1.0)


def test_collect_timing_cache_on_reuses_encoder_features_when_actor_and_critic_share_tensor():
    num_envs = 3
    obs_dim = 4
    policy = DummyEncoderPolicy(use_cache=True, obs_dim=obs_dim)
    ppo = PPO(policy, desired_kl=None, device="cpu")
    ppo.init_storage("rl", num_envs, 1, [obs_dim], [obs_dim], [1])
    policy.reset_call_history()
    ppo.reset_encoder_call_stats()
    ppo.reset_collect_timing()

    obs = torch.ones(num_envs, obs_dim)
    ppo.act(obs, obs)

    stats = ppo.get_encoder_call_stats()
    assert len(policy.cached_encoder_observations) == 1
    assert torch.equal(policy.cached_encoder_observations[0], obs)
    assert _stage_total(stats, "collect_cache") == num_envs


def test_storage_cache_requires_complete_cached_feature_helpers():
    policy = IncompleteCachedPolicy()
    ppo = PPO(policy, desired_kl=None, device="cpu")
    ppo.init_storage("rl", 2, 1, [4], [4], [1])

    assert ppo._use_cached_features is False
    assert ppo.storage.encoder_features is None


def test_encoder_call_stats_cache_off_counts_collect_returns_and_learning():
    num_envs = 4
    num_steps = 3
    num_epochs = 2
    stats, summary = _run_one_ppo_iteration(use_cache=False, num_envs=num_envs, num_steps=num_steps, num_epochs=num_epochs)

    assert _stage_total(stats, "collect_actor") == num_envs * num_steps
    assert _stage_total(stats, "collect_critic") == num_envs * num_steps
    assert _stage_total(stats, "compute_returns") == num_envs
    assert _stage_total(stats, "learn_actor") == num_envs * num_steps * num_epochs
    assert _stage_total(stats, "learn_critic") == num_envs * num_steps * num_epochs
    assert _stage_total(stats, "collect_cache") == 0

    expected_per_env = (2 * num_steps) + 1 + (2 * num_steps * num_epochs)
    assert stats["per_env"].cpu().tolist() == [expected_per_env] * num_envs
    assert stats["total"] == num_envs * expected_per_env
    assert "[cache=off]" in summary
    assert "unique=4" in summary


def test_encoder_call_stats_cache_on_counts_only_real_encoder_extractions():
    num_envs = 4
    num_steps = 3
    num_epochs = 2
    stats, summary = _run_one_ppo_iteration(use_cache=True, num_envs=num_envs, num_steps=num_steps, num_epochs=num_epochs)

    assert _stage_total(stats, "collect_cache") == 2 * num_envs * num_steps
    assert _stage_total(stats, "compute_returns") == num_envs
    assert _stage_total(stats, "collect_actor") == 0
    assert _stage_total(stats, "collect_critic") == 0
    assert _stage_total(stats, "learn_actor") == 0
    assert _stage_total(stats, "learn_critic") == 0

    expected_per_env = (2 * num_steps) + 1
    assert stats["per_env"].cpu().tolist() == [expected_per_env] * num_envs
    assert stats["total"] == num_envs * expected_per_env
    assert "[cache=on]" in summary
    assert "stages=collect_cache:24,compute_returns:4" in summary


def test_encoder_call_stats_counts_symmetry_inference_without_data_augmentation():
    num_envs = 4
    num_steps = 3
    num_epochs = 2
    symmetry_cfg = {
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "data_augmentation_func": _double_batch_augmentation,
        "mirror_loss_coeff": 0.1,
    }
    stats, summary = _run_one_ppo_iteration(
        use_cache=False,
        num_envs=num_envs,
        num_steps=num_steps,
        num_epochs=num_epochs,
        symmetry_cfg=symmetry_cfg,
    )

    assert _stage_total(stats, "learn_symmetry") == 2 * num_envs * num_steps * num_epochs
    expected_per_env = (2 * num_steps) + 1 + (2 * num_steps * num_epochs) + (2 * num_steps * num_epochs)
    assert stats["per_env"].cpu().tolist() == [expected_per_env] * num_envs
    assert "learn_symmetry:48" in summary


def test_encoder_call_stats_counts_symmetry_inference_with_data_augmentation():
    num_envs = 4
    num_steps = 3
    num_epochs = 2
    symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": _double_batch_augmentation,
        "mirror_loss_coeff": 0.1,
    }
    stats, summary = _run_one_ppo_iteration(
        use_cache=True,
        num_envs=num_envs,
        num_steps=num_steps,
        num_epochs=num_epochs,
        symmetry_cfg=symmetry_cfg,
    )

    assert _stage_total(stats, "learn_actor") == 2 * num_envs * num_steps * num_epochs
    assert _stage_total(stats, "learn_critic") == 2 * num_envs * num_steps * num_epochs
    assert _stage_total(stats, "learn_symmetry") == 2 * num_envs * num_steps * num_epochs
    expected_per_env = (2 * num_steps) + 1 + (6 * num_steps * num_epochs)
    assert stats["per_env"].cpu().tolist() == [expected_per_env] * num_envs
    assert "learn_symmetry:48" in summary

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self._use_cached_features = False
        self._use_cached_preprocessing = False
        self._shared_actor_critic_observations = True
        self._policy_has_cached_feature_helpers = self._has_cached_feature_helpers()
        self._policy_has_preprocessing_cache_helpers = self._has_preprocessing_cache_helpers()
        self._collect_timing_seconds = {
            "encoder": 0.0,
            "actor_critic": 0.0,
            "env_step": 0.0,
            "transfer_normalize": 0.0,
            "process_env_step": 0.0,
            "bookkeeping": 0.0,
        }
        self._track_encoder_feature_calls = bool(getattr(self.policy, "tracks_encoder_feature_calls", False))
        self._encoder_call_stage_names = (
            "collect_cache",
            "collect_actor",
            "collect_critic",
            "compute_returns",
            "learn_actor",
            "learn_critic",
            "learn_symmetry",
        )
        self._encoder_call_counts_by_stage: dict[str, torch.Tensor] | None = None

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

        # Cache non-learned discrete preprocessing independently of trainable
        # encoder weights.  This avoids repeating FPS/KNN for every PPO epoch.
        if (
            bool(getattr(self.policy, "supports_trainable_preprocessing_cache", False))
            and self._policy_has_preprocessing_cache_helpers
            and self._policy_has_cached_feature_helpers
        ):
            with torch.no_grad():
                dummy_obs = torch.zeros(1, actor_obs_shape[0], device=self.device)
                encoder_features, extra_state = self.policy.get_trainable_preprocessing_cache(
                    dummy_obs
                )
                critic_obs_dim = (
                    critic_obs_shape[0]
                    if critic_obs_shape is not None
                    else actor_obs_shape[0]
                )
                dummy_critic_obs = torch.zeros(1, critic_obs_dim, device=self.device)
                critic_encoder_features, critic_extra_state = (
                    self.policy.get_trainable_preprocessing_cache(dummy_critic_obs)
                )
            self.storage.enable_encoder_feature_cache(
                encoder_features_shape=encoder_features.shape[1:],
                extra_state_shape=extra_state.shape[1:],
                critic_encoder_features_shape=critic_encoder_features.shape[1:],
                critic_extra_state_shape=critic_extra_state.shape[1:],
                encoder_features_dtype=encoder_features.dtype,
                extra_state_dtype=extra_state.dtype,
                critic_encoder_features_dtype=critic_encoder_features.dtype,
                critic_extra_state_dtype=critic_extra_state.dtype,
            )
            self._use_cached_preprocessing = True
            self._use_cached_features = False
            print(
                "[PPO] Enabled compact trainable preprocessing cache "
                f"(dtype={encoder_features.dtype}, width={encoder_features.shape[-1]})"
            )
        # Enable complete encoder feature caching if the encoder is frozen.
        elif (
            hasattr(self.policy, 'supports_cached_features')
            and self.policy.supports_cached_features
            and self._policy_has_cached_feature_helpers
        ):
            encoder_is_frozen = bool(
                getattr(self.policy, "freeze_encoder", False)
                or getattr(self.policy, "freeze_point2vec", False)
            )
            if encoder_is_frozen:
                # Get feature dimensions by running dummy forward passes.
                with torch.no_grad():
                    dummy_obs = torch.zeros(1, actor_obs_shape[0], device=self.device)
                    encoder_features, extra_state = self.policy.get_cached_encoder_features(dummy_obs)
                    critic_obs_dim = critic_obs_shape[0] if critic_obs_shape is not None else actor_obs_shape[0]
                    dummy_critic_obs = torch.zeros(1, critic_obs_dim, device=self.device)
                    critic_encoder_features, critic_extra_state = self.policy.get_cached_encoder_features(
                        dummy_critic_obs
                    )
                self.storage.enable_encoder_feature_cache(
                    encoder_features_shape=encoder_features.shape[1:],
                    extra_state_shape=extra_state.shape[1:],
                    critic_encoder_features_shape=critic_encoder_features.shape[1:],
                    critic_extra_state_shape=critic_extra_state.shape[1:],
                    encoder_features_dtype=encoder_features.dtype,
                    extra_state_dtype=extra_state.dtype,
                    critic_encoder_features_dtype=critic_encoder_features.dtype,
                    critic_extra_state_dtype=critic_extra_state.dtype,
                )
                self._use_cached_features = True
                print("[PPO] Enabled encoder feature caching for frozen encoder optimization")
            else:
                self._use_cached_features = False
        else:
            self._use_cached_features = False
            self._use_cached_preprocessing = False
        self._init_encoder_call_stats()

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # Cache fixed search indices, then run the trainable PointNet once for
        # the actor and critic when their observations are shared.
        if self._use_cached_preprocessing:
            with torch.no_grad():
                timing_start = self._collect_timing_start()
                actor_preprocessing, actor_extra_state = (
                    self.policy.get_trainable_preprocessing_cache(obs)
                )
                actor_encoder_features = self.policy.materialize_trainable_preprocessing(
                    obs, actor_preprocessing
                )
                self._collect_timing_stop("encoder", timing_start)
                self._record_encoder_feature_calls("collect_cache")

                observations_are_shared = critic_obs is obs
                self._shared_actor_critic_observations = bool(
                    self._shared_actor_critic_observations and observations_are_shared
                )
                if observations_are_shared:
                    critic_preprocessing = actor_preprocessing
                    critic_extra_state = actor_extra_state
                    critic_encoder_features = actor_encoder_features
                else:
                    timing_start = self._collect_timing_start()
                    critic_preprocessing, critic_extra_state = (
                        self.policy.get_trainable_preprocessing_cache(critic_obs)
                    )
                    critic_encoder_features = (
                        self.policy.materialize_trainable_preprocessing(
                            critic_obs, critic_preprocessing
                        )
                    )
                    self._collect_timing_stop("encoder", timing_start)
                    self._record_encoder_feature_calls("collect_cache")

                self.transition.encoder_features = actor_preprocessing
                self.transition.extra_state = actor_extra_state
                self.transition.critic_encoder_features = critic_preprocessing
                self.transition.critic_extra_state = critic_extra_state

                timing_start = self._collect_timing_start()
                self.transition.actions = self.policy.act_from_cached_features(
                    actor_encoder_features,
                    actor_extra_state,
                ).detach()
                self.transition.values = self.policy.evaluate_from_cached_features(
                    critic_encoder_features,
                    critic_extra_state,
                ).detach()
                self.transition.actions_log_prob = (
                    self.policy.get_actions_log_prob_from_cached_features(
                        self.transition.actions
                    ).detach()
                )
                self.transition.action_mean = self.policy.action_mean.detach()
                self.transition.action_sigma = self.policy.action_std.detach()
                self._collect_timing_stop("actor_critic", timing_start)
        # Compute complete encoder features and cache them if enabled.
        elif self._use_cached_features:
            with torch.no_grad():
                timing_start = self._collect_timing_start()
                actor_encoder_features, actor_extra_state = self.policy.get_cached_encoder_features(obs)
                self._collect_timing_stop("encoder", timing_start)
                self._record_encoder_feature_calls("collect_cache")

                if critic_obs is obs:
                    critic_encoder_features = actor_encoder_features
                    critic_extra_state = actor_extra_state
                else:
                    timing_start = self._collect_timing_start()
                    critic_encoder_features, critic_extra_state = self.policy.get_cached_encoder_features(critic_obs)
                    self._collect_timing_stop("encoder", timing_start)
                    self._record_encoder_feature_calls("collect_cache")

                self.transition.encoder_features = actor_encoder_features
                self.transition.extra_state = actor_extra_state
                self.transition.critic_encoder_features = critic_encoder_features
                self.transition.critic_extra_state = critic_extra_state

            # Use cached features for action/value computation
            timing_start = self._collect_timing_start()
            self.transition.actions = self.policy.act_from_cached_features(
                actor_encoder_features,
                actor_extra_state,
            ).detach()
            self.transition.values = self.policy.evaluate_from_cached_features(
                critic_encoder_features,
                critic_extra_state,
            ).detach()
            self.transition.actions_log_prob = self.policy.get_actions_log_prob_from_cached_features(
                self.transition.actions
            ).detach()
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()
            self._collect_timing_stop("actor_critic", timing_start)
        elif self._policy_has_cached_feature_helpers:
            timing_start = self._collect_timing_start()
            actor_encoder_features, actor_extra_state = self.policy.get_cached_encoder_features(obs)
            self._collect_timing_stop("encoder", timing_start)
            self._record_encoder_feature_calls("collect_actor")

            timing_start = self._collect_timing_start()
            self.transition.actions = self.policy.act_from_cached_features(
                actor_encoder_features,
                actor_extra_state,
            ).detach()
            self._collect_timing_stop("actor_critic", timing_start)

            timing_start = self._collect_timing_start()
            critic_encoder_features, critic_extra_state = self.policy.get_cached_encoder_features(critic_obs)
            self._collect_timing_stop("encoder", timing_start)
            self._record_encoder_feature_calls("collect_critic")

            timing_start = self._collect_timing_start()
            self.transition.values = self.policy.evaluate_from_cached_features(
                critic_encoder_features,
                critic_extra_state,
            ).detach()
            self.transition.actions_log_prob = self.policy.get_actions_log_prob_from_cached_features(
                self.transition.actions
            ).detach()
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()
            self._collect_timing_stop("actor_critic", timing_start)
        else:
            # Standard path (no caching)
            self.transition.actions = self.policy.act(obs).detach()
            self._record_encoder_feature_calls("collect_actor")
            self.transition.values = self.policy.evaluate(critic_obs).detach()
            self._record_encoder_feature_calls("collect_critic")
            self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self._record_encoder_feature_calls("compute_returns")
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                include_env_ids=self._track_encoder_feature_calls,
                include_critic_encoder_features=(
                    self._use_cached_features or self._use_cached_preprocessing
                ),
            )

        # iterate over batches
        for mini_batch in generator:
            env_ids_batch = None
            if len(mini_batch) in (15, 17):
                env_ids_batch = mini_batch[-1]
                mini_batch = mini_batch[:-1]
            critic_encoder_features_batch = None
            critic_extra_state_batch = None
            if len(mini_batch) == 16:
                critic_encoder_features_batch = mini_batch[-2]
                critic_extra_state_batch = mini_batch[-1]
                mini_batch = mini_batch[:-2]

            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
                encoder_features_batch,
                extra_state_batch,
            ) = mini_batch

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry - cached features are no longer valid
                # because observations are transformed
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                # Invalidate cached features when augmentation is applied
                encoder_features_batch = None
                extra_state_batch = None
                critic_encoder_features_batch = None
                critic_extra_state_batch = None
                if env_ids_batch is not None:
                    env_ids_batch = env_ids_batch.repeat(num_aug)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # Use cached features if available (frozen encoder optimization)
            if self._use_cached_preprocessing and encoder_features_batch is not None:
                actor_tokens = self.policy.materialize_trainable_preprocessing(
                    obs_batch, encoder_features_batch
                )
                self.policy.act_from_cached_features(actor_tokens, extra_state_batch)
                self._record_encoder_feature_calls("learn_actor", env_ids_batch)
                actions_log_prob_batch = (
                    self.policy.get_actions_log_prob_from_cached_features(actions_batch)
                )
                if self._shared_actor_critic_observations:
                    critic_tokens = actor_tokens
                    critic_context = extra_state_batch
                else:
                    if critic_encoder_features_batch is None:
                        raise RuntimeError(
                            "critic preprocessing cache is missing for distinct critic observations"
                        )
                    critic_tokens = self.policy.materialize_trainable_preprocessing(
                        critic_obs_batch, critic_encoder_features_batch
                    )
                    critic_context = critic_extra_state_batch
                    self._record_encoder_feature_calls("learn_critic", env_ids_batch)
                value_batch = self.policy.evaluate_from_cached_features(
                    critic_tokens, critic_context
                )
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]
            elif self._use_cached_features and encoder_features_batch is not None:
                # -- actor (using cached encoder features)
                self.policy.act_from_cached_features(encoder_features_batch, extra_state_batch)
                actions_log_prob_batch = self.policy.get_actions_log_prob_from_cached_features(actions_batch)
                # -- critic (using cached encoder features)
                if critic_encoder_features_batch is None:
                    critic_encoder_features_batch = encoder_features_batch
                    critic_extra_state_batch = extra_state_batch
                value_batch = self.policy.evaluate_from_cached_features(
                    critic_encoder_features_batch,
                    critic_extra_state_batch,
                )
                # -- entropy
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]
            else:
                # -- actor (standard path)
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                if env_ids_batch is not None:
                    self._record_encoder_feature_calls("learn_actor", env_ids_batch)
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                # -- critic
                value_batch = self.policy.evaluate(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                if env_ids_batch is not None:
                    self._record_encoder_feature_calls("learn_critic", env_ids_batch)
                # -- entropy
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                symmetry_env_ids_batch = env_ids_batch
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    if env_ids_batch is not None:
                        symmetry_env_ids_batch = env_ids_batch.repeat(num_aug)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                if symmetry_env_ids_batch is not None:
                    self._record_encoder_feature_calls("learn_symmetry", symmetry_env_ids_batch)

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if hasattr(self.policy, "diagnostic_stats"):
            loss_dict.update(self.policy.diagnostic_stats())

        return loss_dict

    def reset_collect_timing(self):
        for key in self._collect_timing_seconds:
            self._collect_timing_seconds[key] = 0.0

    def collect_timing_summary(self, iteration: int, total_time: float | None = None) -> str:
        encoder = self._collect_timing_seconds["encoder"]
        actor_critic = self._collect_timing_seconds["actor_critic"]
        if total_time is None:
            total = encoder + actor_critic
        else:
            total = max(float(total_time), 0.0)
        other = max(total - encoder - actor_critic, 0.0)

        def pct(seconds: float) -> float:
            return 100.0 * seconds / total if total > 0.0 else 0.0

        if self._use_cached_preprocessing:
            cache_status = "preprocessing"
        else:
            cache_status = "on" if self._use_cached_features else "off"
        summary = (
            f"[CollectTiming][rank {self.gpu_global_rank}/{self.gpu_world_size}][cache={cache_status}] "
            f"iter={iteration} total={total:.3f}s "
            f"encoder={encoder:.3f}s ({pct(encoder):.1f}%) "
            f"actor_critic={actor_critic:.3f}s ({pct(actor_critic):.1f}%) "
            f"other={other:.3f}s ({pct(other):.1f}%)"
        )
        other_keys = ("env_step", "transfer_normalize", "process_env_step", "bookkeeping")
        accounted_other = sum(self._collect_timing_seconds[key] for key in other_keys)
        unaccounted_other = max(other - accounted_other, 0.0)
        other_detail = " ".join(
            f"{key}={self._collect_timing_seconds[key]:.3f}s"
            for key in other_keys
        )
        return (
            f"{summary}\n"
            f"[CollectOtherTiming][rank {self.gpu_global_rank}/{self.gpu_world_size}] "
            f"iter={iteration} {other_detail} unaccounted={unaccounted_other:.3f}s"
        )

    def collect_timing_start(self) -> float:
        return self._collect_timing_start()

    def collect_timing_stop(self, key: str, start: float):
        if key not in self._collect_timing_seconds:
            raise KeyError(f"Unknown collect timing key: {key}")
        self._collect_timing_stop(key, start)

    def sync_collect_timing_cuda(self):
        self._collect_timing_sync_cuda()

    def _has_cached_feature_helpers(self) -> bool:
        return all(
            hasattr(self.policy, name)
            for name in (
                "get_cached_encoder_features",
                "act_from_cached_features",
                "evaluate_from_cached_features",
                "get_actions_log_prob_from_cached_features",
            )
        )

    def _has_preprocessing_cache_helpers(self) -> bool:
        return all(
            hasattr(self.policy, name)
            for name in (
                "get_trainable_preprocessing_cache",
                "materialize_trainable_preprocessing",
            )
        )

    def _collect_timing_sync_cuda(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _collect_timing_start(self) -> float:
        self._collect_timing_sync_cuda()
        return time.perf_counter()

    def _collect_timing_stop(self, key: str, start: float):
        self._collect_timing_sync_cuda()
        self._collect_timing_seconds[key] += time.perf_counter() - start

    def _init_encoder_call_stats(self):
        if not self._track_encoder_feature_calls or self.storage is None:
            self._encoder_call_counts_by_stage = None
            return
        self._encoder_call_counts_by_stage = {
            stage: torch.zeros(self.storage.num_envs, dtype=torch.long, device=self.device)
            for stage in self._encoder_call_stage_names
        }

    def reset_encoder_call_stats(self):
        if not self._track_encoder_feature_calls or self.storage is None:
            return
        if (
            self._encoder_call_counts_by_stage is None
            or len(next(iter(self._encoder_call_counts_by_stage.values()))) != self.storage.num_envs
        ):
            self._init_encoder_call_stats()
            return
        for counts in self._encoder_call_counts_by_stage.values():
            counts.zero_()

    def _record_encoder_feature_calls(self, stage: str, env_ids: torch.Tensor | None = None):
        if not self._track_encoder_feature_calls or self._encoder_call_counts_by_stage is None:
            return
        counts = self._encoder_call_counts_by_stage[stage]
        with torch.no_grad():
            if env_ids is None:
                counts.add_(1)
            else:
                env_ids = env_ids.to(device=counts.device, dtype=torch.long).reshape(-1)
                if env_ids.numel() == 0:
                    return
                counts.add_(torch.bincount(env_ids, minlength=self.storage.num_envs)[: self.storage.num_envs])

    def get_encoder_call_stats(self):
        if not self._track_encoder_feature_calls or self._encoder_call_counts_by_stage is None:
            return None
        stage_counts = {
            stage: counts.detach().clone()
            for stage, counts in self._encoder_call_counts_by_stage.items()
        }
        per_env = torch.zeros(self.storage.num_envs, dtype=torch.long, device=self.device)
        for counts in stage_counts.values():
            per_env.add_(counts)
        return {
            "cache_enabled": self._use_cached_features or self._use_cached_preprocessing,
            "cache_mode": (
                "preprocessing"
                if self._use_cached_preprocessing
                else ("on" if self._use_cached_features else "off")
            ),
            "local_envs": self.storage.num_envs,
            "per_env": per_env,
            "stages": stage_counts,
            "total": int(per_env.sum().item()),
            "unique": int((per_env > 0).sum().item()),
        }

    def encoder_call_stats_summary(self, iteration: int) -> str | None:
        stats = self.get_encoder_call_stats()
        if stats is None:
            return None
        per_env = stats["per_env"]
        total = stats["total"]
        local_envs = stats["local_envs"]
        mean = total / local_envs if local_envs else 0.0
        stage_parts = []
        for stage in self._encoder_call_stage_names:
            stage_total = int(stats["stages"][stage].sum().item())
            if stage_total > 0:
                stage_parts.append(f"{stage}:{stage_total}")
        stages = ",".join(stage_parts) if stage_parts else "none"
        cache_status = stats["cache_mode"]
        return (
            f"[EncoderCalls][rank {self.gpu_global_rank}/{self.gpu_world_size}][cache={cache_status}] "
            f"iter={iteration} local_envs={local_envs} total={total} "
            f"per_env=min:{int(per_env.min().item())} mean:{mean:.2f} max:{int(per_env.max().item())} "
            f"stages={stages} unique={stats['unique']}"
        )

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

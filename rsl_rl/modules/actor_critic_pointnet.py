import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
            nn.ELU(),
        )

    def forward(self, x):
        # x: (batch, num_points, input_dim)
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]  # max pooling over points
        return x

class ActorCriticPointNet(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        pointnet_point_dim,
        pointnet_num_points,
        pointnet_output_dim=128,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        fuser_hidden_dims=None,
        **kwargs,
    ):
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        # PointNet for point cloud
        self.pointnet = PointNet(pointnet_point_dim, pointnet_output_dim)
        self.pointnet_num_points = pointnet_num_points
        self.pointnet_output_dim = pointnet_output_dim

        # The rest obs (non-pointcloud)
        self.nonpc_obs_dim = num_actor_obs - (pointnet_point_dim * pointnet_num_points)
        fusion_input_dim = self.nonpc_obs_dim + pointnet_output_dim

        # Fuser MLP for feature fusion (optional)
        self.use_fuser = fuser_hidden_dims is not None and len(fuser_hidden_dims) > 0
        if self.use_fuser:
            fuser_layers = [nn.Linear(fusion_input_dim, fuser_hidden_dims[0]), activation_fn]
            for i in range(len(fuser_hidden_dims) - 1):
                fuser_layers.append(nn.Linear(fuser_hidden_dims[i], fuser_hidden_dims[i + 1]))
                fuser_layers.append(activation_fn)
            self.fuser = nn.Sequential(*fuser_layers)
            mlp_input_dim_a = fuser_hidden_dims[-1]
            mlp_input_dim_c = fuser_hidden_dims[-1]
        else:
            self.fuser = None
            mlp_input_dim_a = fusion_input_dim
            mlp_input_dim_c = fusion_input_dim

        # Policy
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation_fn]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation_fn]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs):
        # obs: (batch, total_obs_dim)
        pc_flat = obs[:, :self.pointnet_num_points * self.pointnet.mlp[0].in_features]
        pc = pc_flat.view(-1, self.pointnet_num_points, self.pointnet.mlp[0].in_features)
        rest = obs[:, self.pointnet_num_points * self.pointnet.mlp[0].in_features:]
        return pc, rest

    def _get_features(self, obs):
        pc, rest = self._split_obs(obs)
        pc_feat = self.pointnet(pc)
        if rest.shape[1] > 0:
            features = torch.cat([pc_feat, rest], dim=-1)
        else:
            features = pc_feat
        if self.use_fuser:
            features = self.fuser(features)
        return features

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations):
        features = self._get_features(observations)
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        features = self._get_features(observations)
        return self.actor(features)

    def evaluate(self, critic_observations, **kwargs):
        features = self._get_features(critic_observations)
        return self.critic(features)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

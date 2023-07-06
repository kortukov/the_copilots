"""Neural network for the Dueling DQN algorithm."""
import numpy as np
import torch
from torch import nn


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration.

    Based on the paper: https://arxiv.org/pdf/1706.10295.pdf (Noisy Networks for Exploration, Fortunato et al. 2017)

    Used version: (a) Independent Gaussian noise
    """

    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        nn.init.uniform_(
            self.weight_mu,
            a=-np.sqrt(3 / in_features),
            b=np.sqrt(3 / in_features),
        )
        nn.init.uniform_(
            self.bias_mu, a=-np.sqrt(3 / in_features), b=np.sqrt(3 / in_features)
        )

        nn.init.constant_(self.weight_sigma, 0.017)  # following original paper
        nn.init.constant_(self.bias_sigma, 0.017)  # following original paper

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(
                self.weight_mu
            )

            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return torch.nn.functional.linear(input, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_layer = nn.Sequential(
            NoisyLinear(self.input_dim[0], 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q = values + (advantages - advantages.mean())

        return q

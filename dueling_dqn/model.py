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

    def __init__(self, in_features, out_features, noisy=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noisy = noisy

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.register_buffer("weight_epsilon", torch.empty(self.out_features, self.in_features))

        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.register_buffer("bias_epsilon", torch.empty(self.out_features))

        nn.init.uniform_(
            self.weight_mu,
            a=-np.sqrt(3 / self.in_features),
            b=np.sqrt(3 / self.in_features),
        )
        nn.init.uniform_(
            self.bias_mu, a=-np.sqrt(3 / self.in_features), b=np.sqrt(3 / self.in_features)
        )

        nn.init.constant_(self.weight_sigma, 0.017)  # following original paper
        nn.init.constant_(self.bias_sigma, 0.017)  # following original paper

    def forward(self, input):
        if self.noisy and self.training:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(
                self.weight_mu
            )

            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return torch.nn.functional.linear(input, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, noisy=True):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim[0]
        self.output_dim = output_dim

        self.feature_layer = nn.Sequential(
            NoisyLinear(self.input_dim, 256, noisy),
            nn.ReLU(),
            NoisyLinear(256, 256, noisy),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(256, 256, noisy), nn.ReLU(), NoisyLinear(256, 1, noisy)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(256, 256, noisy), nn.ReLU(), NoisyLinear(256, self.output_dim, noisy)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q = values + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q

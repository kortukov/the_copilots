import gymnasium as gym
from gymnasium import spaces

import numpy as np

import torch

from wrapper import EnvWrapper


class Args:
    """
    Hyperparameters for the DQN agent.

    Attributes
    ----------
    num_episodes : int
        Total number of episodes for agent training.
    batch_size : int
        Number of experiences to sample from memory during training.
    gamma : float
        Discount factor for future rewards.
    initial_epsilon : float
        Initial value of epsilon for the epsilon-greedy action selection policy.
    final_epsilon : float
        Final value of epsilon for the epsilon-greedy action selection policy.
    decay_rate : float
        Decay rate for epsilon.
    target_update : int
        Frequency (number of steps) for updating the target network.
    replay_memory_size : int
        Size of the replay memory.
    learning_rate : float
        Learning rate for the neural network.
    bins : int
        Number of bins for the prioritized replay buffer.
    episode_length : int
        Maximum length of an episode.
    replay_episodes : int
        Number of episodes to populate the replay memory before training starts.
    prioritize : bool
        Use prioritized experience replay if True.
    prob_alpha : float
        Exponent determining how much prioritization is used.
    beta_start : float
        Initial value of beta for importance-sampling.
    noisy : bool
        Use noisy layers if True.
    """

    def __init__(self, **kwargs):
        self.num_episodes = kwargs.get("num_episodes", 10000)
        self.batch_size = kwargs.get("batch_size", 128)
        self.gamma = kwargs.get("gamma", 0.99)
        self.initial_epsilon = kwargs.get("initial_epsilon", 0.5)
        self.final_epsilon = kwargs.get("final_epsilon", 0.01)
        self.decay_rate = kwargs.get("decay_rate", 0.99)
        self.target_update = kwargs.get("target_update", 10)
        self.replay_memory_size = kwargs.get("replay_memory_size", 100000)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.bins = kwargs.get("bins", 10)
        self.episode_length = kwargs.get("episode_length", 400)
        self.replay_episodes = kwargs.get("replay_episodes", 10)
        self.prioritize = kwargs.get("prioritize", True)
        self.prob_alpha = kwargs.get("prob_alpha", 0.6)
        self.beta_start = kwargs.get("beta_start", 0.4)
        self.noisy = kwargs.get("noisy", True)


class DiscreteActionWrapper(gym.ActionWrapper):
    """A wrapper for converting a 1D continuous actions into discrete ones."""

    def __init__(self, env: gym.Env, bins=5):
        """
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        """
        Converts discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action / (self.bins - 1.0) * (
            self.orig_action_space.high - self.orig_action_space.low
        )


CUSTOM_HOCKEY_ACTIONS = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 1, 0],
    [-1, -1, 0, 0],
    [-1, 1, 0, 0],
    [1, -1, 0, 0],
    [1, 1, 0, 0],
    [-1, -1, -1, 0],
    [-1, -1, 1, 0],
    [-1, 1, -1, 0],
    [-1, 1, 1, 0],
    [1, -1, -1, 0],
    [1, -1, 1, 0],
    [1, 1, -1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, -1],
]


def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_env(env_name, bins):
    """Get the hockey environment."""

    env = EnvWrapper(env_name, bins)
    eval_env = EnvWrapper(env_name, bins, eval=True)

    return env, eval_env

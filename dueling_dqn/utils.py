import gymnasium as gym
from gymnasium import spaces

import random
from collections import deque
import numpy as np

import torch


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
    """

    def __init__(self, **kwargs):
        self.num_episodes = kwargs.get('num_episodes', 10000)
        self.batch_size = kwargs.get('batch_size', 128)
        self.gamma = kwargs.get('gamma', 0.99)
        self.initial_epsilon = kwargs.get('initial_epsilon', 0.5)
        self.final_epsilon = kwargs.get('final_epsilon', 0.01)
        self.decay_rate = kwargs.get('decay_rate', 0.99)
        self.target_update = kwargs.get('target_update', 10)
        self.replay_memory_size = kwargs.get('replay_memory_size', 100000)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.bins = kwargs.get('bins', 10)
        self.episode_length = kwargs.get('episode_length', 400)
        self.replay_episodes = kwargs.get('replay_episodes', 10)
        self.prioritize = kwargs.get('prioritize', True)
        self.prob_alpha = kwargs.get('prob_alpha', 0.6)
        self.beta_start = kwargs.get('beta_start', 0.4)



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


class ReplayBuffer:
    """Implementation of a fixed size Replay Buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Implementation of Prioritized Experience Replay Memory.

    Based on https://arxiv.org/pdf/1511.05952.pdf (Prioritized Experience Replay, Schaul et al. 2015)
    """

    def __init__(self, capacity: int, prob_alpha: float = 0.6, beta: float = 0.4):
        """Initializes the buffer.

        Args:
        - capacity: The maximum capacity of the buffer. Once the buffer is full, older experiences are removed.
        - prob_alpha: Exponent for probabilities. Determines how much prioritization is used.
        - beta: Exponent for the importance sampling weights. Adjusts the bias introduced by this non-uniform sampling.

        """
        self.prob_alpha = prob_alpha
        self.beta = beta
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.pos = 0
        self.priorities = np.zeros(
            (capacity,), dtype=np.float32
        )  # priority values for each experience

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        The maximum priority is assigned to the new experience so that it has a chance to be sampled in the next update.

        Args:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The state that resulted from the action.
        - done: A boolean indicating whether the episode ended.
        """
        assert state.ndim == next_state.ndim
        max_prio = (
            self.priorities.max() if self.buffer else 1.0
        )  # max priority for the new experience

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                (state, action, reward, next_state, done)
            )  # add new experience to the buffer
        else:
            self.buffer[self.pos] = (
                state,
                action,
                reward,
                next_state,
                done,
            )  # overwrite old experience if the buffer is full

        self.priorities[
            self.pos
        ] = max_prio  # assign max priority to the new experience
        self.pos = (self.pos + 1) % self.capacity  # increment the position index

    def sample(self, batch_size: int, beta: float = None):
        """Sample a batch of experiences from memory.

        Sampling is done based on the priorities of the experiences.

        Args:
        - batch_size: The number of experiences to sample.
        - beta: Exponent for the importance sampling weights. Adjusts the bias introduced by this non-uniform sampling.

        Returns:
        - samples: The sampled experiences.
        - indices: The indices of the sampled experiences.
        - weights: The importance sampling weights for the sampled experiences.
        """
        if beta is None:
            beta = self.beta

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        # compute sampling probabilities from the priorities
        probs = prios**self.prob_alpha
        probs /= probs.sum()

        # sample indices based on the probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # compute importance sampling weights from the probabilities
        total = len(self.buffer)
        weights = (total * probs[indices]) ** -beta
        weights /= weights.max()  # normalize the weights

        # separate the experiences into their components
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            (
                np.array(states),
                actions,
                rewards,
                np.array(next_states),
                dones,
            ),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities for sampled transitions.

        After the agent learns from a batch of experiences, the priorities in the buffer should be updated.
        The TD errors from the learning step give a measure of how surprising or unexpected the experiences were,
        so they are used to update the priorities.

        Args:
        - batch_indices: The indices of the experiences that were used for learning.
        - batch_priorities: The new priority values for these experiences, typically based on the TD errors.
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio  # update the priority for each experience

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

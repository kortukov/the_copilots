import random
import gym
from gym import spaces


class Args:
    """Hyperparameters for the DQN agent."""

    def __init__(self):
        self.num_episodes = 1000
        self.batch_size = 64
        self.gamma = 0.99
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.decay_rate = 0.0001
        self.target_update = 10
        self.replay_memory_size = 10000
        self.clip = 1.0
        self.alpha = 0.95
        self.epsilon = 0.01
        self.learning_rate = 0.0005


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins=5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
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
        """discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action / (self.bins - 1.0) * (
            self.orig_action_space.high - self.orig_action_space.low
        )


class Transition:
    """Transition object that represents a transition in the environment."""

    def __init__(self, state, action, reward, next_state, done):
        """Initialize a Transition object."""
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayMemory:
    """Replay memory that stores the transitions."""

    def __init__(self, capacity):
        """Initialize a ReplayMemory object."""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Returns a batch of samples."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the length of the memory."""
        return len(self.memory)

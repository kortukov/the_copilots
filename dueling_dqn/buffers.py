import random
from collections import deque

import numpy as np
import numpy_ringbuffer as rb


class ReplayBuffer:
    """Implementation of a fixed size Replay Buffer."""

    def __init__(self, capacity: int):
        # self.buffer = deque(maxlen=capacity)
        self.buffer = rb.RingBuffer(capacity, dtype=object)

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
        # self.buffer = deque(maxlen=capacity)
        self.buffer = rb.RingBuffer(capacity, dtype=object)
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

        self.buffer.append(
            np.array((state, action, reward, next_state, done), dtype=object)
        )  # add new experience to the buffer
        self.pos = len(self.buffer) - 1

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
        samples = self.buffer[indices]

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

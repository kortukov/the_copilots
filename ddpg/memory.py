from collections import deque
import numpy as np
import numpy_ringbuffer as rb
import random

# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        # self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        # This is much faster than np.random.choice
        self.inds=random.sample(range(self.size), k=batch)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]

class ReplayBuffer:
    """Implementation of a fixed size Replay Buffer."""

    def __init__(self, capacity: int):
        # self.buffer = deque(maxlen=capacity)
        self.buffer = rb.RingBuffer(capacity, dtype=object)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        # self.buffer.append((state, action, reward, next_state, done))
        self.buffer.append(np.array((state, action, reward, next_state, done), dtype=object))

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        batch = np.concatenate(state), np.array(action), np.array(reward), np.concatenate(next_state), np.array(done)
        indices, weights = None, None # no need for these in non-prioritized replay buffer
        return batch, indices, weights 

    def update_priorities(self, batch_indices, batch_priorities):
        """For compatibility with prioritized replay buffer."""
        pass

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

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                np.array((state, action, reward, next_state, done), dtype=object)
            )  # add new experience to the buffer
        else:
            self.buffer[self.pos] = np.array(
                (state, action, reward, next_state, done), dtype=object
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
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
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
        
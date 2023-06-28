import math
import random

import torch
from torch import nn
from torch import optim

from model import DuelingDQN
from utils import Transition, ReplayMemory


class Agent:
    """Agent class that interacts with and learns from the environment."""

    def __init__(self, env, args):
        """Initialize an Agent object."""
        self.env = env
        self.args = args
        print(self.env.action_space)
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main network.
        self.model = DuelingDQN(self.num_actions).to(self.device)
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.args.learning_rate,
            alpha=self.args.alpha,
            eps=self.args.epsilon,
        )

        # Target network.
        self.target_model = DuelingDQN(self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Replay memory.
        self.memory = ReplayMemory(self.args.replay_memory_size)

        # Number of training iterations.
        self.num_iterations = 0

        # Epsilon for epsilon greedy policy.
        self.epsilon = self.args.initial_epsilon

        # Loss function.
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """Select an action from the input state."""
        if random.random() <= self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.model(state).max(1)[1].item()
        return action

    def get_state_action_values(self, state_batch, action_batch):
        return self.model(state_batch).gather(1, action_batch.unsqueeze(1))

    def get_next_state_values(self, non_final_next_states, non_final_mask):
        next_state_values = torch.zeros(self.args.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_model(non_final_next_states).max(1)[0].detach()
        )
        return next_state_values

    def get_expected_state_action_values(self, next_state_values, reward_batch):
        return (next_state_values * self.args.gamma) + reward_batch

    def optimize_model(self):
        if len(self.memory) < self.args.batch_size:
            return

        transitions = self.memory.sample(self.args.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [
                torch.FloatTensor(s).to(self.device)
                for s in batch.next_state
                if s is not None
            ]
        )
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)

        state_action_values = self.get_state_action_values(state_batch, action_batch)
        next_state_values = self.get_next_state_values(
            non_final_next_states, non_final_mask
        )
        expected_state_action_values = self.get_expected_state_action_values(
            next_state_values, reward_batch
        )

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-self.args.clip, self.args.clip)
        self.optimizer.step()

    def update_target_network(self):
        """Update the target network."""
        if self.num_iterations % self.args.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """Update the epsilon value."""
        self.epsilon = self.args.final_epsilon + (
            (self.args.initial_epsilon - self.args.final_epsilon)
            * math.exp(-1.0 * self.num_iterations / self.args.decay_rate)
        )
        self.num_iterations += 1

    def train(self):
        """Train the agent."""
        # Initialize the environment and state.
        state = self.env.reset()

        # Loop over episodes.
        for episode in range(self.args.num_episodes):
            # Initialize the episode.
            total_reward = 0.0
            state = self.env.reset()
            done = False

            # Loop over steps.
            while not done:
                # Select an action.
                action = self.select_action(state)

                # Execute the action.
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Store the transition in memory.
                self.memory.push(state, action, reward, next_state, done)

                # Update the state.
                state = next_state

                # Perform one step of the optimization (on the target network).
                self.optimize_model()

                # Update the target network, it's better to update after some episodes.
                if episode % self.args.target_update == 0:
                    self.update_target_network()

                # Update the epsilon value.
                self.update_epsilon()

                # Print the total reward of the episode.
                print("Episode: {}, total reward: {}".format(episode + 1, total_reward))

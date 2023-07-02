import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import DuelingDQN
from utils import DiscreteActionWrapper
from utils import ReplayBuffer

import numpy as np


class Agent:
    def __init__(self, env_name, args):
        self.env_name = env_name

        self.env = gym.make(env_name)
        self.env = DiscreteActionWrapper(self.env, args.bins)

        self.eval_env = gym.make(env_name, render_mode="human")
        self.eval_env = DiscreteActionWrapper(self.eval_env, args.bins)

        self.input_dim = self.env.observation_space.shape
        self.output_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(self.input_dim, self.output_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.MSE_loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(args.replay_memory_size)

        self.gamma = args.gamma
        self.batch_size = args.batch_size

        self.eps_start = args.initial_epsilon
        self.eps_end = args.final_epsilon
        self.eps_decay = args.decay_rate

        self.target_update = args.target_update
        self.num_episodes = args.num_episodes
        self.episode_length = args.episode_length

        self.total_steps = 0

    def decay_epsilon(self):
        self.eps_start = max(self.eps_end, self.eps_start * self.eps_decay)

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state, eps=None):
        if eps is None:
            eps = self.eps_start

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = self.model(state).argmax().item()
        else:
            action = self.env.action_space.sample()

        return action

    def plot_rewards(self, rewards, filename="rewards.png"):
        plt.figure(figsize=(12, 8))
        plt.plot(rewards)
        plt.title("Reward per episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def save_checkpoint(self, filename):
        checkpoint = {
            "env_name": self.env.spec.id,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.env = gym.make(checkpoint["env_name"])
        print(f"Checkpoint loaded from {filename}")

    def evaluate(self, test_episodes=2):
        self.model.eval()
        for episode in range(test_episodes):
            state, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            counter = 0

            while not done:
                action = (
                    self.model(torch.FloatTensor(state).to(self.device)).argmax().item()
                )
                next_state, reward, done, _, _ = self.eval_env.step(action)
                total_reward += reward
                state = next_state
                if counter == self.episode_length:
                    break
                counter += 1

    def train(self):
        self.model.train()

        episodes_rewards = []
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            counter = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(self.replay_buffer) > self.batch_size:
                    self.update(self.batch_size)

                self.total_steps += 1
                if self.total_steps % self.target_update == 0:
                    self.model.load_state_dict(self.model.state_dict())

                if counter == self.episode_length:
                    break
                counter += 1

            print(f"Reward for current episode {episode}: ", total_reward)
            print("Epsilon: ", self.eps_start)
            episodes_rewards.append(total_reward)
            if episode % 100 == 0:
                self.save_checkpoint(f"checkpoint_{episode}_{self.env_name}.pth")
                self.evaluate()

            self.decay_epsilon()
            self.plot_rewards(episodes_rewards)

        print("Training completed.")

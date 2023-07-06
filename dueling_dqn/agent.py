import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils.save_video import save_video

import plots
from model import DuelingDQN
from utils import DiscreteActionWrapper
from utils import ReplayBuffer, PrioritizedReplayBuffer


class Agent:
    def __init__(self, env_name, args):
        self.env_name = env_name

        self.env = gym.make(env_name)
        self.env = DiscreteActionWrapper(self.env, args.bins)

        self.eval_env = gym.make(env_name, render_mode="rgb_array_list")
        self.eval_env = DiscreteActionWrapper(self.eval_env, args.bins)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(
            self.env.observation_space.shape, self.env.action_space.n
        ).to(self.device)
        self.target_model = DuelingDQN(
            self.env.observation_space.shape, self.env.action_space.n
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.MSE_loss = nn.MSELoss()

        self.gamma = args.gamma
        self.batch_size = args.batch_size

        self.eps_start = args.initial_epsilon
        self.eps_end = args.final_epsilon
        self.eps_decay = args.decay_rate

        self.target_update = args.target_update
        self.replay_episodes = args.replay_episodes

        self.num_episodes = args.num_episodes
        self.episode_length = args.episode_length
        self.episode_continue = None
        self.prioritize = args.prioritize

        if self.prioritize:
            self.replay_buffer = PrioritizedReplayBuffer(
                args.replay_memory_size,
                prob_alpha=args.prob_alpha,
                beta=args.beta_start,
            )
        else:
            self.replay_buffer = ReplayBuffer(args.replay_memory_size)

    def decay_epsilon(self):
        self.eps_start = max(self.eps_end, self.eps_start * self.eps_decay)

    def compute_loss(self, batch, weights=None):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.model.forward(next_states).argmax(dim=1)
        next_Q = (
            self.target_model.forward(next_states)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )
        expected_Q = rewards + (1.0 - dones) * self.gamma * next_Q

        # If weights are not provided (i.e., when using the simple replay buffer)
        if weights is None:
            loss = self.MSE_loss(curr_Q, expected_Q)
            return loss, None  # Here we return None for the TD errors

        # Compute TD errors for updating priorities
        td_errors = torch.abs(curr_Q - expected_Q).detach().cpu().numpy()

        # Compute the weighted loss
        weights = torch.FloatTensor(weights).to(self.device)
        loss = (weights * self.MSE_loss(curr_Q, expected_Q)).mean()

        return loss, td_errors

    def update(self, batch_size, prioritize=False):
        if prioritize:
            batch, indices, weights = self.replay_buffer.sample(batch_size)
        else:
            batch = self.replay_buffer.sample(batch_size)
            indices, weights = None, None

        loss, td_errors = self.compute_loss(batch, weights)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in the buffer
        if prioritize:
            self.replay_buffer.update_priorities(indices, td_errors)

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

    def save_checkpoint(self, filename, episode):
        folder = f"checkpoints/{self.env_name}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{filename}"
        checkpoint = {
            "env_name": self.env.spec.id,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": self.eps_start,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.target_model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.eps_start = checkpoint["epsilon"]
        self.episode_continue = checkpoint["episode"]
        print(f"Checkpoint loaded from {filename}")

    def evaluate(self, episode_n, test_episodes=2):
        self.model.eval()
        for episode in range(test_episodes):
            state, _ = self.eval_env.reset()
            total_reward = 0
            episode_length_counter = 0

            while True:
                action = (
                    self.model(torch.FloatTensor(state).to(self.device)).argmax().item()
                )
                next_state, reward, done, trunk, info = self.eval_env.step(action)
                total_reward += reward
                state = next_state
                if (episode_length_counter == self.episode_length) or done or trunk:
                    os.makedirs(f"videos/{self.env_name}", exist_ok=True)
                    save_video(
                        self.eval_env.render(),
                        f"videos/{self.env_name}",
                        name_prefix=f"{self.env_name}_eval_{episode_n}",
                        fps=self.eval_env.metadata["render_fps"],
                        step_starting_index=episode_length_counter,
                        episode_index=episode,
                    )
                    break
                episode_length_counter += 1
        self.model.train()

    def replay(self, replay_episodes=5):
        for _ in range(replay_episodes):
            self.update(self.batch_size, self.prioritize)

    def train(self):
        episodes_rewards = []
        times = []
        if self.episode_continue is None:
            start_episode = 0
        else:
            start_episode = self.episode_continue + 1

        for episode in range(start_episode, self.num_episodes):
            start_time = time.time()
            state, _ = self.env.reset()
            total_reward = 0
            episode_length_counter = 0
            while True:
                action = self.act(state)
                next_state, reward, done, trunk, info = self.env.step(action)
                total_reward += reward
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if (episode_length_counter == self.episode_length) or done or trunk:
                    break
                episode_length_counter += 1

            self.replay(self.replay_episodes)
            self.decay_epsilon()

            if episode % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            end_time = time.time()  # End timer after the episode
            episode_duration = end_time - start_time
            times.append(episode_duration)

            print(f"Reward for current episode {episode}: ", total_reward)
            print("Epsilon: ", self.eps_start)

            episodes_rewards.append(total_reward)

            if episode % 500 == 0:
                self.save_checkpoint(
                    f"checkpoint_{episode}_{self.env_name}.pth", episode
                )
                self.evaluate(episode)

            if episode % 50 == 0:
                plots.plot_rewards(episodes_rewards, path=f"plots/{self.env_name}")
                plots.plot_episode_duration(times, path=f"plots/{self.env_name}")

        print("Training completed.")

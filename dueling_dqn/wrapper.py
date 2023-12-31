import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video
from laserhockey import hockey_env

import shared_constants
from ddpg.TD3 import TD3Agent
from .utils import DiscreteActionWrapper, CUSTOM_HOCKEY_ACTIONS, load_hockey_args

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from shared_utils import save_frames_as_gif

import torch


def calculate_rewards(observation):
    # Unpack observation values
    player1_pos = np.array([observation[0], observation[1]])
    player1_angle = observation[2]
    player1_vel = np.array([observation[3], observation[4]])

    puck_pos = np.array([observation[12], observation[13]])
    puck_vel = np.array([observation[14], observation[15]])

    puck_possession_time_player1 = observation[16]
    puck_possession_time_player2 = observation[17]

    # Initialize reward
    reward = 0

    # Reward for puck possession time
    reward += puck_possession_time_player1 - puck_possession_time_player2

    # Reward for a puck direction towards opponent's goal
    if puck_vel[0] > 0:
        reward += 1

    # Reward for the puck being in the opponent's half
    if puck_pos[0] > 0:
        reward += 1

    # Negative reward for distance to the puck
    reward -= np.linalg.norm(player1_pos - puck_pos)

    # Reward for agent speed towards puck
    vec_to_puck = puck_pos - player1_pos
    if np.dot(player1_vel, vec_to_puck) < 0:
        reward += 1

    # Reward for facing towards the puck
    direction_to_puck = np.arctan2(vec_to_puck[1], vec_to_puck[0])
    angle_diff = player1_angle - direction_to_puck
    reward += np.cos(angle_diff)

    return reward


class AdaptedHockeyEnv(hockey_env.HockeyEnv):
    def reset(self, *args, **kwargs):
        acceptable_keys = {"one_starting", "mode"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in acceptable_keys}

        return super().reset(*args)

    def render(self, *args, **kwargs):
        return super().render(mode="rgb_array")


class SelfPlayEnv(AdaptedHockeyEnv):
    def __init__(self, model_agent_path, agent, mode=hockey_env.HockeyEnv.NORMAL):
        super().__init__(mode=mode)
        self.normal_opponent = hockey_env.BasicOpponent(weak=False)
        self.weak_opponent = hockey_env.BasicOpponent(weak=True)
        self.model_agent = self.load_model_agent(
            model_agent_path, agent
        )  # Implement this function to load model agent
        self.td3_agent = self.load_td3_agent("ddpg/resulting_models/strongest_td3.pth")
        self.losses_weak = 11
        self.losses_normal = 1
        self.losses_model = 1
        self.losses_td3 = 1
        self.total_games = 14
        self.total_games_temp = 14
        self.current_opponent = None

    def load_model_agent(self, model_agent_path, agent):
        hockey_args = load_hockey_args()
        loaded_agent = agent("HockeyNormal", hockey_args)
        loaded_agent.load_checkpoint(model_agent_path, only_network=True)
        loaded_agent._act = loaded_agent.act
        loaded_agent.act = lambda obs: CUSTOM_HOCKEY_ACTIONS[
            loaded_agent._act(obs, eps=0.0)
        ]

        return loaded_agent

    def load_td3_agent(self, model_agent_path):
        env = hockey_env.HockeyEnv(mode=hockey_env.HockeyEnv.NORMAL)
        agent1 = TD3Agent(
            env.observation_space,
            env.action_space,
            **shared_constants.DEFAULT_TD3_PARAMS,
        )
        agent1_state = torch.load(model_agent_path)
        agent1.restore_state(agent1_state)

        return agent1

    def act(self, obs):
        if self.total_games > self.total_games_temp or self.current_opponent is None:
            p = np.array(
                [
                    np.log(self.losses_weak + 1),
                    np.log(self.losses_normal + 1),
                    np.log(self.losses_model + 1),
                    np.log(self.losses_td3 + 1),
                ]
            ) / (
                np.sum(
                    np.array(
                        [
                            np.log(self.losses_weak + 1),
                            np.log(self.losses_normal + 1),
                            np.log(self.losses_model + 1),
                            np.log(self.losses_td3 + 1),
                        ]
                    )
                )
            )
            self.current_opponent = np.random.choice(
                [
                    self.weak_opponent,
                    self.normal_opponent,
                    self.model_agent,
                    self.td3_agent,
                ],
                size=1,
                p=p,
            )[0]
            self.total_games_temp = self.total_games

        return self.current_opponent.act(obs)

    def step(self, action):
        next_state, reward, done, trunk, info = super().step(action)
        if done:
            if info["winner"] != 1:
                self.total_games += 1
                if self.current_opponent == self.weak_opponent:
                    self.losses_weak += 1
                elif self.current_opponent == self.normal_opponent:
                    self.losses_normal += 1
                elif self.current_opponent == self.model_agent:
                    self.losses_model += 1
                else:
                    self.losses_td3 += 1
            print(
                f"Total games: {self.total_games}, losses weak: {self.losses_weak}, "
                f"losses normal: {self.losses_normal}, losses model: {self.losses_model}, losses td3: {self.losses_td3}"
            )
        return next_state, reward, done, trunk, info


class EnvWrapper:
    def __init__(self, env_name, bins, eval=False, **kwargs):
        self.env_name = env_name
        self.bins = bins

        if "Hockey" in env_name:
            self.touching_puck = 0
            if env_name == "HockeyTrainDefense":
                self.env = AdaptedHockeyEnv(mode=hockey_env.HockeyEnv.TRAIN_DEFENSE)
            elif env_name == "HockeyTrainShooting":
                self.env = AdaptedHockeyEnv(mode=hockey_env.HockeyEnv.TRAIN_SHOOTING)
            elif env_name == "HockeyNormal":
                self.env = AdaptedHockeyEnv(mode=hockey_env.HockeyEnv.NORMAL)
                self.player2 = hockey_env.BasicOpponent(weak=False)
            elif env_name == "HockeyWeak":
                self.env = AdaptedHockeyEnv(mode=hockey_env.HockeyEnv.NORMAL)
                self.player2 = hockey_env.BasicOpponent(weak=True)
            elif env_name == "HockeySelfPlay":
                self.env = SelfPlayEnv(
                    "dueling_dqn/resulting_models/checkpoint_29750_HockeySelfPlay.pth",
                    kwargs["agent"],
                )
                self.player2 = self.env
        else:
            if "Pendulum" in env_name:
                env_name = "Pendulum-v1"
            elif "HalfCheetah" in env_name:
                env_name = "HalfCheetah-v4"

            if eval:
                self.env = DiscreteActionWrapper(
                    gym.make(env_name, render_mode="rgb_array_list"), self.bins
                )
            else:
                self.env = DiscreteActionWrapper(gym.make(env_name), bins=self.bins)
        self.frames = []
        self.episode_step = 0

    @property
    def n(self):
        if "Hockey" not in self.env_name:
            return self.env.action_space.n
        else:
            return len(CUSTOM_HOCKEY_ACTIONS)

    def sample_action(self):
        if "Hockey" not in self.env_name:
            return self.env.action_space.sample()
        else:
            return np.random.randint(len(CUSTOM_HOCKEY_ACTIONS))

    def seed(self, seed):
        if "Hockey" not in self.env_name:
            return self.env.action_space.seed(seed)
        else:
            self.env.seed(seed)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def metadata(self):
        return self.env.metadata

    def reset(self):
        self.episode_step = 0
        return self.env.reset()

    def step(self, action, eval=False):
        if "Hockey" not in self.env_name:
            next_state, reward, done, trunk, info = self.env.step(action)
        elif not eval:
            cont_action = CUSTOM_HOCKEY_ACTIONS[action]
            if self.env_name in ["HockeyTrainDefense", "HockeyTrainShooting"]:
                cont_action = np.hstack([cont_action, [0, 0, 0, 0]])
            else:
                obs_p2 = self.env.obs_agent_two()
                cont_action = np.hstack([cont_action, self.player2.act(obs_p2)])
            next_state, reward, done, trunk, info = self.env.step(cont_action)
            winner = info["winner"]
            if winner == 0:
                if done:
                    winner = -0.3  # encourage winning
                else:
                    winner = -0.002 * self.episode_step  # pushing to play aggressively
            reward = winner * 100
            reward += calculate_rewards(next_state)
            self.episode_step += 1
        else:
            cont_action = CUSTOM_HOCKEY_ACTIONS[action]
            if self.env_name in ["HockeyTrainDefense", "HockeyTrainShooting"]:
                cont_action = np.hstack([cont_action, [0, 0, 0, 0]])
            else:
                obs_p2 = self.env.obs_agent_two()
                cont_action = np.hstack([cont_action, self.player2.act(obs_p2)])
            next_state, reward, done, trunk, info = self.env.step(cont_action)
        return next_state, reward, done, trunk, info

    def render(self):
        return self.env.render()

    def make_video(self, end, episode_length_counter, episode_n, seed):
        if "Hockey" not in self.env_name:
            if end:
                os.makedirs(f"videos/{self.env_name}", exist_ok=True)
                save_video(
                    self.render(),
                    f"videos/{self.env_name}",
                    name_prefix=f"{self.env_name}_eval_{episode_n}_seed_{seed}",
                    fps=self.metadata["render_fps"],
                    step_starting_index=episode_length_counter,
                    episode_index=0,
                )
        else:
            frame = self.render()
            self.frames.append(frame)
            if end:
                os.makedirs(f"videos/{self.env_name}", exist_ok=True)
                gif_filename = f"videos/{self.env_name}/{self.env_name}_eval_{episode_n}_seed_{seed}.gif"
                save_frames_as_gif(self.frames, path="", filename=gif_filename)
                self.frames = []


def get_env(env_name, bins, **kwargs):
    """Get the hockey environment."""

    env = EnvWrapper(env_name, bins, **kwargs)
    eval_env = EnvWrapper(env_name, bins, eval=True, **kwargs)

    return env, eval_env

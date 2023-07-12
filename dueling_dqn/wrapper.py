import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video
from laserhockey import hockey_env

import utils
from shared_utils import save_frames_as_gif


def calculate_rewards(observation):
    # Unpack observation values
    player1_pos = np.array([observation[0], observation[1]])
    player1_angle = observation[2]
    player1_vel = np.array([observation[3], observation[4]])
    player1_angular_vel = observation[5]

    player2_pos = np.array([observation[6], observation[7]])
    player2_angle = observation[8]
    player2_vel = np.array([observation[9], observation[10]])
    player2_angular_vel = observation[11]

    puck_pos = np.array([observation[12], observation[13]])
    puck_vel = np.array([observation[14], observation[15]])

    puck_possession_time_player1 = observation[16]
    puck_possession_time_player2 = observation[17]

    # Initialize reward
    reward = 0

    # Reward for puck possession time
    reward += puck_possession_time_player1 - puck_possession_time_player2

    # Reward for puck direction towards opponent's goal
    if puck_vel[0] > 0:
        reward += 1

    # Reward for puck being in the opponent's half
    if puck_pos[0] > 0:
        reward += 1

    # Negative reward for distance to the puck
    reward -= np.linalg.norm(player1_pos - puck_pos)

    # Reward for agent speed towards puck
    vec_to_puck = puck_pos - player1_pos
    if np.dot(player1_vel, vec_to_puck) < 0:
        reward += 1

    # Negative reward for high player speed (energy conservation)
    reward -= np.linalg.norm(player1_vel)

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


class EnvWrapper:
    def __init__(self, env_name, bins, eval=False):
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
        else:
            if eval:
                self.env = utils.DiscreteActionWrapper(
                    gym.make(env_name, render_mode="rgb_array_list"), self.bins
                )
            else:
                self.env = utils.DiscreteActionWrapper(
                    gym.make(env_name), bins=self.bins
                )
        self.frames = []
        self.episode_step = 0

    @property
    def n(self):
        if "Hockey" not in self.env_name:
            return self.env.action_space.n
        else:
            return len(utils.CUSTOM_HOCKEY_ACTIONS)

    def sample_action(self):
        if "Hockey" not in self.env_name:
            return self.env.action_space.sample()
        else:
            return np.random.randint(len(utils.CUSTOM_HOCKEY_ACTIONS))

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
            cont_action = utils.CUSTOM_HOCKEY_ACTIONS[action]
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
                    winner = -0.001 * self.episode_step  # pushing to play aggressively
            reward = winner * 100 + calculate_rewards(next_state)
            self.episode_step += 1
        else:
            cont_action = utils.CUSTOM_HOCKEY_ACTIONS[action]
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

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

plt.style.use("ggplot")  # Use 'ggplot' style for more appealing visuals
sns.set_context(
    "poster"
)  # Set seaborn context for larger, more visually appealing plots


def plot_rewards(rewards, path="plots", filename="rewards.svg", window_size=100):
    """
    Plot episode rewards, running average, and standard deviation interval.

    Args:
        rewards (list): List of rewards received per episode.
        path (str): Path to the directory where the plot will be saved.
        filename (str): Name of the file where the plot will be saved.
        window_size (int): Size of the sliding window for calculating the
            running average and standard deviation.
    """
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, filename)

    plt.figure(figsize=(16, 10))

    rewards_series = pd.Series(rewards)
    rewards_rolling = rewards_series.rolling(window_size, min_periods=1).mean()
    rewards_std = rewards_series.rolling(window_size, min_periods=1).std()

    plt.plot(rewards, label="Episode Rewards", alpha=0.6, color="dodgerblue")
    plt.plot(rewards_rolling, label="Running Average", color="darkorange", linewidth=3)
    plt.fill_between(
        range(len(rewards)),
        rewards_rolling - rewards_std,
        rewards_rolling + rewards_std,
        color="darkorange",
        alpha=0.2,
        label="Standard Deviation",
    )

    plt.title("Reward per episode", fontsize=24, fontweight="bold", color="dimgrey")
    plt.xlabel("Episode", fontsize=18, color="dimgrey")
    plt.ylabel("Total Reward", fontsize=18, color="dimgrey")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlim(1)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()


def plot_episode_duration(times, path="plots", filename="times.svg", window_size=100):
    """
    Plot episode durations, running average, and standard deviation interval.

    Args:
        times (list): List of time durations per episode.
        path (str): Path to the directory where the plot will be saved.
        filename (str): Name of the file where the plot will be saved.
        window_size (int): Size of the sliding window for calculating the
            running average and standard deviation.
    """
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, filename)

    plt.figure(figsize=(16, 10))

    times_series = pd.Series(times)
    times_rolling = times_series.rolling(window_size, min_periods=1).mean()
    times_std = times_series.rolling(window_size, min_periods=1).std()

    plt.plot(times, label="Episode Duration", alpha=0.6, color="dodgerblue")
    plt.plot(times_rolling, label="Running Average", color="darkorange", linewidth=3)
    plt.fill_between(
        range(len(times)),
        times_rolling - times_std,
        times_rolling + times_std,
        color="darkorange",
        alpha=0.2,
        label="Standard Deviation",
    )

    plt.title(
        "Time Duration per episode", fontsize=24, fontweight="bold", color="dimgrey"
    )
    plt.xlabel("Episode", fontsize=18, color="dimgrey")
    plt.ylabel("Time Duration", fontsize=18, color="dimgrey")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlim(1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()


def plot_hockey(wins, losses, ties, path="plots"):
    """
    Plot normalized cumulative wins, losses, and ties.

    Args:
        wins (list): List of wins over time.
        losses (list): List of losses over time.
        ties (list): List of ties over time.
        path (str): Path to the directory where the plot will be saved.
    """
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, "hockey_plot.svg")

    plt.figure(figsize=(16, 10))

    wins = np.cumsum(np.array(wins))
    losses = np.cumsum(np.array(losses))
    ties = np.cumsum(np.array(ties))

    total_games = wins + losses + ties

    plt.plot(wins / total_games, label="Wins", alpha=0.6, color="dodgerblue")
    plt.plot(losses / total_games, label="Losses", color="darkorange", alpha=0.6)
    plt.plot(ties / total_games, label="Ties", color="grey", alpha=0.6)

    plt.title(
        "Normalized Cumulative Wins, Losses, and Ties Over Time",
        fontsize=24,
        fontweight="bold",
        color="dimgrey",
    )
    plt.xlabel("Game", fontsize=18, color="dimgrey")
    plt.ylabel("Normalized Cumulative Results", fontsize=18, color="dimgrey")
    plt.legend(loc="upper left", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()


def dump_array(rewards, path, filename="rewards.pkl"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, filename)
    with open(filename, "wb") as f:
        pickle.dump(rewards, f)


def load_rewards(filename="rewards.pkl"):
    with open(filename, "rb") as f:
        rewards = pickle.load(f)
    return rewards

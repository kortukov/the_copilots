import pandas as pd
import matplotlib.pyplot as plt


def plot_rewards(rewards, filename="rewards.png", window_size=100):
    plt.figure(figsize=(12, 8))

    # Calculate running mean and standard deviation
    rewards_series = pd.Series(rewards)
    rewards_rolling = rewards_series.rolling(window_size).mean()
    rewards_std = rewards_series.rolling(window_size).std()

    plt.plot(rewards, label="Episode Rewards", alpha=0.5)
    plt.plot(rewards_rolling, label="Running Average", color="orange", linewidth=3)
    plt.fill_between(
        range(len(rewards)),
        rewards_rolling - rewards_std,
        rewards_rolling + rewards_std,
        color="orange",
        alpha=0.2,
    )
    plt.xlim(0)

    plt.title("Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

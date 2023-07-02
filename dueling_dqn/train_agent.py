import gymnasium as gym
from agent import Agent
from utils import Args, DiscreteActionWrapper


def train_agent():
    # Define hyperparameters
    args = Args()  # Replace Args() with your own hyperparameters setting method
    args.num_episodes = 1000
    args.batch_size = 64
    args.gamma = 0.99
    args.initial_epsilon = 0.2
    args.final_epsilon = 0.01
    args.decay_rate = 0.99
    args.target_update = 10
    args.replay_memory_size = 10000
    args.clip = 1.0
    args.alpha = 0.95
    args.learning_rate = 0.01
    args.bins = 30
    args.episode_length = 300

    # Initialize the environment and agent
    env_name = "Pendulum-v1"
    agent = Agent(env_name, args)

    agent.train()

    print("Training completed.")


if __name__ == "__main__":
    train_agent()

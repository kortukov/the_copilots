from agent import Agent
import utils


def train_agent():
    # Define hyperparameters
    args = utils.Args()  # Replace Args() with your own hyperparameters setting method
    args.num_episodes = 2000
    args.batch_size = 64
    args.gamma = 0.99
    args.initial_epsilon = 0.5
    args.final_epsilon = 0.01
    args.decay_rate = 0.99
    args.target_update = 10
    args.replay_memory_size = 50000
    args.clip = 1.0
    args.learning_rate = 0.001
    args.bins = 5
    args.episode_length = 300
    args.replay_episodes = 10

    seed = 42
    utils.set_seed(seed)

    # Initialize the environment and agent
    env_name = "Pendulum-v1"
    agent = Agent(env_name, args)

    agent.train()

    print("Training completed.")


if __name__ == "__main__":
    train_agent()

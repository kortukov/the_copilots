import gym
from agent import Agent
from utils import Args, DiscreteActionWrapper


def evaluate_agent(agent, env, episodes=5):
    """Evaluate the agent performance."""
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return sum(total_rewards) / len(total_rewards)


def train_agent():
    # Define hyperparameters
    args = Args()  # Replace Args() with your own hyperparameters setting method
    args.num_episodes = 1000
    args.batch_size = 64
    args.gamma = 0.99
    args.initial_epsilon = 1.0
    args.final_epsilon = 0.05
    args.decay_rate = 0.0001
    args.target_update = 10
    args.replay_memory_size = 10000
    args.clip = 1.0
    args.alpha = 0.95
    args.epsilon = 0.01
    args.learning_rate = 0.0005

    # Initialize the environment and agent
    env = gym.make('Pendulum-v1')
    env = DiscreteActionWrapper(env, bins=5)
    agent = Agent(env, args)

    # Training loop
    for episode in range(args.num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize_model()
            if episode % args.target_update == 0:
                agent.update_target_network()
            agent.update_epsilon()

        # Log rewards
        print(f"Episode: {episode}, Total reward: {total_reward}")

        # Every 100 episodes, evaluate agent's performance
        if episode % 100 == 0:
            evaluation_score = evaluate_agent(agent, env)
            print(f"Episode: {episode}, Evaluation score: {evaluation_score}")

    print("Training completed.")


if __name__ == "__main__":
    train_agent()

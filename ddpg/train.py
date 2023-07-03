import json
import gymnasium as gym
import numpy as np 
import optparse
import pickle
import time
from tqdm import tqdm
import torch
import wandb

from DDPG import DDPGAgent
from TD3 import TD3Agent

with open('secrets.json', 'r') as f:
    SECRETS = json.load(f)

def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='int',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('--start-steps',action='store',  type='int',
                         dest='start_steps',default=10000,
                         help='Number of steps to sample random actions before training (default %default)')
    optParser.add_option('--update-after',action='store',  type='int',
                         dest='update_after',default=1000,
                         help='After which timestep to start training (to ensure fullness of replay buffer)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('--polyak',action='store',  type='float',
                         dest='polyak',default=0.995,
                         help='The parameter for polyak averaging. None means do not do Polyak averaging')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=42,
                         help='random seed (default %default)')
    optParser.add_option('--model', action='store', type='string',
                        dest='model', default=None,
                        help='Path to load a model checkpoint from')
    optParser.add_option('--results-dir', action='store', type='string',
                        dest='results_dir', default='results',
                        help='Directory to store results in')
    optParser.add_option('--agent', action='store', type='string',
                          dest='agent', default='TD3',
                          help='Agent to use (DDPG or TD3)')  

    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    start_steps = opts.start_steps # Steps sampling random actions
    update_after = opts.update_after # After which timestep to start training (to ensure fullness of replay buffer)

    random_seed = opts.seed
    #############################################
    # Parameters for loading and saving models and results
    model_path = opts.model
    results_dir = opts.results_dir


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if opts.agent == "DDPG": 
        agent = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                        update_target_every = opts.update_every, polyak = opts.polyak)
    elif opts.agent == "TD3":
        agent = TD3Agent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                        update_target_every = opts.update_every, polyak = opts.polyak)
    else:
        raise ValueError(f"Unknown agent {opts.agent}")

    if model_path:
        print(f"Loading model from {model_path}")
        agent_state = torch.load(model_path)
        agent.restore_state(agent_state)

    # Initialize wandb logging
    wandb.login(key=SECRETS["wandb_key"])

    wandb_config = agent._config
    wandb_config.update(
        {"env_name": env_name, "algorithm": opts.agent, "episodes": max_episodes}
    )

    wandb.init(project="the_copilots", config=agent._config)
    wandb.watch((agent.Q1, agent.Q2, agent.policy))

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"{results_dir}/{opts.agent}_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, max_episodes+1):
        start_ts = time.time()
        times = {}
        ob, _info = env.reset()
        agent.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            if timestep > start_steps:
                a = agent.act(ob)
            else:
                a = env.action_space.sample()

            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            agent.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            if done or trunc: break

        if timestep > update_after:
            episode_losses = agent.train(train_iter, times)
        else:
            episode_losses = [(0, 0, 0)]

        losses.extend(episode_losses)

        episode_losses = np.array(episode_losses)
        mean_ep_loss = np.mean(episode_losses, axis=0)
        q1_loss_value = mean_ep_loss[0]
        q2_loss_value = mean_ep_loss[1]
        actor_loss = mean_ep_loss[2]


        rewards.append(total_reward)
        lengths.append(t)
        
        # Log one episode to wandb
        wandb.log(
            {
                "q1_loss": q1_loss_value, 
                "q2_loss": q2_loss_value, 
                "actor_loss": actor_loss,
                "reward": total_reward,
                "time": time.time() - start_ts,
            } 
        )

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'{results_dir}/{opts.agent}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
            print(f"Episode times: {times}")
    save_statistics()

if __name__ == '__main__':
    main()

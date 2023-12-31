import itertools
import json
import laserhockey.hockey_env as h_env
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import numpy as np 
import optparse
import pickle
import pathlib
import time
from tqdm import tqdm
import torch
import wandb

from DDPG import DDPGAgent
from TD3 import TD3Agent
import utils
import evaluate 

# Hack for importing from parent directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dueling_dqn.agent import Agent as DDQN_Agent
from dueling_dqn.utils import load_hockey_args, CUSTOM_HOCKEY_ACTIONS
import shared_constants
import shared_utils

cwd = os.getcwd()
print(cwd)
with open('secrets.json', 'r') as f:
    SECRETS = json.load(f)

HOCKEY_ENVS = {"HockeyNormal", "HockeyWeak", "HockeyTrainShooting", "HockeyTrainDefense", "SelfPlay", "SelfPlayVar"}

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
                         dest='max_episodes',default=300,
                         help='number of episodes (default %default)')
    optParser.add_option('--start-steps',action='store',  type='int',
                         dest='start_steps',default=10000,
                         help='Number of steps to sample random actions before training (default %default)')
    optParser.add_option('--update-after',action='store',  type='int',
                         dest='update_after',default=2049,
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
    optParser.add_option('--gif-interval', action='store',  type='int',
                         dest='gif_interval', default=1000,
                         help='Interval between gif generations (default %default)')
    optParser.add_option('--model', action='store', type='string',
                        dest='model', default=None,
                        help='Path to load a model checkpoint from')
    optParser.add_option('--results-dir', action='store', type='string',
                        dest='results_dir', default='results',
                        help='Directory to store results in')
    optParser.add_option('--agent', action='store', type='string',
                          dest='agent', default='TD3',
                          help='Agent to use (DDPG or TD3)')  
    optParser.add_option('--disable-wandb', action='store_true',
                          help='Whether to not run the wandb logging.')  
    optParser.add_option('--prioritize', action='store_true',
                        help='Whether to use prioritized replay buffer.')  
    optParser.add_option('--custom-reward', action='store_true',
                        help='Whether to use our custom reward function.')  
    optParser.add_option('--eval-every', action='store', type='int',
                        dest="eval_every", default=250,
                        help='How often to run evaluation during training.') 
    optParser.add_option('--big-model', action='store_true',
                        dest="big_model", help='Whether to use bigger model.')  

    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    render_mode = "rgb_array"
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True, render_mode=render_mode)
    elif env_name == "HockeyNormal":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    elif env_name == "HockeyWeak":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    elif env_name == "HockeyTrainShooting":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    elif env_name == "HockeyTrainDefense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif env_name == "SelfPlay" or env_name == "SelfPlayVar":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    else:
        env = gym.make(env_name, render_mode="rgb_array")
    render = False
    log_interval = 100           # print avg reward in the interval
    gif_interval = opts.gif_interval         # create gif sometimes
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
    disable_wandb = opts.disable_wandb

    # Create results directory
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Additional agent parameters
    agent_params = {
        "eps": eps,
        "learning_rate_actor": lr,
        "update_target_every": opts.update_every,
        "polyak": opts.polyak,
        "prioritize": opts.prioritize,
        "is_hockey_env": env_name in HOCKEY_ENVS, 
    }

    if opts.big_model:
        agent_params.update({
            "hidden_sizes_actor": [512,512, 512],
            "hidden_sizes_critic": [512,512,256,64],
            })

    if opts.agent == "DDPG": 
        agent = DDPGAgent(env.observation_space, env.action_space, **agent_params)
    elif opts.agent == "TD3":
        agent = TD3Agent(env.observation_space, env.action_space, **agent_params)
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

    wandb.init(project="the_copilots", config=agent._config, mode="disabled" if disable_wandb else "online")
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

    if env_name == "HockeyNormal":
        player2 = h_env.BasicOpponent(weak=False)   
    elif env_name == 'HockeyWeak':
        player2 = h_env.BasicOpponent(weak=True)
    elif env_name == "SelfPlayVar":
        strong = h_env.BasicOpponent(weak=False)   
        hockey_args = load_hockey_args()
        agent2 = DDQN_Agent("HockeyNormal", hockey_args)
        agent2.load_checkpoint(
            "/mnt/lustre/oh/owl288/the_copilots/dueling_dqn/resulting_models/checkpoint_29750_HockeyNormal.pth",
             only_network=True
        )
        agent2._act = agent2.act
        agent2.act = lambda obs: CUSTOM_HOCKEY_ACTIONS[agent2._act(obs, eps=0.0)]
        player2 = agent2
    else:
        player2 = None

    # How to generate action of opponent
    if env_name == "HockeyNormal" or env_name == "HockeyWeak":
        player_2_act_func = lambda obs: player2.act(obs)
    elif env_name == "SelfPlay":
        player_2_act_func = lambda obs: agent.act(obs, eps=0.0)
    elif env_name == "SelfPlayVar":
        # Cycle through agent 2 and self using itertools
        action_functions_cycle = itertools.cycle(
            [('Self (TD3)', lambda obs: agent.act(obs, eps=0.0)), 
            ('DDQN', lambda obs: agent2.act(obs)),
              ('Strong', lambda obs: strong.act(obs))]
            )
    else:
        player_2_act_func = lambda obs: [0, 0., 0, 0]

    # training loop
    total_wins = 0
    total_losses = 0
    total_ties = 0
    list_eval_returns = []
    list_train_rewards = []
    for i_episode in range(1, max_episodes+1):
        if env_name == "SelfPlayVar":
            # Each new episode play against a new opponent
            opponent_name, player_2_act_func = next(action_functions_cycle)
        frames = []
        start_ts = time.time()
        times = {}
        ob, _info = env.reset()
        agent.reset()
        if env_name in HOCKEY_ENVS:
            obs_agent2 = env.obs_agent_two()
        total_reward=0
        total_closeness_to_puck = 0
        total_touch_puck = 0
        total_puck_direction = 0
        step_starting_index = timestep
        for t in range(max_timesteps):
            timestep += 1
            done = False
            # Action of main agent
            if timestep > start_steps:
                a1 = agent.act(ob)
            else:
                a1 = agent.random_action()# env.action_space.sample()

            if env_name in HOCKEY_ENVS:
                a2 = player_2_act_func(obs_agent2)
                (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
                obs_agent2 = env.obs_agent_two()
                total_closeness_to_puck += _info["reward_closeness_to_puck"]
                total_touch_puck += _info["reward_touch_puck"]
                total_puck_direction += _info["reward_puck_direction"]
            else:
                (ob_new, reward, done, trunc, _info) = env.step(a1)

            if opts.custom_reward:
                winner = _info["winner"]
                if winner == 0:
                    if done:
                        winner = -0.3  # encourage winning
                    else:
                        winner = -0.001 * t  # pushing to play aggressively
                reward = winner * 100 + utils.calculate_rewards(ob_new)

            total_reward+= reward
            agent.store_transition((ob, a1, reward, ob_new, done))
            ob=ob_new

            if i_episode % gif_interval == 0:
                if env_name in HOCKEY_ENVS:
                    frame = env.render(mode=render_mode)
                else:
                    frame = env.render()
                frames.append(frame)

            if done or trunc: 
                if env_name in HOCKEY_ENVS:
                    total_wins += int(_info["winner"] == 1)
                    total_losses += int(_info["winner"] == -1)
                    total_ties += int(_info["winner"] == 0)

                break

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
        wandb_log_dict = {
            "q1_loss": q1_loss_value, 
            "q2_loss": q2_loss_value, 
            "actor_loss": actor_loss,
            "reward": total_reward,
            "time": time.time() - start_ts,
        }
        if env_name in HOCKEY_ENVS:
            wandb_log_dict.update(
                {
                    "wins_avg": total_wins / i_episode,
                    "losses_avg": total_losses / i_episode,
                    "ties_avg": total_ties / i_episode,
                    "closeness_to_puck": total_closeness_to_puck,
                    "touch_puck": total_touch_puck,
                    "puck_direction": total_puck_direction,
                }
            )
        list_train_rewards.append(total_reward)

        if i_episode % opts.eval_every == 0:
            # Run evaluation
            if env_name == "SelfPlayVar":
                # Only evaluate against fixed agent
                player_2_act_func = lambda obs: agent2.act(obs)
            eval_results = evaluate.run_evaluation(
                agent, env, is_hockey_env=env_name in HOCKEY_ENVS, player_2_act_func=player_2_act_func,
            )

            list_eval_returns.append(eval_results.returns)
            wandb_log_dict.update(
                {
                    'eval_mean_return': eval_results.mean_return,
                    'eval_std_return': eval_results.std_return,
                }
            )
            if env_name in HOCKEY_ENVS:
                wandb_log_dict.update(
                    {
                        "eval_wins": eval_results.wins,
                        "eval_losses": eval_results.losses,
                        "eval_ties": eval_results.ties,
                    }
                )
        
        wandb.log(wandb_log_dict)

        # save every 1000 episodes
        if i_episode % 1000 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'{results_dir}/{opts.agent}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            if env_name == "SelfPlayVar":
                print(f"Playing against {opponent_name}")

            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))


            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
            print(f"Episode times: {times}")
        
        # Save episode as a gif
        if i_episode % gif_interval == 0:
            gif_filename = f"{results_dir}/last_episode.gif"
            utils.save_frames_as_gif(frames, path="", filename=gif_filename)
            wandb.log(
                {"video": wandb.Video(gif_filename, fps=25, format="gif")}
                )

    np_eval_returns = np.array(list_eval_returns)
    np_train_rewards = np.array(list_train_rewards)
    # Save eval results to numpy file
    eval_returns_filename = f"{results_dir}/eval_returns.npy"
    train_rewards_filename = f"{results_dir}/train_rewards.npy"
    np.save(eval_returns_filename, np_eval_returns)
    np.save(train_rewards_filename, np_train_rewards)
    
    wandb.save(eval_returns_filename)
    wandb.save(train_rewards_filename)
    save_statistics()

if __name__ == '__main__':
    main()

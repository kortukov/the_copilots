import json
import numpy as np
import pathlib
import torch
import wandb
import random

from laserhockey.hockey_env import BasicOpponent, HockeyEnv
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], 'ddpg'))
from ddpg.TD3 import TD3Agent
import ddpg.evaluate as evaluate
import shared_constants

GAMES_DIR = 'games/games/2023/8/9'

with open('ddpg/secrets.json', 'r') as f:
    SECRETS = json.load(f)

def get_game_paths():
    return sorted(pathlib.Path(GAMES_DIR).glob('*.npz'))

def load_td3_agent(path):
    env = HockeyEnv(mode=HockeyEnv.NORMAL)
    agent = TD3Agent(env.observation_space, env.action_space, **shared_constants.DEFAULT_TD3_PARAMS)
    agent_state = torch.load(path)
    agent.restore_state(agent_state)
    return agent

def player_in_good_teams(player):
    good_teams = ["AlphaPuck", "Arizona", "Psychedelic", "RLcochet", "HC_Slavoj", "ReinforcedLaziness", "HotChocolate", "Eigentor",
                  "The Q-Learners", "BetaBiscuit", "LaserLearningLunatics", "What the Puck?","Galatasaray",
                  ]
    for team in good_teams:
        if team in player:
            return True
    return False
def load_observations_into_memory(game_arrays, td3_agent):
    observations = []
    for game in game_arrays:
        try:
            game_arr = game['arr_0'].item()
            p1 = game_arr['player_one']
            p2 = game_arr['player_two']
            if not player_in_good_teams(p2):
                continue 
            transitions = game_arr['transitions']
        except:
            continue
        observations.append(transitions)
        # for transition in transitions:
        #     obs, a, next_obs, r, done, trunc, info = transition
        #     td3_agent.store_transition((obs, a, r, next_obs, done))
    for transitions in observations:
        for transition in transitions:
            obs, a, next_obs, r, done, trunc, info = transition
            td3_agent.store_transition((obs, a, r, next_obs, done))




game_arrays = [np.load(path, allow_pickle=True) for path in get_game_paths()]
print(f"Loaded {len(game_arrays)} games")
TD3_PATH = "agents/td3_updated.pth"
FIXED_TD3_PATH = "agents/td3.pth"
td3_agent = load_td3_agent(TD3_PATH)
fixed_agent = load_td3_agent(FIXED_TD3_PATH)
load_observations_into_memory(game_arrays, td3_agent)
print(f"Loaded {len(td3_agent.replay_buffer)} observations into memory")

wandb.login(key=SECRETS["wandb_key"])
wandb.init()
for epoch in range(30000):
    print(f"Epoch {epoch}")
    game_paths = get_game_paths()
    random.shuffle(game_paths)
    game_arrays = [np.load(path, allow_pickle=True) for path in game_paths[:100]]
    load_observations_into_memory(game_arrays, td3_agent)

    episode_losses = td3_agent.train(512, {})

    episode_losses = np.array(episode_losses)
    mean_ep_loss = np.mean(episode_losses, axis=0)
    q1_loss_value = mean_ep_loss[0]
    q2_loss_value = mean_ep_loss[1]
    actor_loss = mean_ep_loss[2]
    env = HockeyEnv(mode=HockeyEnv.NORMAL)
    
    player_2_act_func = lambda obs: fixed_agent.act(obs, eps=0.0)
    eval_results = evaluate.run_evaluation(
        td3_agent, env, is_hockey_env=True, player_2_act_func=player_2_act_func,
    )

    # Log one epoch to wandb
    wandb_log_dict = {
        "q1_loss": q1_loss_value, 
        "q2_loss": q2_loss_value, 
        "actor_loss": actor_loss,
        "eval_wins": eval_results.wins,
        "eval_losses": eval_results.losses,
        "eval_ties": eval_results.ties,
        'eval_mean_return': eval_results.mean_return,
        'eval_std_return': eval_results.std_return,
    }
    wandb.log(wandb_log_dict)
    # Saving the agent
    torch.save(td3_agent.state(), f"agents/td3_updated.pth")



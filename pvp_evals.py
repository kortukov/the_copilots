import dataclasses
import laserhockey.hockey_env as h_env
import optparse
import numpy as np
import torch

# Hack for importing from lower directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], 'ddpg'))
from TD3 import TD3Agent
import shared_constants
import shared_utils

@dataclasses.dataclass
class PvpResults:
    """Results of of two agents against each other."""
    p1_wins: int = 0
    p2_wins: int = 0
    ties: int = 0

def pvp_evaluation(agent1, agent2, env):
    # Hack to support both our agents and default agents
    if type(agent1) == h_env.BasicOpponent:
        agent1_act = lambda obs: agent1.act(obs)
    else:
        agent1_act = lambda obs: agent1.act(obs, eps=0.0)

    if type(agent2) == h_env.BasicOpponent:
        agent2_act = lambda obs: agent2.act(obs)
    else:
        agent2_act = lambda obs: agent2.act(obs, eps=0.0)

    p1_wins = 0
    p2_wins = 0
    ties = 0
    returns = []

    for i, seed in enumerate(shared_constants.PVP_EVALUATION_SEEDS):
        if i % 10 == 0:
            print(f"Evaluating on seed number {i}")
        shared_utils.set_seed(seed)
        ob, _info = env.reset()
        # agent1.reset()
        obs_agent2 = env.obs_agent_two()

        episode_return = 0
        while True: 
            a1 = agent1_act(ob)
            a2 = agent2_act(obs_agent2)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()

            episode_return += reward
            ob = ob_new
            if done or trunc: 
                p1_wins += int(_info["winner"] == 1)
                p2_wins += int(_info["winner"] == -1)
                ties += int(_info["winner"] == 0)
                break

        returns.append(episode_return)

    pvp_results = PvpResults(
        p1_wins=p1_wins / len(shared_constants.PVP_EVALUATION_SEEDS),
        p2_wins=p2_wins / len(shared_constants.PVP_EVALUATION_SEEDS),
        ties=ties / len(shared_constants.PVP_EVALUATION_SEEDS),
    )
    return pvp_results


if __name__ == "__main__":
    optParser = optparse.OptionParser()
    AGENT_TYPES = {'Weak', 'Strong', 'TD3', 'DDQN'}
    optParser.add_option('--agent1', action='store', type='string',
                         dest='agent1', default="Weak",
                         help=f'Type of agent1, one of {AGENT_TYPES}')

    optParser.add_option('--agent1-path', action='store', type='string',
                         dest='agent1_path', default=None,
                         help=f'Path to model for agent1')

    optParser.add_option('--agent2', action='store', type='string',
                         dest='agent2', default="Weak",
                         help=f'Type of agent 2, one of {AGENT_TYPES}')

    optParser.add_option('--agent2-path', action='store', type='string',
                         dest='agent2_path', default=None,
                         help=f'Path to model for agent1')

    opts, args = optParser.parse_args()


    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    
    if opts.agent1 == "Weak":
        agent1 = h_env.BasicOpponent(weak=True) 
    elif opts.agent1 == "Strong":
        agent1 = h_env.BasicOpponent(weak=False) 
    elif opts.agent1 == "TD3":
        if opts.agent1_path is None:
            raise ValueError("Provide --agent1-path")
        agent1 = TD3Agent(env.observation_space, env.action_space, **shared_constants.DEFAULT_TD3_PARAMS)
        print(f"Loading agent1 from {opts.agent1_path}")
        agent1_state = torch.load(opts.agent1_path)
        agent1.restore_state(agent1_state)
    elif opts.agent1 == "DDQN":
        if opts.agent1_path is None:
            raise ValueError("Provide --agent1-path")
        raise ValueError("For Sasha to implement")
    else:
        raise ValueError(f"Incorrect --agent1, should be one of {AGENT_TYPES}")
        

    if opts.agent2 == "Weak":
        agent2 = h_env.BasicOpponent(weak=True) 
    elif opts.agent2 == "Strong":
        agent2 = h_env.BasicOpponent(weak=False) 
    elif opts.agent2 == "TD3":
        if opts.agent2_path is None:
            raise ValueError("Provide --agent2-path")
        agent2 = TD3Agent(env.observation_space, env.action_space, **shared_constants.DEFAULT_TD3_PARAMS)
        print(f"Loading agent2 from {opts.agent2_path}")
        agent2_state = torch.load(opts.agent2_path)
        agent2.restore_state(agent2_state)
    elif opts.agent2 == "DDQN":
        if opts.agent2_path is None:
            raise ValueError("Provide --agent2-path")
        raise ValueError("For Sasha to implement")
    else:
        raise ValueError(f"Incorrect --agent1, should be one of {AGENT_TYPES}")

    pvp_results = pvp_evaluation(agent1, agent2, env) 
    print(pvp_results)




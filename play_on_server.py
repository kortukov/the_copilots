import json
import optparse
import numpy as np
import torch

from laserhockey.hockey_env import BasicOpponent, HockeyEnv
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

# Hack for importing from lower directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], 'ddpg'))
from ddpg.TD3 import TD3Agent
from dueling_dqn.agent import Agent as DDQN_Agent
from dueling_dqn.utils import load_hockey_args, CUSTOM_HOCKEY_ACTIONS

import shared_constants


with open('ddpg/secrets.json', 'r') as f:
    SECRETS = json.load(f)

class RemoteBasicOpponent(BasicOpponent, RemoteControllerInterface):

    def __init__(self, weak, keep_mode=True):
        BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier='StrongBasicOpponent')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)

class TD3Controller(RemoteControllerInterface):
    def __init__(self, path):
        RemoteControllerInterface.__init__(self, identifier='TD3')
        env = HockeyEnv(mode=HockeyEnv.NORMAL)
        self.agent = TD3Agent(env.observation_space, env.action_space, **shared_constants.DEFAULT_TD3_PARAMS)
        agent_state = torch.load(path)
        self.agent.restore_state(agent_state)
        
    def remote_act(self, obs):
        return self.agent.act(obs, eps=0.0)

class DDQNController(RemoteControllerInterface):
    def __init__(self, path):
        RemoteControllerInterface.__init__(self, identifier='DDQN')
        env_name = "HockeyNormal"
        hockey_args = load_hockey_args()
        self.agent = DDQN_Agent(env_name, hockey_args)
        self.agent.load_checkpoint(
            path,
            only_network=True,
        )
    def remote_act(self, obs):
        return np.array(CUSTOM_HOCKEY_ACTIONS[self.agent.act(obs, eps=0.0)])

if __name__ == '__main__':
    AGENT_TYPES = {'TD3', 'DDQN'}
    optParser = optparse.OptionParser()
    optParser.add_option('--agent', action='store', type='string',
                         dest='agent', default="Weak",
                         help=f'Type of agent1, one of {AGENT_TYPES}')

    optParser.add_option('--agent-path', action='store', type='string',
                         dest='agent_path', default=None,
                         help=f'Path to model for agent')

    optParser.add_option('--interactive', action='store_true', dest='interactive', default=False)
    optParser.add_option('--num-games', action='store', type='int', dest='num_games', default=None)
    optParser.add_option('--output-path', action='store', type='string', dest='output_path', default="")
    opts, args = optParser.parse_args()


    if opts.agent == "TD3":
        if opts.agent_path is None:
            raise ValueError("Provide --agent-path")
        controller = TD3Controller(path=opts.agent_path)
    elif opts.agent == "DDQN":
        if opts.agent_path is None:
            raise ValueError("Provide --agent-path")
        controller = DDQNController(path=opts.agent_path)
    else:
        print("Default is Weak player")
        controller = RemoteBasicOpponent(weak=False)

    output_path = f'./games/{opts.output_path}' 

    if opts.interactive:
        # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
        client = Client(
            username='The Copilots',
            password=SECRETS["password"],
            controller=controller, 
            output_path=output_path,
        )
    else:
        # Play n (None for an infinite amount) games and quit
        client = Client(
            username='The Copilots',
            password=SECRETS["password"],
            controller=controller, 
            output_path=output_path,
            interactive=False,
            op='start_queuing',
            num_games=opts.num_games
        )



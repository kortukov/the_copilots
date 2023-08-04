import dataclasses
import numpy as np
import random
import torch

# Hack for importing from parent directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import shared_constants
import shared_utils


@dataclasses.dataclass
class EvaluationResults:
    """Results of evaluation on EVALUATION SEEDS."""
    mean_return: float = 0.0
    std_return: float = 0.0
    returns: list = dataclasses.field(default_factory=list)
    # For hockey environments
    wins: int = 0
    losses: int = 0
    ties: int = 0


def run_evaluation(agent, env, is_hockey_env, player_2_act_func):
    returns = []
    total_wins = 0
    total_losses = 0
    total_ties = 0

    for seed in shared_constants.EVALUATION_SEEDS:
        shared_utils.set_seed(seed)
        ob, _info = env.reset()
        agent.reset()
        if is_hockey_env:
            obs_agent2 = env.obs_agent_two()

        episode_return = 0
        while True: 
            a1 = agent.act(ob, eps=0)
            if is_hockey_env:
                a2 = player_2_act_func(obs_agent2)
                (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
                obs_agent2 = env.obs_agent_two()
            else:
                (ob_new, reward, done, trunc, _info) = env.step(a1)

            episode_return += reward
            ob = ob_new
            if done or trunc: 
                if is_hockey_env:
                    total_wins += int(_info["winner"] == 1)
                    total_losses += int(_info["winner"] == -1)
                    total_ties += int(_info["winner"] == 0)
                break

        returns.append(episode_return)
    eval_results = EvaluationResults(
        mean_return=np.mean(returns),
        std_return=np.std(returns),
        returns=returns,
        wins=total_wins / len(shared_constants.EVALUATION_SEEDS),
        losses=total_losses / len(shared_constants.EVALUATION_SEEDS),
        ties=total_ties / len(shared_constants.EVALUATION_SEEDS),
    )
    return eval_results

from agent import Agent
import utils
from omegaconf import DictConfig, OmegaConf


def evaluate(env_name: str, cfg: DictConfig, path: str) -> None:
    args = utils.Args(**cfg.Args)
    seed = 42
    utils.set_seed(seed)

    # Initialize the environment and agent
    agent = Agent(env_name, args)
    agent.load_checkpoint(
        path,
        only_network=True,
    )
    agent.evaluate()

    print(f"Evaluation completed for {env_name}.")


def evaluate_agent(env_name: str, cfg: DictConfig, path: str) -> None:
    evaluate(env_name, cfg, path)


def main(env_name) -> None:
    envs_dict = {
        "HockeyWeak": ("HockeyWeak", "hockey_weak_config.yaml", "path"),
        "HockeyNormal": (
            "HockeyNormal",
            "hockey_normal_config.yaml",
            "D:/git_projects/ML_Masters/Reinforcement_Learning/project/the_copilots/dueling_dqn/checkpoints/HockeyNormal/checkpoint_25000_HockeyNormal.pth",
        ),
        "HockeyTrainShooting": (
            "HockeyTrainShooting",
            "hockey_shooting_config.yaml",
            "path",
        ),
        "HockeyTrainDefense": (
            "HockeyTrainDefense",
            "hockey_defense_config.yaml",
            "path",
        ),
    }
    env_name, cfg_name, path = envs_dict[env_name]
    cfg = OmegaConf.load(f"configs/{cfg_name}")
    evaluate_agent(env_name, cfg, path)


if __name__ == "__main__":
    for env_name in ["HockeyNormal"]:
        main(env_name)

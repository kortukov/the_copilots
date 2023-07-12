from agent import Agent
import utils
import hydra
from omegaconf import DictConfig, OmegaConf


def train(env_name: str, cfg: DictConfig) -> None:
    args = utils.Args(**cfg.Args)
    seed = 42
    utils.set_seed(seed)

    # Initialize the environment and agent
    agent = Agent(env_name, args)
    agent.load_checkpoint(
        "D:/git_projects/ML_Masters/Reinforcement_Learning/project/the_copilots/dueling_dqn/checkpoints/HockeyWeak/checkpoint_30000_HockeyWeak.pth",
        only_network=True,
    )
    agent.train()

    print(f"Training completed for {env_name}.")


@hydra.main(config_path="configs")
def train_agent(cfg: DictConfig) -> None:
    train(env_name, cfg)


if __name__ == "__main__":
    envs_and_configs = [
        # ("Pendulum-v1", "pendulum_config.yaml"),
        # ("HalfCheetah-v4", "cheetah_config.yaml"),
        # ("HockeyWeak", "hockey_weak_config.yaml"),
        ("HockeyNormal", "hockey_normal_config.yaml"),
        # ("HockeyTrainShooting", "hockey_shooting_config.yaml"),
        # ("HockeyTrainDefense", "hockey_defense_config.yaml"),
        # Add more environments and configs as needed
    ]

    for env_name, cfg_name in envs_and_configs:
        cfg = OmegaConf.load(f"configs/{cfg_name}")
        train_agent(cfg)

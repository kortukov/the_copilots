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
        "/resulting_models/checkpoint_29750_HockeyNormal.pth",
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
        # ("Pendulum-v1-noisy", "pendulum_config_noisy.yaml"),
        # ("Pendulum-v1-prioritize", "pendulum_config_prioritize.yaml"),
        # ("Pendulum-v1-noisy-prioritize", "pendulum_config_noisy_prioritize.yaml"),
        # ("HalfCheetah-v4", "cheetah_config.yaml"),
        # ("HalfCheetah-v4-noisy", "cheetah_config_noisy.yaml"),
        # ("HalfCheetah-v4-prioritize", "cheetah_config_prioritize.yaml"),
        # ("HalfCheetah-v4-noisy-prioritize", "cheetah_config_noisy_prioritize.yaml"),
        # ("HockeyWeak", "hockey_weak_config.yaml"),
        ("HockeyNormal", "hockey_normal_config.yaml"),
        # ("HockeyTrainShooting", "hockey_shooting_config.yaml"),
        # ("HockeyTrainDefense", "hockey_defense_config.yaml"),
        # Add more environments and configs as needed
    ]

    for env_name, cfg_name in envs_and_configs:
        cfg = OmegaConf.load(f"configs/{cfg_name}")
        train_agent(cfg)

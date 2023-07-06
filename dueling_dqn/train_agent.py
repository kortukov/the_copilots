from agent import Agent
import utils
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="cheetah_config.yaml", version_base="1.1")
def train_agent(cfg: DictConfig) -> None:
    args = utils.Args(**cfg.Args)
    seed = 42
    utils.set_seed(seed)

    # Initialize the environment and agent
    env_name = "HalfCheetah-v4"
    agent = Agent(env_name, args)
    # agent.load_checkpoint("checkpoints/checkpoint_1300_HalfCheetah-v4.pth")
    agent.train()

    print("Training completed.")


if __name__ == "__main__":
    train_agent()

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    
    from env.env import Env_generator
    Env_generator = Env_generator(cfg)
    envs = Env_generator.sample_n_envs()

    print("Got the envs")
    print(envs.shape)


if __name__ == "__main__":
    main()
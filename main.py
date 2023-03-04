import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    
    from env.env import Env_generator
    Env_generator = Env_generator(cfg)
    
    from mae.mae import MAE
    mae = MAE(Env_generator.sample_n_envs, cfg)

if __name__ == "__main__":
    main()
import time
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    from env.env import Env_generator
    Env_generator = Env_generator(cfg)
    
    if cfg.task.lower() == "pretrain_encoder":
        from encoders.encoder import Encoder
        encoder = Encoder(cfg, Env_generator.sample_n_envs)
        encoder.start_training()

    elif cfg.task.lower() == "train":
        from encoders.encoder import Encoder
        from decoders.decoder import Decoder

        encoder = Encoder(cfg).load()
        decoder = Decoder(cfg, encoder, Env_generator)

        decoder.start_training()

    else:
        print("Task ill-defined")

if __name__ == "__main__":
    main()
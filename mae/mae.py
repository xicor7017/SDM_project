import time
import torch
from mae.model import MaskedAutoencoderViT

class MAE:
    def __init__(self, envs_generator, config):
        self.config = config
        self.envs_generator = envs_generator

        self.device = torch.device(self.config.common.device)
        self.model = MaskedAutoencoderViT().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.encoder.lr, betas=(0.9, 0.95), weight_decay=0.05)

        x = self.train_iteration()

    def train_iteration(self):
        sample_envs = self.envs_generator(batch_size=self.config.encoder.batch_size)


        print(sample_envs.shape, sample_envs.device)

        time.sleep(1000)
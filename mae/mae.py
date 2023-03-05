import time
import torch

import numpy as np
from tqdm import tqdm

from mae.encoder import get_encoder_n_optimizer
from torch.utils.tensorboard import SummaryWriter

class MAE:
    def __init__(self, envs_generator, config):
        self.config = config
        self.envs_generator = envs_generator
        self.log_location = self.config.encoder.logdir + self.config.encoder.run_name
        
        self.model, self.optimizer = get_encoder_n_optimizer(config)

    def perform_iteration(self, batch_size, eval=False):
        self.model.train()
        if eval: self.model.eval()
        sample_envs = self.envs_generator(batch_size)

        loss, pred, mask = self.model(sample_envs, mask_ratio=self.config.encoder.mask_ratio)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def init_training(self):
        self.logger = SummaryWriter(log_dir=self.log_location)

    def log(self, training_loss, testing_loss, itr):
        self.logger.add_scalar("Loss/training_loss", training_loss, itr)
        self.logger.add_scalar("Loss/testing_loss", testing_loss, itr)
        self.save()

    def save(self):
        torch.save(self.model.state_dict(), self.log_location + "model.state_dict")

    def load(self):
        pass

    def start_training(self):
        self.init_training()

        for itr in tqdm(range(1, self.config.encoder.training_iterations)):
            training_loss  = self.perform_iteration(self.config.encoder.batch_size)

            if itr%50 == 0:
                testing_loss = self.perform_iteration(self.config.encoder.batch_size, eval=True)
                self.log(training_loss.item(), testing_loss.item(), itr)
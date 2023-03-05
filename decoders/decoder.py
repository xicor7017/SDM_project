import time
import torch

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Decoder:
    def __init__(self, config, encoder, env_generator):
        self.config = config
        self.encoder = encoder
        self.env_generator = env_generator

        self.log_location = self.config.encoder.logdir + self.config.decoder.run_name

    def init_training(self):
        self.logger = SummaryWriter(log_dir=self.log_location)

    def perform_iteration(self, batch_size, eval=False):

        print("Inside eval")
        time.sleep(1000)

        self.model.train()
        if eval: self.model.eval()

        sample_envs = self.envs_generator(batch_size)

        loss, pred, mask = self.model(sample_envs, mask_ratio=self.config.encoder.mask_ratio)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def start_training(self):
        self.init_training()

        for itr in tqdm(range(1, self.config.decoder.training_iterations)):
            training_loss  = self.perform_iteration(self.config.decoder.batch_size)

            if itr%50 == 0:
                testing_loss = self.perform_iteration(self.config.decoder.batch_size, eval=True)
                self.log(training_loss.item(), testing_loss.item(), itr)
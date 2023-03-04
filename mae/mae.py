import time
import torch

import numpy as np
from tqdm import tqdm

from mae.model import MaskedAutoencoderViT
from torch.utils.tensorboard import SummaryWriter

class MAE:
    def __init__(self, envs_generator, config):
        self.config = config
        self.envs_generator = envs_generator

        #Setting encoding parameters to adapt the model to n-dof envs
        if self.config.env.dof % 2: self.in_chans = 2
        else: self.in_chans = 1

        self.img_size = int(np.sqrt((self.config.env.dof_size ** self.config.env.dof) / self.in_chans))
        self.patch_size = self.img_size // 8

        #initializing model and parameters
        self.device = torch.device(self.config.common.device)
        self.model = MaskedAutoencoderViT(img_size=self.img_size, in_chans=self.in_chans, patch_size=self.patch_size).to(self.device)
        
    def perform_iteration(self, batch_size, eval=False):
        self.model.train()
        if eval: self.model.eval()
        sample_envs = self.envs_generator(batch_size)
        loss, pred, mask = self.model(sample_envs, mask_ratio=self.config.encoder.mask_ratio)

        return loss

    def init_training(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.encoder.lr, betas=(0.9, 0.95), weight_decay=0.05)
        self.logger = SummaryWriter(log_dir=self.config.encoder.logdir)


    def log(self, loss, itr):
        self.logger.add_scalar("Loss/loss", loss, itr)
        self.save()

    def save(self):
        torch.save(self.model.state_dict(), self.config.encoder.logdir + "model.state_dict")

    def load(self):
        pass

    def start_training(self):
        self.init_training()

        for itr in tqdm(range(self.config.encoder.training_iterations)):
            loss = self.perform_iteration(self.config.encoder.batch_size)

            if itr % 40 == 0:
                self.log(loss.item(), itr)
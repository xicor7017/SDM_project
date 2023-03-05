import time
import torch

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Encoder:
    def __init__(self, config, envs_generator):
        self.config = config
        self.envs_generator = envs_generator
        self.log_location = self.config.encoder.logdir + self.config.encoder.run_name
        
        self.model, self.optimizer = self.get_encoder_n_optimizer()

    def get_encoder_n_optimizer(self):
        self.device = torch.device(self.config.common.device)

        if self.config.encoder.arch == "mae":
            #Setting encoding parameters to adapt the model to n-dof envs
            if self.config.env.dof % 2: in_chans = 2
            else: in_chans = 1

            img_size = int(np.sqrt((self.config.env.dof_size ** self.config.env.dof) / in_chans))
            patch_size = 4

            from encoders.mae import MaskedAutoencoderViT
            model = MaskedAutoencoderViT(img_size=img_size, in_chans=in_chans, patch_size=patch_size, embed_dim=64, decoder_embed_dim=64).to(self.device)
        
        elif self.config.encoder.arch == "FF":    
            from encoders.FF import FF
            model = FF(self.config.env.dof_size ** self.config.env.dof).to(self.device)
        
        elif self.config.encoder.arch == "cnn":
            from encoders.CNN import CNN
            model = CNN(self.config.env.dof, self.config.env.dof_size).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.encoder.lr, betas=(0.9, 0.95))
        return model, optimizer

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
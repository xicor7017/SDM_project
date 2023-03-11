import time
import torch

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Encoder:
    def __init__(self, config, envs_generator=None):
        self.config         = config
        self.device         = torch.device(self.config.common.device)
        self.log_location   = self.config.encoder.logdir + self.config.encoder.run_name
        self.envs_generator = envs_generator
        
    def get_encoder(self):
        if self.config.encoder.arch.lower() == "mae":
            #Setting encoding parameters to adapt the model to n-dof envs
            if self.config.env.dof % 2: in_chans = 2
            else: in_chans = 1

            img_size = int(np.sqrt((self.config.env.dof_size ** self.config.env.dof) / in_chans))
            patch_size = 4

            from encoders.mae import MaskedAutoencoderViT
            model = MaskedAutoencoderViT(img_size=img_size, in_chans=in_chans, patch_size=patch_size, embed_dim=64, decoder_embed_dim=64).to(self.device)
        
        elif self.config.encoder.arch.lower() == "ff":    
            from encoders.FF import FF
            model = FF(self.config.env.dof_size ** self.config.env.dof).to(self.device)
        
        elif self.config.encoder.arch.lower() == "cnn":
            from encoders.CNN import CNN
            model = CNN(self.config.env.dof, self.config.env.dof_size).to(self.device)

        return model

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
        self.best      = float("inf")
        self.model     = self.get_encoder()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.encoder.lr, betas=(0.9, 0.95))
        self.logger = SummaryWriter(log_dir=self.log_location)

    def log(self, training_loss, testing_loss, itr):
        current = training_loss + testing_loss
        self.logger.add_scalar("Loss/training_loss", training_loss, itr)
        self.logger.add_scalar("Loss/testing_loss", testing_loss, itr)
        best_model_yet = current < self.best
        if best_model_yet:
            self.best = current
        self.save(best_model_yet)

    def save(self, best):
        if best:
            torch.save(self.model.state_dict(), self.log_location + "/best_model.state_dict")
        else:
            torch.save(self.model.state_dict(), self.log_location + "/model.state_dict")

    def load(self):
        self.model = self.get_encoder()
        self.model.load_state_dict(torch.load(self.log_location + "/model.state_dict"))
        return self.model

    def start_training(self):
        self.init_training()

        for itr in tqdm(range(1, self.config.encoder.training_iterations)):
            training_loss  = self.perform_iteration(self.config.encoder.batch_size)

            if itr%50 == 0:
                testing_loss = self.perform_iteration(self.config.encoder.batch_size, eval=True)
                self.log(training_loss.item(), testing_loss.item(), itr)
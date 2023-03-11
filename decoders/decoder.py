import time
import torch
import random

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Decoder:
    def __init__(self, config, encoder, env_generator):
        self.config = config
        self.encoder = encoder
        self.env_generator = env_generator
        self.device = torch.device(self.config.common.device)
        self.log_location = self.config.decoder.logdir + self.config.decoder.run_name

    def init_training(self):
        self.best      = float("inf")
        self.model = self.get_decoder()
        self.set_seeds(self.config.common.seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.decoder.lr)
        self.logger = SummaryWriter(log_dir=self.log_location)

    def set_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def get_decoder(self):
        if self.config.decoder.arch == "ff":
            from decoders.ff import FF
            state_size = self.config.env.dof_size
            env_size   = self.config.env.dof_size ** self.config.env.dof
            model      = FF(input_dims=env_size + state_size + state_size, output_dim=self.config.env.action_size, config=self.config).to(self.device)
        elif self.config.decoder.arch == "rnn":
            from decoders.rnn import RNN
            state_size = 2
            env_size   = self.config.env.dof_size ** self.config.env.dof
            model      = RNN(input_dims=env_size + state_size + state_size, output_dim=self.config.env.action_size, device=self.device, config=self.config).to(self.device)
        
        else:
            print("Decoder arch not defined")
            exit()

        return model

    def perform_iteration(self, batch_size, eval=False):
        self.model.train()
        if self.config.decoder.arch != "rnn":
            if eval: self.model.eval()
            else: self.model.train()

        envs, start_position, actions, traj = self.env_generator.sample_env_n_trajs(batch_size, self.config.decoder.max_traj_len)
        
        _, encoded_envs, _ = self.encoder(envs, mask_ratio=0)

        _, loss, log_loss = self.model(encoded_envs, start_position, actions, traj)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return log_loss

    def start_training(self):
        self.init_training()

        for itr in tqdm(range(1, self.config.decoder.training_iterations)):
            training_loss  = self.perform_iteration(self.config.decoder.batch_size)


            if itr%50 == 0:
                testing_loss = self.perform_iteration(self.config.decoder.batch_size*10, eval=True)
                self.log(training_loss.item(), testing_loss.item(), itr)

    
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
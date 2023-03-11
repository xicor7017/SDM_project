import time
import torch

class Env_generator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.common.device)
        self.env_shape = [self.config.env.dof_size for _ in range(self.config.env.dof)]

        self.window_size = self.config.env.dof_size//2

    def sample_n_envs(self, batch_size=32):
        envs = torch.zeros((batch_size, *self.env_shape), device=self.device, requires_grad=False)

        #Adding walls to each dim
        for dof in range(1, self.config.env.dof+1):
            if dof != 1:
                envs = envs.transpose(1, dof)

            if self.config.env.walled:
                wall = torch.ones((batch_size, *self.env_shape[1:]), device=self.device, requires_grad=False)
                random_wall_index = torch.randint(low=0, high=self.config.env.dof_size, size=(batch_size,), device=self.device, requires_grad=False)

                random_window_index = torch.randint(low=0, high=self.config.env.dof_size//2+1, size=(batch_size,), device=self.device, requires_grad=False)
                
                for window_index in range(self.window_size):
                    wall[torch.arange(batch_size), random_window_index+window_index] = 0.0

                envs[torch.arange(batch_size), random_wall_index] = wall

            if dof != 1:
                envs = envs.transpose(dof, 1)

        return envs

    def sample_action(self, env, actions, position):
        pass

    def sampled_trajectory(envs, start_pos):
        pass

    def sample_positions(self, envs):
        positions_found = False
        while not positions_found:
            idx = torch.tensor([range(envs.shape[0])]).int()
            positions = torch.randint(low=0, high=self.config.env.dof_size, size=(envs.shape[0], 2))
            x = torch.cat((idx.T,positions), 1)
            env_state = torch.stack([envs[i,j,k] for i,j,k in x])

            if (env_state == 0.0).all():
                positions_found=True

        return positions


    def sample_env_n_trajs(self, batch_size, traj_len):
        envs = self.sample_n_envs(batch_size)

        start_pos = self.sample_positions(envs)
        sampled_trajectory = self.sampled_trajectory(envs, start_pos)


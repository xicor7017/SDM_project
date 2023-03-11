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

    def sample_action(self, envs, position, last_action=None):
        feasible_actions = False

        if last_action is not None:
            inverse_action = -1 * last_action   #To make sure we dont backtraj trajectory through random actions

        while not feasible_actions:
            
            if last_action is None:
                random_actions = torch.randint(low=-1, high=2, size=(envs.shape[0], self.config.env.dof))
            else:
                random_actions = torch.randint(low=-1, high=2, size=(envs.shape[0], self.config.env.dof))
                while (random_actions == inverse_action).all(1).any():
                    random_actions = torch.randint(low=-1, high=2, size=(envs.shape[0], self.config.env.dof))
                
            expected_positions = position + random_actions
            feasible_actions = self.check_position_feasibility(envs, expected_positions)

        return random_actions, expected_positions


    def sample_trajectory(self, envs, start_pos, traj_len):
        positions = []
        actions = []

        position = start_pos
        action = None
        for _ in range(traj_len):
            action, position = self.sample_action(envs, position, last_action=action)
            actions.append(action)
            positions.append(position)

        actions = torch.stack(actions)
        positions = torch.stack(positions)

        return actions, positions

    def check_position_feasibility(self, envs, positions):
        # Check for boundary violations
        clipped_positions = torch.clip(positions, min=0, max=self.config.env.dof_size-1)

        if not (clipped_positions==positions).all():
            return False
        else:
            # Check for wall collisions
            env_ids = torch.tensor([range(envs.shape[0])]).int()
            idx = torch.cat((env_ids.T,positions), 1)
            env_state = torch.stack([envs[i,j,k] for i,j,k in idx])
            if not (env_state == 0.0).all():    # The positions collide with walls
                return False
    
        return True

    def sample_positions(self, envs):
        positions_found = False
        while not positions_found:
            random_positions = torch.randint(low=0, high=self.config.env.dof_size, size=(envs.shape[0], 2))
            positions_found = self.check_position_feasibility(envs, random_positions)
        return random_positions

    def sample_env_n_trajs(self, batch_size, traj_len):
        envs = self.sample_n_envs(batch_size)
        start_pos = self.sample_positions(envs)
        sampled_actions, sampled_trajectory = self.sample_trajectory(envs, start_pos, traj_len)

        return sampled_actions, sampled_trajectory
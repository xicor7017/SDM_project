import time
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dims, output_dim, device, config):
        super(RNN, self).__init__()
        self.config = config
        self.device = device
        self.gru    = nn.GRU(input_size=input_dims, hidden_size=config.decoder.hidden_dims, num_layers=1, batch_first=True)

        self.layers = nn.Sequential(
                                nn.Linear(self.config.decoder.hidden_dims, self.config.decoder.hidden_dims),
                                nn.Tanh(),
                                nn.Linear(self.config.decoder.hidden_dims, output_dim)
                                    )
        
        self.cross_entr_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.functional.mse_loss

    def init_hidden_state(self, batch_size=1):
        self.hidden_states = torch.zeros((1, batch_size, self.config.decoder.hidden_dims)).to(self.device)

    def forward(self, encoded_envs, start_position, actions, traj, single_step=False):
        if not single_step:
            self.init_hidden_state(batch_size=encoded_envs.shape[0])

        encoded_envs_expanded = encoded_envs.unsqueeze(0).repeat(actions.shape[0],1,1)
        current_pos = torch.cat((start_position.unsqueeze(0), traj[:-1]), 0)
        end_pos = traj[-1].unsqueeze(0).repeat(current_pos.shape[0], 1, 1)

        network_input = torch.cat((encoded_envs_expanded, current_pos, end_pos), -1).transpose(0,1)

        output, self.hidden_states = self.gru(network_input, self.hidden_states)

        pred_actions = self.layers(output).reshape(-1) + 1   #To make class labels start from 0
        expected_actions = actions.transpose(0,1).reshape(-1) + 1

        if self.config.decoder.entropy_loss:
            loss = self.cross_entr_loss(pred_actions.float(), expected_actions.float())
            log_loss = self.mse_loss(pred_actions.float(), expected_actions.float())
        else:
            loss = self.mse_loss(pred_actions.float(), expected_actions.float())
            log_loss = loss

        #################################
        # Outputs should be converted to int -1 0 1 to be used as actions
        #################################

        return output, loss, log_loss

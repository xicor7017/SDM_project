import time
import torch
import torch.nn as nn

class FF(nn.Module):
    def __init__(self, input_dims, output_dim, config):
        super(FF, self).__init__()

        self.ff = nn.Sequential(
                                        nn.Linear(input_dims, config.decoder.hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(config.decoder.hidden_dims, config.decoder.hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(config.decoder.hidden_dims, output_dim)
                                    )

    def forward_loss(self, in_data, pred_data):
        squared_error =  (in_data - pred_data)**2
        return squared_error.mean()

    def forward(self, in_data, mask_ratio):

        print("Implement forward and forward loss")
        time.sleep(1000)

        return loss, decoded, None
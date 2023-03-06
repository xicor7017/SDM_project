import time
import torch
import torch.nn as nn

class FF(nn.Module):
    def __init__(self, in_dim, output_dims):
        super(FF, self).__init__()

        self.ff = nn.Sequential(
                                        nn.Linear(in_dim, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, output_dims)
                                    )

    def forward_loss(self, in_data, pred_data):
        squared_error =  (in_data - pred_data)**2
        return squared_error.mean()

    def forward(self, in_data, mask_ratio):

        print("Implement forward and forward loss")
        time.sleep(1000)

        return loss, decoded, None
import torch
import torch.nn as nn

class FF(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=512):
        super(FF, self).__init__()

        self.encoder = nn.Sequential(
                                        nn.Linear(in_dim, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, bottleneck_dim)
                                    )

        self.decoder = nn.Sequential(
                                        nn.Linear(bottleneck_dim, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, in_dim)
                                    )


    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, x):
        return self.decoder(x)

    def forward_loss(self, in_data, pred_data):
        squared_error =  (in_data - pred_data)**2
        return squared_error.mean()

    def forward(self, in_data, mask_ratio):

        in_data = in_data.view((in_data.shape[0], -1))

        encoded = self.forward_encoder(in_data)
        decoded = self.forward_decoder(encoded)

        loss = self.forward_loss(in_data, decoded)

        return loss, decoded, None
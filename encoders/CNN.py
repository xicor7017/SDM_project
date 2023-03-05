import torch
import torch.nn as nn
from convNd import convNd

class CNN(nn.Module):
    def __init__(self, dof, dof_size, bottleneck_dim=512):
        super(CNN, self).__init__()

        weight = torch.rand(1)[0]
        bias = torch.rand(1)[0]

        self.cnn = convNd(
            in_channels=1, 
            out_channels=1,
            num_dims=dof, 
            kernel_size=2, 
            stride=tuple([1]*dof), 
            padding=0, 
            padding_mode='zeros',
            output_padding=0,
            is_transposed=False,
            use_bias=True, 
            groups=1,
            kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
            bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).cuda()

        self.encoder = nn.Sequential(
                                        nn.Linear((dof_size-1)**dof, 1024),
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
                                        nn.Linear(1024, dof_size ** dof)
                                    )

    def forward_encoder(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)

        x = x.view((x.shape[0], -1))
        return self.encoder(x)

    def forward_decoder(self, x, output_shape):
        x = self.decoder(x).view((*output_shape))
        return x

    def forward_loss(self, in_data, pred_data):
        squared_error =  (in_data - pred_data)**2
        return squared_error.mean()

    def forward(self, in_data, mask_ratio):
        encoded = self.forward_encoder(in_data)
        decoded = self.forward_decoder(encoded, in_data.shape)
        loss = self.forward_loss(in_data, decoded)
        return loss, decoded, None
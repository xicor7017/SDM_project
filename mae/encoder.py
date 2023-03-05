import torch

import numpy as np
from mae.model import MaskedAutoencoderViT
from mae.FF import FF, CNN

def get_encoder_n_optimizer(config):
    device = torch.device(config.common.device)

    #Setting encoding parameters to adapt the model to n-dof envs
    if config.env.dof % 2: in_chans = 2
    else: in_chans = 1

    img_size = int(np.sqrt((config.env.dof_size ** config.env.dof) / in_chans))
    patch_size = 4

    #initializing model and parameters
    device = torch.device(config.common.device)

    #model = MaskedAutoencoderViT(img_size=img_size, in_chans=in_chans, patch_size=patch_size, embed_dim=64, decoder_embed_dim=64).to(device)
    model = FF(config.env.dof_size ** config.env.dof).to(device)
    
    #model = CNN(config.env.dof, config.env.dof_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.encoder.lr, betas=(0.9, 0.95))
    return model, optimizer
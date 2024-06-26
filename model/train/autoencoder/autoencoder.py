from model.train.autoencoder.encoder import Encoder
from model.train.autoencoder.decoder import Decoder
from model.settings import *

import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_space_size: int = AE_LATENT_SPACE_SIZE) -> None:
        super().__init__()

        self.encoder = Encoder(latent_space_size=latent_space_size)
        self.decoder = Decoder(latent_space_size=latent_space_size)

    def forward(self, x):
        out1 = self.encoder(x)
        out2 = self.decoder(out1)
        return out1, out2
from model.train.variational_autoencoder.variational_encoder import VariationalEncoder
from model.train.variational_autoencoder.variational_decoder import VariationalDecoder

import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_space_size: int = 64) -> None:
        super().__init__()
        self.latent_space_size = latent_space_size
        self.encoder = VariationalEncoder(latent_space_size=latent_space_size)
        self.decoder = VariationalDecoder(latent_space_size=latent_space_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)

        z = self.reparametrize(mu, log_var)

        x_hat = self.decoder(z)

        return (mu, log_var), x_hat
    
    @staticmethod
    def reparametrize(mu, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z
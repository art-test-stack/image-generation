from model.settings import *
import torch
import torch.nn as nn


class VariationalEncoder(nn.Module):
    def __init__(self, latent_space_size: int = VAE_LATENT_SPACE_SIZE) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.mu = nn.Linear(256 * 2 * 2, latent_space_size)
        self.log_var = nn.Linear(256 * 2 * 2, latent_space_size)

    def forward(self, x):
        out = self.main(x)

        mu = self.mu(out)
        log_var = self.log_var(out)

        return mu, log_var
from model.settings import *
import torch
import torch.nn as nn


class VariationalDecoder(nn.Module):
    def __init__(self, latent_space_size: int = VAE_LATENT_SPACE_SIZE) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_space_size, 256 * 2 * 2),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out
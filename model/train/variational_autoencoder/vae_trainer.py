from model.train.trainer import Trainer
from model.train.variational_autoencoder.variational_autoencoder import VariationalAutoEncoder
from model.settings import *

import torch
from torch import nn, optim
import torch.nn.functional as F
from pathlib import Path

class KL_div(nn.Module):
    def __init__(self, beta: float = VAE_BETA_KL) -> None:
        super().__init__()
        self.beta = beta
        self.loss = F.cross_entropy # nn.CrossEntropyLoss()
    
    def forward(self, X, params):
        x_hat, x = X
        mu, log_var = params
        # bce = self.loss(x_hat, x)
        bce = F.cross_entropy(x_hat, x, reduction='mean')
        kld = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var))

        return bce + self.beta * kld
    
class VAETrainer(Trainer):
    def __init__(
            self,
            device = DEVICE,
            model_file: str | Path = VAE_MODEL, 
            trainer_file: str | Path = VAE_TRAINER,
            force_learn: bool = False
        ) -> None:

        model = VariationalAutoEncoder(latent_space_size=VAE_LATENT_SPACE_SIZE)
        loss = nn.KLDivLoss # KL_div()
        opt = optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE, betas=VAE_BETAS)

        super().__init__(model=model, loss=loss, optimizer=opt, device=device, model_file=model_file, trainer_file=trainer_file, force_learn=force_learn)

    def get_output_from_batch(self, batch):
        x, _, _ = batch
        x = x.to(self.device)
        (mu, log_var), x_hat = self.model(x[:,[0],:,:])
        return (x_hat, x), (mu, log_var)
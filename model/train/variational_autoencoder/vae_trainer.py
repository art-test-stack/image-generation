from model.train.trainer import Trainer
from model.train.variational_autoencoder.variational_autoencoder import VariationalAutoEncoder
from settings import *

import torch
from pathlib import Path

class VAETrainer(Trainer):
    def __init__(
            self, 
            model: VariationalAutoEncoder, 
            loss, 
            optimizer,
            device = DEVICE,
            file_name: str | Path = VAE_MODEL, 
            force_learn: bool = False
        ) -> None:
        super().__init__(model, loss, optimizer, device, file_name, force_learn)

    def get_output_from_batch(self, batch):
        x, _, _ = batch
        x = x.to(self.device)
        (mu, log_var), x_hat = self.model(x[:,[0],:,:])
        return (x_hat, x), (mu, log_var)
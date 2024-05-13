from model.train.trainer import Trainer
from model.train.autoencoder.autoencoder import AutoEncoder
from model.settings import *

from pathlib import Path
import torch.nn as nn
from torch import optim

class AutoEncoderTrainer(Trainer):
    def __init__(
            self,
            device = DEVICE,
            model_file: str | Path = AE_MODEL, 
            trainer_file: str | Path = AE_TRAINER, 
            force_learn: bool = False
        ) -> None:

        model = AutoEncoder(latent_space_size=AE_LATENT_SPACE_SIZE)
        loss = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=AE_LEARNING_RATE, betas=AE_BETAS)

        super().__init__(model=model, loss=loss, optimizer=opt, device=device, model_file=model_file, trainer_file=trainer_file, force_learn=force_learn)

    def get_output_from_batch(self, batch):
        x, _, _ = batch
        x = x.to(self.device)
        _, output = self.model(x[:,[0],:,:])
        return x, output
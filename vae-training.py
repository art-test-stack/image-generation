from model.data import StackedMNIST
from model.train.variational_autoencoder.variational_autoencoder import VariationalAutoEncoder
from model.train.variational_autoencoder.vae_trainer import VAETrainer

from model.settings import *
from model.test.verification_net import VerificationNet
from model.utils import get_gpu

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path

if __name__=="__main__":
    get_gpu()

    trainset = StackedMNIST(train=True, mode=AE_DATAMODE)
    testset = StackedMNIST(train=False, mode=AE_DATAMODE)

    trainer = VAETrainer()
    trainer.force_relearn = True

    # Trainer = Trainer.load_trainer(trainer_file)

    trainer.train(trainset=trainset,valset=testset)

    plt.plot(trainer.losses, label="losses")
    plt.plot(trainer.val_losses, label="val losses")
    plt.legend()

    trainer.print_reconstructed_img(testset)
    verifNet = VerificationNet()


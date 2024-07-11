from model.data import StackedMNIST
from model.train.autoencoder.autoencoder import AutoEncoder
from model.train.autoencoder.ae_trainer import AutoEncoderTrainer

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

    trainer = AutoEncoderTrainer()
    trainer.force_relearn = True
    trainer.train(trainset=trainset,valset=testset)

    plt.plot(trainer.losses, label="losses")
    plt.plot(trainer.val_losses, label="val losses")
    plt.legend()

    # Trainer = Trainer.load_trainer(trainer_file)

    train_set = DataLoader(trainset, shuffle=True, batch_size=2048)
    test_set = DataLoader(testset, shuffle=True, batch_size=2048)

    trainer.print_reconstructed_img(testset)

    verifNet = VerificationNet()

    trainer.print_class_coverage_and_predictability(verifNet, trainset)
    trainer.print_class_coverage_and_predictability(verifNet, testset)

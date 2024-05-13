from archives.util import tile_tv_images

# from autoencoder.autoencoder import AutoEncoder
from model.settings import *
from model.data import StackedMNIST

import torch
from torch.utils.data import DataLoader
import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import pickle

class Trainer:
    def __init__(
            self,
            model,
            loss,
            optimizer,
            model_file: str | Path,
            trainer_file: str | Path,
            device = DEVICE,
            force_learn: bool = False
        ) -> None:
        self.file = trainer_file

        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.model_file = model_file
        self.load_weights()
        self.force_relearn = force_learn

    
        self.device = device
        print('device:', device)
        self.model.to(device)

        self.device_not_cpu = not (device == torch.device('cpu'))
        self.losses = []
        self.val_losses = []
        self.load(self)
        
    def load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_file))
            done_training = True

        except:
            print(
                f"Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False

        self.done_training = done_training
    
    def train(
            self, 
            trainset: StackedMNIST, 
            valset: StackedMNIST, 
            epochs: int = 10, 
            batch_size: int = 256
        ) -> bool:
        self.load_weights()

        if self.force_relearn or self.done_training is False:
            train_set = DataLoader(
                trainset, 
                shuffle=True, 
                batch_size=batch_size,
                ) 
            val_set = DataLoader(
                valset, shuffle=True, 
                batch_size=batch_size,
                )
            self.fit(
                train_set=train_set,
                validation_set=val_set,
                epochs=epochs,
                )
            torch.save(self.model.state_dict(), self.model_file)
            self.save()
            self.done_training = True

    def fit(
            self,
            train_set: DataLoader, 
            validation_set: DataLoader, 
            epochs: int = 10,
        ) -> Tuple[object, list, list, object]:
        
        device = self.device
        self.model.to(device)

        for _ in tqdm(range(epochs)):
            val_losses_ep = []
            for _, batch in enumerate(validation_set, 0):
                y_val, outval = self.get_output_from_batch(batch)
                val_losses_ep.append(self.loss(y_val, outval).to("cpu").item())
            
            self.val_losses.append(np.mean(val_losses_ep))

            losses_ep = []
            for _, batch in enumerate(train_set, 0):
                
                y, output = self.get_output_from_batch(batch)

                self.optimizer.zero_grad()
                loss_batch = self.loss(y, output)

                loss_batch.backward()
                self.optimizer.step()
                
                losses_ep.append(loss_batch.to("cpu").item())
            self.losses.append(np.mean(losses_ep))
    
    def get_output_from_batch(self, batch):
        x, y, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)

        return y, output
    
    def print_reconstructed_img(self, dataset, batch_size: int = 25):
        data = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        imgs, _, labels = next(iter(data))

        is_color = imgs.shape[1] == 3

        _, imgs_pred = self.model(imgs.to(self.device))
        labels = labels.detach().numpy()
        imgs_pred = imgs_pred.to("cpu").detach().numpy().reshape(batch_size, 28, 28) if not is_color else imgs_pred.permute(0, 2, 3, 1).to("cpu").detach().numpy()

        tile_tv_images(images=imgs_pred, labels=labels)

    def print_class_coverage_and_predictability(self, VerifNet, dataset, batch_size: int = 10_000, tolerance: float = .8):
        data = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        imgs, _, labels = next(iter(data))

        labels = labels.detach().numpy()
        _, preds = self.model(imgs.to(self.device))

        cov = VerifNet.check_class_coverage(data=imgs, tolerance=tolerance)
        pred, acc = VerifNet.check_predictability(data=preds, correct_labels=labels, tolerance=tolerance)
        print(f"Coverage: {100*cov:.2f}%")
        print(f"Predictability: {100*pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")

    @classmethod
    def load(cls, self):
        with open(self.file, 'rb') as file:
            trainer = pickle.load(file)

        return trainer
    
    def save(self):
        with open(self.file, 'wb') as file:
            pickle.dump(self, file)

        # return trainer
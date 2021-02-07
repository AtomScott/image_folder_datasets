# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['ImageFolderDataModule', 'CNNModel']

# Cell
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, RandomResizedCrop, Normalize
import pytorch_lightning as pl

from fastai.vision.data import ImageDataLoaders

# Cell
class ImageFolderDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = Compose([
            Resize(224),
            RandomResizedCrop(224),
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        data_dir = self.data_dir
        transform = self.transform

        self.dls = ImageDataLoaders.from_folder(data_dir)
        self.trainset = ImageFolder(os.path.join(data_dir, 'train'), transform)
        self.testset = ImageFolder(os.path.join(data_dir, 'valid'), transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def valid_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)

# Cell
class CNNModel(pl.LightningModule):
    def __init__(self, model=None, pretrained=False, log_level=10):
        super().__init__()

        assert model is not None, 'Select model from torchvision'
        self.model = eval(f'torchvision.models.{model}(pretrained={pretrained})')

    def forward(self, x):
        return torch.relu(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
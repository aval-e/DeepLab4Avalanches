import os
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split


class EasyExperiment(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = deeplabv3_resnet50(num_classes=1)
        self.loss = L1Loss()

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return pl.TrainResult(loss)

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     val_loss = self.loss(y_hat, y)
    #     return val_loss
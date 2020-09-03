import torch
from torch.nn import L1Loss, BCELoss
from models.deep_lab_v4 import DeepLabv4
import pytorch_lightning as pl
from pytorch_lightning import TrainResult
from utils import viz_utils


class EasyExperiment(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = DeepLabv4()
        self.loss = BCELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        result = TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        if self.global_step % self.hparams.log_save_interval == 0:
            image = viz_utils.viz_training(x, y, y_hat)
            self.logger.experiment.add_image("Sample", image, self.global_step)
        return result

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     val_loss = self.loss(y_hat, y)
    #     return val_loss
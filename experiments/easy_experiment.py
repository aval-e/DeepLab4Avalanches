import torch
from torch.nn import L1Loss, BCELoss
from models.deep_lab_v4 import DeepLabv4
import pytorch_lightning as pl
from pytorch_lightning import TrainResult, EvalResult
from utils import viz_utils
from argparse import ArgumentParser


class EasyExperiment(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = DeepLabv4(in_channels=hparams.in_channels)
        self.bce_loss = BCELoss()
        self.l1 = L1Loss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.bce_loss(y_hat, y)

        result = TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % self.hparams.log_save_interval == 0:
            image = viz_utils.viz_training(x, y, y_hat)
            self.logger.experiment.add_image("Sample", image, self.global_step)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.round(y_hat) # rounds probability to 0 or 1

        bce_loss = self.bce_loss(y_hat, y)
        l1_loss = self.l1(y_hat, y)
        pred_loss = self.l1(pred, y)

        result = EvalResult()
        result.log('val_bce_loss', bce_loss, sync_dist=True)
        result.log('val_l1_loss', l1_loss, sync_dist=True)
        result.log('pred_loss', pred_loss, sync_dist=True)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimisation algorithm")
        parser.add_argument('--in_channels', type=int, default=4, help="no. of input channels to network")
        return parser

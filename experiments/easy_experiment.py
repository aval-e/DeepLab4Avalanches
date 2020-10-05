import torch
from torch.nn import L1Loss, MSELoss, BCELoss
from models.deep_lab_v4 import DeepLabv4
from jvanvugt_unet.unet import UNet
import pytorch_lightning as pl
from pytorch_lightning import TrainResult, EvalResult
from pytorch_lightning.metrics.functional.classification import auroc
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice
from utils import viz_utils, data_utils
from argparse import ArgumentParser


class EasyExperiment(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.bce_loss = BCELoss()
        self.l1 = L1Loss()
        self.mse = MSELoss()

        if hparams.model == 'unet':
            self.model = UNet(hparams.in_channels, n_classes=1, depth=4, wf=6, padding=True, batch_norm=True)
        elif hparams.model == 'deeplab':
            self.model = DeepLabv4(in_channels=hparams.in_channels)
        else:
            raise('Model not found: ' + hparams.model)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        if self.hparams.optimiser == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimiser == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise Exception('Optimiser not recognised: ' + self.hparams.optimiser)
        
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 600, 1400, 3000, 6000, 10000], gamma=0.5)
        # scheduler = {'scheduler': lr_scheduler,
        #              'interval': 'step'}
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_mask = data_utils.labels_to_mask(y)
        loss = self.bce_loss(y_hat, y_mask)

        result = TrainResult(loss)
        result.log('train loss', loss, on_epoch=True, sync_dist=True)
        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            image = viz_utils.viz_training(x, y, y_hat)
            self.logger.experiment.add_image("Training Sample", image, self.global_step)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.round(y_hat) # rounds probability to 0 or 1
        y_mask = data_utils.labels_to_mask(y)

        bce_loss = self.bce_loss(y_hat, y_mask)
        dice_loss = 1 - soft_dice(y_mask, y_hat)
        precision, recall, f1 = get_precision_recall_f1(y, pred)
        recall1 = recall_for_label(y, pred, 1)
        recall2 = recall_for_label(y, pred, 2)
        recall3 = recall_for_label(y, pred, 3)

        _,_,f1_no_aval = get_precision_recall_f1(y_mask==0, pred==0)
        f1_average = 0.5 * (f1_no_aval + f1)

        # Logging metrics
        result = EvalResult(checkpoint_on=bce_loss)
        result.log('val bce loss', bce_loss, sync_dist=True)
        result.log('val dice', dice_loss, sync_dist=True, reduce_fx=nanmean)
        result.log('precision', precision, sync_dist=True, reduce_fx=nanmean)
        result.log('recall', recall, sync_dist=True, reduce_fx=nanmean)
        result.log('f1 Score', f1, sync_dist=True, reduce_fx=nanmean)
        result.log('recall exact', recall1, sync_dist=True, reduce_fx=nanmean)
        result.log('recall estimated', recall2, sync_dist=True, reduce_fx=nanmean)
        result.log('recall created', recall3, sync_dist=True, reduce_fx=nanmean)
        result.log('f1 average', f1_average, sync_dist=True, reduce_fx=nanmean)
        if batch_idx == self.hparams.val_viz_idx:
            image = viz_utils.viz_training(x, y, y_hat, pred)
            self.logger.experiment.add_image("Validation Sample", image, self.global_step)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='deeplab', help='Model arcitecture. One of "deeplab", "unet"')
        parser.add_argument('--optimiser', type=str, default='adam', help="optimisation algorithm. 'adam' or 'sgd'")
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimisation algorithm")
        parser.add_argument('--momentum', type=float, default=0.9, help="momentum of optimisation algorithm")
        parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay of optimisation algorithm")
        parser.add_argument('--in_channels', type=int, default=4, help="no. of input channels to network")
        parser.add_argument('--train_viz_interval', type=int, default=100, help="image save interval during training")
        parser.add_argument('--val_viz_idx', type=int, default=0, help="batch index to be plotted during validation")
        return parser


def nanmean(x):
    """ Calculate mean ignoring nan values"""
    return x[~torch.isnan(x)].mean()

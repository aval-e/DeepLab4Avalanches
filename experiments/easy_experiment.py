import torch
import argparse
from torch.nn import L1Loss, MSELoss, BCELoss
from models.deep_lab_v4 import DeepLabv4
from segm_models.segmentation_models_pytorch.deeplabv3 import DeepLabV3, DeepLabV3Plus
from models.self_attention_unet import SelfAttentionUNet
from pytorch_lightning import TrainResult, EvalResult, LightningModule
from pytorch_lightning.metrics.functional.classification import auroc
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice
from utils import viz_utils, data_utils
from argparse import ArgumentParser


class EasyExperiment(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # bug in lightning returns hparams as dict when loading from checkpoint
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams

        self.bce_loss = BCELoss()
        self.l1 = L1Loss()
        self.mse = MSELoss()

        if hparams.model == 'deeplab':
            self.model = DeepLabV3('resnet50', in_channels=hparams.in_channels, encoder_weights='imagenet')
        elif hparams.model == 'deeplabv3+':
            self.model = DeepLabV3Plus('resnet50', in_channels=hparams.in_channels, encoder_weights='imagenet')
        elif hparams.model == 'sa_unet':
            self.model = SelfAttentionUNet(hparams.in_channels, 1, depth=4, wf=6, batch_norm=True)
        else:
            raise('Model not found: ' + hparams.model)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        if self.hparams.optimiser == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimiser == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise Exception('Optimiser not recognised: ' + self.hparams.optimiser)

        if self.hparams.lr_scheduler == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.scheduler_steps, gamma=self.hparams.scheduler_gamma)
            scheduler = {'scheduler': lr_scheduler,
                         'interval': 'step'}
            return [optimizer], [scheduler]
        elif self.hparams.lr_scheduler == 'plateau':
            scheduler = {
               'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hparams.scheduler_gamma, patience=2, min_lr=5e-6),
               'interval': 'step',
               'frequency': 250,
               'monitor': 'val_checkpoint_on',
            }
            return [optimizer], [scheduler]
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
            image = viz_utils.viz_training(x, y, y_hat, dem=self.hparams.dem_dir)
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
            image = viz_utils.viz_training(x, y, y_hat, pred, dem=self.hparams.dem_dir)
            self.logger.experiment.add_image("Validation Sample", image, self.global_step)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='deeplab', help='Model arcitecture. One of "deeplab", "deeplabv3+" or "sa_unet"')
        parser.add_argument('--backbone', type=str, default='xception', help='backbone to use in deeplabv3+. "xception", "resnetxx"')

        # optimisation
        parser.add_argument('--optimiser', type=str, default='adam', help="optimisation algorithm. 'adam' or 'sgd'")
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimisation algorithm")
        parser.add_argument('--lr_scheduler', type=str, default=None, help="lr scheduler to be used. ['None', 'multistep', 'plateau']")
        parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='amount by which to decay scheduler lr')
        parser.add_argument('--scheduler_steps', type=int, nargs='+', help='list of steps at which to decrease lr with multistep scheduler')
        parser.add_argument('--momentum', type=float, default=0.9, help="momentum of optimisation algorithm")
        parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay of optimisation algorithm")

        parser.add_argument('--in_channels', type=int, default=4, help="no. of input channels to network")
        parser.add_argument('--train_viz_interval', type=int, default=100, help="image save interval during training")
        parser.add_argument('--val_viz_idx', type=int, default=0, help="batch index to be plotted during validation")
        return parser


def nanmean(x):
    """ Calculate mean ignoring nan values"""
    return x[~torch.isnan(x)].mean()

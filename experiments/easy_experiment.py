import torch
import csv
import os
import warnings
from torch.nn import BCELoss
from torch.nn import functional as F
from modeling.deep_lab_v4 import DeepLabv4
from modeling.avanet import Avanet
from segm_models.segmentation_models_pytorch.deeplabv3 import DeepLabV3, DeepLabV3Plus
from modeling.self_attention_unet import SelfAttentionUNet
from pytorch_lightning import LightningModule
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from utils.data_augmentation import center_crop_batch
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice, crop_to_center, focal_loss, \
    create_loss_weight_matrix, weighted_bce
from utils import viz_utils, data_utils
from utils.utils import nanmean
import argparse
from argparse import ArgumentParser


class EasyExperiment(LightningModule):
    """ Pytorch Lightning Module for easily configuring and running different training experiments with the same logic.
    Everything from the model to the learning rate scheduler and loss function can be set through hparams object
    from argparse. Hyperparameters will automatically be saved to tensorboard and yaml file.
    """

    def __init__(self, hparams):
        super().__init__()

        # Save hyperparameters and make them available through self.hparams
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.val_no = 0

        # loss objects
        self.bce_loss = BCELoss()
        self.register_buffer('edge_weight', create_loss_weight_matrix(hparams.batch_size, hparams.tile_size, distance=100, min_value=0.1))
        self.bce_loss_edges = BCELoss(weight=self.edge_weight)

        # choose model
        if hparams.model == 'deeplab':
            self.model = DeepLabV3(self.hparams.backbone, in_channels=hparams.in_channels, encoder_weights='imagenet')
        elif hparams.model == 'deeplabv3+':
            self.model = DeepLabV3Plus(self.hparams.backbone, in_channels=hparams.in_channels,
                                       encoder_weights='imagenet')
        elif hparams.model == 'deeplabv4':
            self.model = DeepLabv4(self.hparams.backbone, in_channels=hparams.in_channels)
        elif hparams.model == 'avanet':
            self.model = Avanet(in_channels=hparams.in_channels,
                                dem=bool(hparams.dem_dir),
                                backbone=hparams.backbone,
                                decoder=hparams.decoder,
                                replace_stride_with_dilation=hparams.avanet_rep_stride_with_dil,
                                no_blocks=hparams.avanet_no_blocks,
                                deformable=hparams.avanet_deformable,
                                px_per_iter=hparams.avanet_px_per_iter,
                                grad_attention=hparams.avanet_grad_attention,
                                decoder_out_ch=hparams.decoder_out_ch,
                                decoder_dspf_ch=hparams.decoder_dspf_ch,
                                decoder_rates=hparams.decoder_rates,
                                decoder_deformable=hparams.decoder_deformable,
                                )
        elif hparams.model == 'sa_unet':
            self.model = SelfAttentionUNet(hparams.in_channels, 1, depth=4, wf=6, batch_norm=True)
        elif hparams.model == 'mask_rcnn':
            self.model = maskrcnn_resnet50_fpn(False, num_classes=5, trainable_backbone_layers=5,
                                               rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=200,
                                               box_detections_per_img=30, min_size=500,
                                               image_mean=[0, 0, 0], image_std=[1, 1, 1])
        else:
            raise ('Model not found: ' + hparams.model)

    # set up 'test_loss' metric before fit routine starts
    def on_fit_start(self):
        metric_placeholder = {'hp/same_davos_gt': 0,
                              'hp/same_train_gt': 0,
                              'hp/diff_correct': 0,
                              'hp/diff_unkown': 0,
                              'hp/diff_old': 0,
                              'hp/no_correct': 0,
                              'hp/no_wrong': 0,
                              'hp/no_unkown': 0,
                              'hp/no_old': 0,
                              }
        self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def forward(self, x):
        # model may return a list of predictions for deep supervision. If so, apply sigmoid to all of them.
        x = self.model(x)
        if isinstance(x, list):
            return [torch.sigmoid(item) for item in x]
        return torch.sigmoid(x)

    def configure_optimizers(self):
        if self.hparams.optimiser == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimiser == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise Exception('Optimiser not recognised: ' + self.hparams.optimiser)

        if self.hparams.lr_scheduler == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.scheduler_steps,
                                                                gamma=self.hparams.scheduler_gamma)
            scheduler = {'scheduler': lr_scheduler,
                         'interval': 'epoch'}
            return [optimizer], [scheduler]
        elif self.hparams.lr_scheduler == 'plateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hparams.scheduler_gamma,
                                                                        patience=2, min_lr=2e-3),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'loss/bce',
            }
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats = self(x)
        y_mask = data_utils.labels_to_mask(y)

        # y_hat expected to be a list with output at index 0 and intermediate predictions afterwards
        total_loss = torch.tensor(0.0)
        if not isinstance(y_hats, list):
            y_hats = [y_hats]
        for k, y_hat in enumerate(y_hats):
            y_hat = F.interpolate(y_hat, x.shape[2:], mode='bilinear', align_corners=True)

            if self.hparams.loss == 'bce':
                loss = self.bce_loss(y_hat, y_mask)
            elif self.hparams.loss == 'focal':
                loss = focal_loss(y_hat, y_mask)
            elif self.hparams.loss == 'weighted_bce':
                loss = weighted_bce(y_hat, y_mask, y, self.edge_weight)
            elif self.hparams.loss == 'bce_edges':
                loss = self.bce_loss_edges(y_hat, y_mask)
            elif self.hparams.loss == 'soft_dice':
                loss = soft_dice(y_mask, y_hat)
            else:
                warnings.warn('no such loss defined: ' + self.hparams.loss)

            multiplier = 1.0 if k == 0 else 0.25
            total_loss = total_loss + multiplier * loss

        self.log('train_loss/bce', total_loss, on_epoch=True, sync_dist=True)
        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            fig = viz_utils.viz_predictions(x, y, y_hats[0], dem=self.hparams.dem_dir, fig_size=2)
            self.logger.experiment.add_figure("Training Sample", fig, self.global_step)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        pred = torch.round(y_hat)  # rounds probability to 0 or 1

        # evaluate metrics only on center patch
        y_hat_crop = crop_to_center(y_hat)
        y_crop = crop_to_center(y)
        pred_crop = crop_to_center(pred)
        y_mask_crop = data_utils.labels_to_mask(y_crop)

        bce_loss = self.bce_loss(y_hat_crop, y_mask_crop)
        dice_score = soft_dice(y_mask_crop, y_hat_crop)
        precision, recall, f1 = get_precision_recall_f1(y_crop, pred_crop)
        recall1 = recall_for_label(y_crop, pred_crop, 1)
        recall2 = recall_for_label(y_crop, pred_crop, 2)
        recall3 = recall_for_label(y_crop, pred_crop, 3)

        _, _, f1_no_aval = get_precision_recall_f1(y_mask_crop == 0, pred_crop == 0)
        f1_average = 0.5 * (f1_no_aval + f1)

        # Logging metrics
        self.log('loss/bce', bce_loss, sync_dist=True)
        self.log('f1/a_soft_dice', dice_score, sync_dist=True, reduce_fx=nanmean)
        self.log('f1/avalanche', f1, sync_dist=True, reduce_fx=nanmean)
        self.log('f1/average', f1_average, sync_dist=True, reduce_fx=nanmean)
        self.log('pr/precision', precision, sync_dist=True, reduce_fx=nanmean)
        self.log('pr/recall', recall, sync_dist=True, reduce_fx=nanmean)
        self.log('recall/exact', recall1, sync_dist=True, reduce_fx=nanmean)
        self.log('recall/estimated', recall2, sync_dist=True, reduce_fx=nanmean)
        self.log('recall/created', recall3, sync_dist=True, reduce_fx=nanmean)
        if batch_idx == self.hparams.val_viz_idx:
            self.val_no += 1
            if self.val_no % self.hparams.val_viz_interval == 0:
                fig = viz_utils.viz_predictions(x, y, y_hat, pred, dem=self.hparams.dem_dir, fig_size=2)
                self.logger.experiment.add_figure("Validation Sample", fig, self.global_step)
        return bce_loss

    def test_step(self, batch, batch_idx):
        """ Check against ground truth obtained by other means - 'Methodenvergleich' """
        img, _, mapped, status, id = batch
        y_hat = self(img)

        # aval detected if average in 10px patch around point is bigger than 0.5 threshold
        y_hat = center_crop_batch(y_hat, crop_size=10)
        pred = y_hat.mean(dim=[1, 2, 3]) > 0.5

        # check which predictions are the same
        different = pred != mapped
        correct_true = pred * (status == 1)
        correct_false = ~pred * (status == 3)
        wrong_true = ~pred * (status == 1)
        wrong_false = pred * (status == 3)
        correct = correct_true + correct_false
        wrong = wrong_true + wrong_false
        diff_correct = correct * different
        diff_wrong = wrong * different
        diff_unkown = (status == 2) * different
        diff_old = (status == 5) * different

        same_davos_gt = torch.sum(correct).float() / torch.sum(correct + wrong)
        same_train_gt = torch.mean((~different).float())
        correct_score = (torch.sum(diff_correct) - torch.sum(diff_wrong)).float() / torch.sum(diff_correct + diff_wrong)
        unkown_score = (torch.sum(diff_unkown * pred) - torch.sum(diff_unkown * ~pred)).float() / torch.sum(diff_unkown)
        old_score = (torch.sum(diff_old * pred) - torch.sum(diff_old * ~pred)).float() / torch.sum(diff_old)

        # log in the form of tensorboard hyperparameter metrics
        self.log('hp/same_davos_gt', same_davos_gt, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/same_train_gt', same_train_gt, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_correct', correct_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_unkown', unkown_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_old', old_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/no_correct', torch.sum(diff_correct), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_wrong', torch.sum(diff_wrong), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_unkown', torch.sum(diff_unkown), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_old', torch.sum(diff_old), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)

        # remember id's for checking difference later
        ids = {'ids_diff_old': id[diff_old].tolist(),
               'ids_diff_unkown': id[diff_unkown].tolist(),
               'ids_diff_correct': id[diff_correct].tolist(),
               'ids_diff_wrong': id[diff_wrong].tolist()}

        return ids

    def test_epoch_end(self, outputs):
        aggr_outputs = {}
        for key in outputs[0]:
            aggr_outputs[key] = []
            for el in outputs:
                aggr_outputs[key].extend(el[key])

        if not os.path.exists(self.logger.log_dir):
            warnings.warn('Log dir not found. Dumping to stdout instead of csv:\n ')
            print(aggr_outputs)
            return

        # Save results to csv file
        csv_name = os.path.join(self.logger.log_dir, 'davos_test_IDs.csv')
        print('Saving test ids to: ' + csv_name)
        with open(csv_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in aggr_outputs.items():
                line = [key]
                line.extend([str(v) for v in value])
                writer.writerow(line)
        csv_file.close()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ allows adding model specific args via command line and logging them
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='deeplab',
                            help='Model architecture. One of "deeplab", "deeplabv3+", "avanet" or "sa_unet"')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone to use in deeplabv3+. "xception", "resnetxx"')
        parser = Avanet.add_model_specific_args(parser)

        # optimisation
        parser.add_argument('--loss', type=str, default='bce', help='Type of loss to use for training')
        parser.add_argument('--optimiser', type=str, default='adam', help="optimisation algorithm. 'adam' or 'sgd'")
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimisation algorithm")
        parser.add_argument('--lr_scheduler', type=str, default=None,
                            help="lr scheduler to be used. ['None', 'multistep', 'plateau']")
        parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='amount by which to decay scheduler lr')
        parser.add_argument('--scheduler_steps', type=int, nargs='+',
                            help='list of steps at which to decrease lr with multistep scheduler')
        parser.add_argument('--momentum', type=float, default=0.9, help="momentum of optimisation algorithm")
        parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay of optimisation algorithm")

        parser.add_argument('--in_channels', type=int, default=4, help="no. of input channels to network")
        parser.add_argument('--train_viz_interval', type=int, default=100, help="image save interval during training")
        parser.add_argument('--val_viz_idx', type=int, default=0, help="batch index to be plotted during validation")
        parser.add_argument('--val_viz_interval', type=int, default=1, help='how often to save validation image')
        return parser

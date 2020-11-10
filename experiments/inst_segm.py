import torch
from experiments.easy_experiment import EasyExperiment
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice
from utils import viz_utils, data_utils
from argparse import ArgumentParser


class InstSegmentation(EasyExperiment):

    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs)
        return x

    def training_step(self, batch, batch_idx):
        x, targets = batch
        losses = self(x, targets)
        # dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])

        loss = sum(loss for loss in losses.values())

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('loss_classifier', losses['loss_classifier'], on_epoch=True, sync_dist=True)
        self.log('loss_box_reg', losses['loss_box_reg'], on_epoch=True, sync_dist=True)
        self.log('loss_mask', losses['loss_mask'], on_epoch=True, sync_dist=True)
        self.log('loss_objectness', losses['loss_objectness'], on_epoch=True, sync_dist=True)
        self.log('loss_rpn_box_reg', losses['loss_rpn_box_reg'], on_epoch=True, sync_dist=True)

        # Log random images
        # if self.global_step % self.hparams.train_viz_interval == 0:
        #     fig = viz_utils.viz_predictions(x, y, y_hat, dem=self.hparams.dem_dir, fig_size=2)
        #     self.logger.experiment.add_figure("Training Sample", fig, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        outputs = self(x)

        self.log('val_loss', outputs[0]['scores'].sum())

        img = x[0].unsqueeze(dim=0)
        all_masks_gt = torch.sum(targets[0]['masks'], dim=0, keepdim=True).unsqueeze(dim=0)
        all_masks = torch.sum(outputs[0]['masks'], dim=0, keepdim=True)
        all_preds = all_masks.round()

        # pred = torch.round(y_hat)  # rounds probability to 0 or 1
        # y_mask = data_utils.labels_to_mask(y)
        #
        # bce_loss = self.bce_loss(y_hat, y_mask)
        # dice_score = soft_dice(y_mask, y_hat)
        # precision, recall, f1 = get_precision_recall_f1(y, pred)
        # recall1 = recall_for_label(y, pred, 1)
        # recall2 = recall_for_label(y, pred, 2)
        # recall3 = recall_for_label(y, pred, 3)
        #
        # _, _, f1_no_aval = get_precision_recall_f1(y_mask == 0, pred == 0)
        # f1_average = 0.5 * (f1_no_aval + f1)
        #
        # # Logging metrics
        # self.log('loss/bce', bce_loss, sync_dist=True)
        # self.log('f1/a_soft_dice', dice_score, sync_dist=True, reduce_fx=nanmean)
        # self.log('f1/avalanche', f1, sync_dist=True, reduce_fx=nanmean)
        # self.log('f1/average', f1_average, sync_dist=True, reduce_fx=nanmean)
        # self.log('pr/precision', precision, sync_dist=True, reduce_fx=nanmean)
        # self.log('pr/recall', recall, sync_dist=True, reduce_fx=nanmean)
        # self.log('recall/exact', recall1, sync_dist=True, reduce_fx=nanmean)
        # self.log('recall/estimated', recall2, sync_dist=True, reduce_fx=nanmean)
        # self.log('recall/created', recall3, sync_dist=True, reduce_fx=nanmean)
        if batch_idx == self.hparams.val_viz_idx:
            self.val_no += 1
            if self.val_no % self.hparams.val_viz_interval == 0:
                fig = viz_utils.viz_predictions(img, all_masks_gt, all_masks, all_preds, dem=self.hparams.dem_dir, fig_size=2)
                self.logger.experiment.add_figure("Validation Sample", fig, self.global_step)
        return None


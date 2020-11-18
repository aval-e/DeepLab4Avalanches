import torch
from experiments.easy_experiment import EasyExperiment
from utils.data_augmentation import center_crop_batch
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice
from utils import viz_utils, data_utils
from utils.utils import nanmean


class InstSegmentation(EasyExperiment):

    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs)
        return x

    def training_step(self, batch, batch_idx):
        x, targets = batch
        losses = self(x, targets)

        loss = sum(loss for loss in losses.values())

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('losses/classifier', losses['loss_classifier'], on_epoch=True, sync_dist=True)
        self.log('losses/box_reg', losses['loss_box_reg'], on_epoch=True, sync_dist=True)
        self.log('losses/mask', losses['loss_mask'], on_epoch=True, sync_dist=True)
        self.log('losses/objectness', losses['loss_objectness'], on_epoch=True, sync_dist=True)
        self.log('losses/rpn_box_reg', losses['loss_rpn_box_reg'], on_epoch=True, sync_dist=True)

        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            self.eval()
            outputs = self(x)
            self.train()
            fig = viz_utils.viz_aval_instances(x, targets, outputs, dem=self.hparams.dem_dir, fig_size=2)
            self.logger.experiment.add_figure("Training Sample", fig, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        outputs = self(x)

        bce_loss = self.log_and_viz_inst(batch_idx, x, targets, outputs)
        return bce_loss

    def log_and_viz_inst(self, batch_idx, x, targets, outputs):
        masks = []
        for target in targets:
            if target['masks'].numel() == 0:  # check if tensor is empty
                mask = target['masks']
                masks.append(torch.zeros([1, mask.shape[1], mask.shape[2]], dtype=mask.dtype, layout=mask.layout,
                                         device=mask.device))
            else:
                masks.append(target['masks'].max(dim=0, keepdim=True)[0])
        y = torch.stack(masks, dim=0)

        masks = []
        for output in outputs:
            if output['masks'].numel() == 0:
                mask = output['masks']
                masks.append(torch.zeros([1, mask.shape[2], mask.shape[3]], dtype=mask.dtype, layout=mask.layout, device=mask.device))
            else:
                masks.append(output['masks'].squeeze(dim=1).max(dim=0, keepdim=True)[0])
        y_hat = torch.stack(masks, dim=0)

        pred = torch.round(y_hat)  # rounds probability to 0 or 1
        y_mask = data_utils.labels_to_mask(y)

        bce_loss = self._calc_and_log_val_losses(y, y_mask, y_hat, pred)

        if batch_idx == self.hparams.val_viz_idx:
            self.val_no += 1
            if self.val_no % self.hparams.val_viz_interval == 0:
                fig = viz_utils.viz_aval_instances(x, targets, outputs, dem=self.hparams.dem_dir, fig_size=2)
                self.logger.experiment.add_figure("Validation Sample", fig, self.global_step)

        return bce_loss

    def test_step(self, batch, batch_idx):
        """ Check against ground truth obtained by other means - 'Methodenvergleich' """
        x, _, mapped, status, id = batch

        # bring into format as needed for masked rcnn
        x = [sample for sample in x]
        outputs = self(x)

        masks = [output['masks'].squeeze().max(dim=0, keepdim=True)[0] for output in outputs]
        y_hat = torch.stack(masks, dim=0)

        return self.log_test_results(y_hat, mapped, status, id)

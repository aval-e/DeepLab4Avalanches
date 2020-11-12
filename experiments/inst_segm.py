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
        # dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])

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

        masks = []
        for target in targets:
            if target['masks'].numel() == 0:  # check if tensor is empty
                mask = target['masks']
                masks.append(torch.zeros([1, mask.shape[1], mask.shape[2]], dtype=mask.dtype, layout=mask.layout, device=mask.device))
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

        bce_loss = self.bce_loss(y_hat, y_mask)
        dice_score = soft_dice(y_mask, y_hat)
        precision, recall, f1 = get_precision_recall_f1(y, pred)
        recall1 = recall_for_label(y, pred, 1)
        recall2 = recall_for_label(y, pred, 2)
        recall3 = recall_for_label(y, pred, 3)

        _, _, f1_no_aval = get_precision_recall_f1(y_mask == 0, pred == 0)
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
                fig = viz_utils.viz_aval_instances(x, targets, outputs, dem=self.hparams.dem_dir, fig_size=2)
                self.logger.experiment.add_figure("Validation Sample", fig, self.global_step)
        return None

    def test_step(self, batch, batch_idx):
        """ Check against ground truth obtained by other means - 'Methodenvergleich' """
        x, _, mapped, status, id = batch

        # bring into format as needed for masked rcnn
        x = [sample for sample in x]
        outputs = self(x)

        masks = [output['masks'].squeeze().max(dim=0, keepdim=True)[0] for output in outputs]
        y_hat = torch.stack(masks, dim=0)

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
        self.log('hp/same_davos_gt', same_davos_gt, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/same_train_gt', same_train_gt, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_correct', correct_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_unkown', unkown_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/diff_old', old_score, sync_dist=True, reduce_fx=nanmean)
        self.log('hp/no_correct', torch.sum(diff_correct), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_wrong', torch.sum(diff_wrong), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_unkown', torch.sum(diff_unkown), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)
        self.log('hp/no_old', torch.sum(diff_old), sync_dist=True, reduce_fx=torch.sum, sync_dist_op=None)

        ids = {'ids_diff_old': id[diff_old].tolist(),
               'ids_diff_unkown': id[diff_unkown].tolist(),
               'ids_diff_correct': id[diff_correct].tolist(),
               'ids_diff_wrong': id[diff_wrong].tolist()}

        return ids

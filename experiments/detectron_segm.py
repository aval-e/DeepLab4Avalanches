import torch
from experiments.inst_segm import InstSegmentation
from datasets.detectron2_dataset import detectron_targets_to_torchvision, detectron_preds_to_torchvision
from detectron2.utils.events import EventStorage
from utils.data_augmentation import center_crop_batch
from utils.losses import get_precision_recall_f1, recall_for_label, soft_dice
from utils import viz_utils, data_utils
from utils.utils import nanmean


class DetectronSegmentation(InstSegmentation):

    def forward(self, *args, **kwargs):
        with EventStorage() as storage:
            x = self.model(*args, **kwargs)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        losses = self(x)

        targets = [detectron_targets_to_torchvision(sample['instances']) for sample in x]

        print('Targets:')
        print(targets)
        print('\nLosses:')
        print(losses)


        loss = sum(loss for loss in losses.values())

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)

        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            self.eval()
            outputs = self(x)
            self.train()
            fig = viz_utils.viz_aval_instances(x, targets, outputs, dem=self.hparams.dem_dir, fig_size=2)
            self.logger.experiment.add_figure("Training Sample", fig, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)

        targets = [detectron_targets_to_torchvision(sample['instances']) for sample in x]
        outputs = [detectron_preds_to_torchvision(output['instances']) for output in outputs]

        imgs = [el['image'] for el in x]

        bce_loss = self.log_and_viz_inst(batch_idx, imgs, targets, outputs)
        return bce_loss

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

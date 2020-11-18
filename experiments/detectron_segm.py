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
        train_loss = sum(loss for loss in losses.values())

        targets = [detectron_targets_to_torchvision(sample['instances']) for sample in x]

        self.log('train_loss', train_loss, on_epoch=True, sync_dist=True)
        for key, item in losses.items():
            self.log('losses/' + key, item.item(), on_epoch=True, sync_dist=True)


        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            self.eval()
            outputs = self(x)
            self.train()
            imgs = [el['image'] for el in x]
            outputs = [detectron_preds_to_torchvision(output['instances']) for output in outputs]
            fig = viz_utils.viz_aval_instances(imgs, targets, outputs, dem=self.hparams.dem_dir, fig_size=2)
            self.logger.experiment.add_figure("Training Sample", fig, self.global_step)
        return train_loss

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
        x = [{'image': sample} for sample in x]
        outputs = self(x)

        outputs = [detectron_preds_to_torchvision(output['instances']) for output in outputs]

        masks = [output['masks'].squeeze().max(dim=0, keepdim=True)[0] for output in outputs]
        y_hat = torch.stack(masks, dim=0)

        return self.log_test_results(y_hat, mapped, status, id)

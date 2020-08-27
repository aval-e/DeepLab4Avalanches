import torch
from torch.nn import L1Loss, BCELoss, Sigmoid
from torchvision.models.segmentation import deeplabv3_resnet50
import pytorch_lightning as pl
from pytorch_lightning import TrainResult
from utils import viz_utils


class EasyExperiment(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=1)
        self.sigmoid = Sigmoid()
        self.loss = BCELoss()

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self.sigmoid(self(x))
        loss = self.loss(y_hat, y)

        result = TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        image = viz_utils.viz_training(x, y, y_hat)
        self.logger.experiment.add_image("Sample", image, self.current_epoch)
        return result

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     val_loss = self.loss(y_hat, y)
    #     return val_loss
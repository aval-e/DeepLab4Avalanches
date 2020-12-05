import torch
import warnings
from kornia.filters.sobel import SpatialGradient
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder, SeparableConv2d
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from torchvision.models.resnet import Bottleneck
from modeling.backbones.avanet_backbone import AvanetBackbone


class Avanet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_grad = SpatialGradient()
        self.dem_bn = torch.nn.BatchNorm2d(1)

        self.encoder = AvanetBackbone()
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(4, 8, 12),
            output_stride=16,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=1,
            upsampling=2,
        )
        self.classification_head = None

    def _init_weights(self):
        warnings.warn('Avanet init weights function not implemented yet...')
        return

    def forward(self, x):
        # calculate dem gradients and magnitude
        dem = x[:, [-1], :, :]
        x = x[:, :-1, :, :]
        dem_grads = self.spatial_grad(dem).squeeze(dim=1)
        dem = torch.sqrt(torch.square(dem_grads[:, [0], :, :]) + torch.square(dem_grads[:, [1], :, :]))
        dem = self.dem_bn(dem)
        x = torch.cat([x, dem], dim=1)

        x = self.encoder(x, dem_grads)
        x = self.decoder(*x)
        x = self.segmentation_head(x)
        return x
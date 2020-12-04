import torch
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder, SeparableConv2d
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from torchvision.models.resnet import Bottleneck
from modeling.backbones.avanet_backbone import AvanetBackbone
from modeling.reusable_blocks import DeformableBlock


class Avanet(SegmentationModel):
    def __init__(self):
        super().__init__()
        self.encoder = AvanetBackbone(iterations=50)
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(8, 16, 24),
            output_stride=8,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )
        self.classification_head = None

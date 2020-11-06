import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn


class DeepLabv4(torch.nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super(DeepLabv4, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(num_classes=num_classes)
        self.deeplabv3.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.deeplabv3(x)['out']
        return x
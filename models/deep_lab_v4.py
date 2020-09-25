import torch
from torch.nn import Sigmoid
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabv4(torch.nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super(DeepLabv4, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(num_classes=num_classes)
        self.deeplabv3.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.deeplabv3(x)['out']
        return x

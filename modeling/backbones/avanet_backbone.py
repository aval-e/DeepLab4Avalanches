import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from modeling.reusable_blocks import Bottleneck, DeformableBlock, conv1x1
from kornia.filters.sobel import SpatialGradient


class AvanetBackbone(nn.Module):

    def __init__(self, groups=1, width_per_group=64, norm_layer=None):
        super(AvanetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 62, kernel_size=7, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        ch = [3, 64, 128, 256, 512, 512]
        self.out_channels = ch

        self.layer1 = self._make_layer(ch[1], ch[2], 3, stride=2)
        self.layer2 = self._make_layer(ch[2], ch[3], 3, stride=2, deformable=True)
        self.layer3 = self._make_layer(ch[3], ch[4], 3, stride=2, deformable=True)
        self.layer4 = self._make_layer(ch[4], ch[5], 2, stride=2, deformable=True)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1, deformable=False):
        norm_layer = self._norm_layer
        block = Bottleneck if not deformable else DeformableBlock

        layers = []
        layers.append(block(inplanes, planes, stride, self.groups,
                            self.base_width, 1, norm_layer))
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes, planes, groups=self.groups,
                                     base_width=self.base_width, dilation=1,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, grads):
        features = [nn.Identity()]

        x = self.conv1(x)
        x = torch.cat([x, grads], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features
